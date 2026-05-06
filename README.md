# LLM Inference Pipeline Accelerator

**A memory-bound architecture and performance study for single-token autoregressive decode.**

> Course project — Computer Architecture, CSULB  
> Authors: Koushik Ravikumar · Dhathresh Prathap Kora

---

## Overview

This repository implements a **cycle-level analytical simulator** for a pipeline-aware LLM inference accelerator. It models every stage of a decoder-only transformer's single-token decode step — from embedding lookup through sampling — and quantifies how memory bandwidth, KV-cache traffic, and stage scheduling interact to determine end-to-end latency.

Key findings:
- All 16 decode stages have arithmetic intensity < 1 FLOP/byte → **memory-bandwidth-limited**, not compute-limited
- QKV projection + attention front-end accounts for **~50.5 %** of total cycles
- INT8 weight quantization yields a **1.85× speedup** by halving DRAM bandwidth
- Kernel fusion (FlashAttention + fused QKV + MLP) can reduce memory traffic by **5–8×**

---

## Repository Structure

```
llm_inference_pipeline_accelerator/
├── llm_inference_accel/          # Core Python package
│   ├── config.py                 # ModelConfig, HardwareConfig, PrecisionConfig dataclasses
│   ├── model.py                  # NumPy functional reference model + KV-cache
│   ├── accelerator.py            # Event-driven cycle-level simulator
│   ├── experiments.py            # Figure/table generation for the report
│   └── host.py                   # Host detection + NumPy kernel calibration (M2/M4)
│
├── run_demo.py                   # Quick demo: decode a few tokens + print analytic latency
├── tests.py                      # Correctness and trend checks (all 4 must pass)
├── reproduce.py                  # Regenerates all 7 figures + 6 CSV tables
├── benchmark_host.py             # Machine-calibrated benchmark (run once per host)
├── compare_host_runs.py          # Compare results from two hosts side-by-side
├── build_report.py               # Compile results into a Word/PDF report
│
├── results/                      # Pre-generated outputs (committed for reproducibility)
│   ├── figures/                  # Pipeline diagram, latency breakdown, KV footprint, etc.
│   ├── tables/                   # CSV tables for all experiments
│   └── validation/               # Numerical validation results
│
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Koushik-Ravikumar/CECS-530-LLM-Inference-Pipeline-Accelerator.git
cd CECS-530-LLM-Inference-Pipeline-Accelerator
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `matplotlib`, `python-docx`, `Pillow`  
Python ≥ 3.10 recommended.

### 4. Run tests

```bash
python tests.py
```

Expected output: **8/8 tests pass**, `max_err < 1e-4`.

### 5. Run the demo

```bash
python run_demo.py
```

Runs a small functional decode and prints the analytic latency estimate for a baseline hardware configuration.

### 6. Reproduce all figures and tables

```bash
python reproduce.py --output-dir results
```

Regenerates:
- `results/figures/` — 7 PNG figures (pipeline diagram, architecture diagram, latency breakdown, latency vs sequence length, KV footprint, precision comparison, model size sensitivity)
- `results/tables/` — 6 CSV tables
- `results/validation/validation.json` — numerical validation report

---

## Host-Calibrated Benchmarking (M2 vs M4)

The host-calibrated path measures actual NumPy throughput on your machine (projection matvec, MLP matvec, attention score/value, vector ops, DMA copy) and feeds those rates into the analytical model. This means the **same model produces different latency estimates on M2 and M4**.

**Step 1: Run once on each machine**

```bash
# On your M2 Mac
python benchmark_host.py --label m2 --output-dir results/host_runs

# On your M4 Mac
python benchmark_host.py --label m4 --output-dir results/host_runs
```

Each run writes a folder containing:
- `summary.json` — calibration + derived `HardwareConfig`
- `latency_vs_sequence.csv` / `.png`
- `kernel_throughput.png`
- `host_metrics.csv`

**Step 2: Compare the two hosts**

```bash
python compare_host_runs.py \
  results/host_runs/m2/summary.json \
  results/host_runs/m4/summary.json \
  --output-dir results/host_comparison
```

Outputs: `headline_comparison.csv`, `latency_comparison.png`, `throughput_comparison.png`, `comparison_summary.json`.

---

## Architecture

### Functional reference model (`model.py`)

- Single-token incremental decode with KV-cache (append + prefix read)
- Grouped Query Attention (GQA) support
- Full-sequence causal forward pass for numerical validation
- Greedy and top-k sampling

### Cycle-level simulator (`accelerator.py`)

The simulator builds an **event-driven task graph** for each decode step and schedules it across 5 independent engines:

| Engine | Handles |
|--------|---------|
| `projection` | QKV, output projection, LM head |
| `attention` | QKᵀ score computation, α·V value mix |
| `mlp` | FC1 (up+gate), SwiGLU, FC2 (down) |
| `vector` | LayerNorm, Softmax, Residual adds, Sampling |
| `dma` | Weight streaming, KV-cache read/write |

DMA tasks run **concurrently** with compute tasks, hiding memory latency behind useful work. A **current-token K/V bypass** path avoids write-then-read stalls.

### Default baseline configuration

| Parameter | Value |
|-----------|-------|
| Clock | 1.0 GHz |
| Projection MACs/cycle | 2048 |
| MLP MACs/cycle | 2048 |
| Attention MACs/cycle | 1024 |
| DMA bytes/cycle | 256 |
| Activation precision | BF16 |
| Weight precision | INT8 |
| KV-cache precision | BF16 |
| Accumulator | FP32 |

### Custom configuration example

```python
from llm_inference_accel.config import ModelConfig, HardwareConfig, PrecisionConfig
from llm_inference_accel.accelerator import DecodeAcceleratorModel

model = ModelConfig(
    vocab_size=32000, max_seq_len=8192,
    num_layers=24, d_model=512,
    num_heads=8, num_kv_heads=8, ffn_dim=2048
)
hw = HardwareConfig(projection_macs_per_cycle=4096, dma_bytes_per_cycle=512)
prec = PrecisionConfig(weight_bytes=1, weight_name="INT8")

accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)
result = accel.simulate_token(seq_len=2048)

print(f"Latency : {result.total_time_ns / 1e6:.3f} ms/token")
print(f"Throughput: {1e9 / result.total_time_ns:.0f} tokens/s")
print("Top bottlenecks:", result.top_critical_stages(k=3))
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Analytic latency (baseline) | **1.11 ms / token** |
| Throughput | **901 tokens/s** |
| Dominant stage | QKV + attention front-end (50.5 %) |
| Second-largest stage | MLP (35.0 %) |
| INT8 vs BF16 speedup | **1.85×** |
| Kernel fusion potential | **5–8× traffic reduction** |

KV-cache traffic overtakes weight streaming at **ctx ≈ 4096 tokens**, becoming the sole bottleneck for long-context inference.

---

## Limitations

- Functional model does not include rotary positional embeddings (RoPE), tensor parallelism, or speculative decoding.
- The accelerator model is **analytical**, not RTL. It captures latency trends and bottlenecks, not exact silicon timing.
- Host-calibrated benchmarking measures NumPy throughput, not direct Metal / GPU throughput.
- Weight tiling is abstracted as DMA tasks; no physical floorplanning or bank-conflict model is included.

---

## License

For academic use only. All rights reserved by the authors.
