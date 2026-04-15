# LLM Inference Pipeline Accelerator

This project implements a **single-token decode accelerator model** for a decoder-only transformer. The package combines:

1. A **functional reference model** with correct KV-cache handling.
2. A **cycle-level analytical simulator** that models stage latency, DMA overlap, and cache streaming.
3. **Reproducible experiments** that generate figures and tables for the report.
4. A **host-calibrated benchmarking path** so you can compare runs on different Macs such as **M2** and **M4** using the local machine's measured throughput.

## Project structure

- `llm_inference_accel/config.py` - model, precision, and hardware configuration dataclasses.
- `llm_inference_accel/model.py` - reference transformer layer(s), incremental decode, full-sequence validation, KV-cache implementation.
- `llm_inference_accel/accelerator.py` - event-driven cycle model for single-token decode.
- `llm_inference_accel/experiments.py` - charts, tables, and summary generation.
- `llm_inference_accel/host.py` - local host detection and NumPy-based calibration.
- `run_demo.py` - small demo for functional decode plus analytical latency estimate.
- `benchmark_host.py` - detect the local machine and generate host-dependent results.
- `compare_host_runs.py` - compare summaries from two or more machines.
- `reproduce.py` - regenerates all report figures/tables.
- `tests.py` - correctness and trend checks.

## Quick start

```bash
python tests.py
python run_demo.py
python reproduce.py --output-dir results
```

## Host-dependent M2 vs M4 comparison

Run this once on each Mac:

```bash
python benchmark_host.py --label m2 --output-dir results/host_runs
python benchmark_host.py --label m4 --output-dir results/host_runs
```

Each run writes a folder containing:

- `summary.json`
- `latency_vs_sequence.csv`
- `latency_vs_sequence.png`
- `kernel_throughput.png`
- `host_metrics.csv`

Then compare the runs together:

```bash
python compare_host_runs.py \
  results/host_runs/m2/summary.json \
  results/host_runs/m4/summary.json \
  --output-dir results/host_comparison
```

This creates:

- `headline_comparison.csv`
- `latency_comparison.csv`
- `latency_comparison.png`
- `throughput_comparison.png`
- `comparison_summary.json`

## What the host-calibrated path does

The host-calibrated runner measures local NumPy throughput for kernels that resemble decode-time work:

- projection-style matvec
- MLP matvec
- attention score/value streaming
- vector elementwise work
- memory copy bandwidth

Those measured rates are converted into a `HardwareConfig`, which means the same analytical model produces **different latency estimates on M2 and M4** based on the machine that actually runs the code.

## What the accelerator model covers

The simulator explicitly models the following stages:

- token embedding lookup
- LayerNorm
- fused QKV projection
- KV-cache write
- attention score generation
- softmax
- attention value mixing
- output projection and residual connection
- second LayerNorm
- SwiGLU MLP up/gate and down projection
- output projection to vocabulary and token sampling

## Baseline microarchitecture

- vector-style GEMV datapaths, chosen because batch-1 decode behaves like GEMV rather than large GEMM
- separate projection, attention, MLP, and vector engines
- software-managed scratchpad and double-buffered DMA
- current-token K/V bypass path to avoid write-then-read bubbles
- BF16 activations and KV cache, INT8 weights by default in the performance model

## Reproducibility notes

- Functional validation uses a **small model** so it runs quickly in NumPy.
- Performance studies use a **larger analytical configuration** without instantiating all weights in memory.
- Host-calibrated results depend on the machine, Python version, NumPy build, and BLAS backend installed on that machine.

## Limitations

- The functional model is intentionally compact and does not include rotary embeddings, tensor parallelism, or speculative decoding.
- The accelerator model is analytical, not RTL. It aims to capture latency trends and bottlenecks rather than exact silicon timing.
- The host-calibrated path is a **machine-aware calibration**, not a direct Metal or GPU benchmark.
- Weight tiling is abstracted as DMA tasks; no physical floorplanning or bank-conflict model is included.
