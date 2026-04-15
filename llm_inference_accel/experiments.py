from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .accelerator import DecodeAcceleratorModel
from .config import HardwareConfig, ModelConfig, PrecisionConfig
from .model import initialize_weights, validate_incremental_decode


FIG_DPI = 180


def mib(nbytes: float) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def kb(nbytes: float) -> float:
    return float(nbytes) / 1024.0


def cycles_to_us(cycles: float, clock_ghz: float) -> float:
    return cycles / (clock_ghz * 1_000.0)


def default_small_validation_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        num_layers=2,
        d_model=128,
        num_heads=4,
        num_kv_heads=4,
        ffn_dim=512,
    )


def default_analytic_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        max_seq_len=8192,
        num_layers=24,
        d_model=512,
        num_heads=8,
        num_kv_heads=8,
        ffn_dim=2048,
    )


def default_hardware_config() -> HardwareConfig:
    return HardwareConfig(
        projection_macs_per_cycle=2048,
        mlp_macs_per_cycle=2048,
        attention_macs_per_cycle=1024,
        vector_ops_per_cycle=256,
        dma_bytes_per_cycle=256,
        compute_efficiency=0.80,
        dma_efficiency=0.85,
        scratchpad_bytes=2 * 1024 * 1024,
        weight_buffer_bytes=512 * 1024,
        kv_stream_buffer_bytes=512 * 1024,
        clock_ghz=1.0,
        use_bypass_for_current_kv=True,
    )


def default_precision_int8_weights() -> PrecisionConfig:
    return PrecisionConfig(
        activation_bytes=2,
        weight_bytes=1,
        kv_bytes=2,
        accumulator_bytes=4,
        activation_name="BF16",
        weight_name="INT8",
        kv_name="BF16",
    )


def bf16_everywhere_precision() -> PrecisionConfig:
    return PrecisionConfig(
        activation_bytes=2,
        weight_bytes=2,
        kv_bytes=2,
        accumulator_bytes=4,
        activation_name="BF16",
        weight_name="BF16",
        kv_name="BF16",
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _baseline_stage_result(seq_len: int, precision: PrecisionConfig | None = None):
    model = default_analytic_model_config()
    hw = default_hardware_config()
    prec = precision or default_precision_int8_weights()
    accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)
    return accel.simulate_token(seq_len)


def plot_pipeline_diagram(output_path: Path, model: ModelConfig) -> None:
    fig, ax = plt.subplots(figsize=(18, 4.2))
    ax.set_xlim(0, 16.9)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.25, 2.2, 1.0, 0.9, "Embedding", f"x_t: d={model.d_model}", "light"),
        (1.45, 2.2, 1.0, 0.9, "LayerNorm", f"norm: {model.d_model}", "light"),
        (2.65, 2.1, 1.4, 1.1, "Fused QKV", f"q: {model.d_model}\nk/v: {model.kv_dim}", "critical"),
        (4.3, 2.2, 1.05, 0.9, "KV write", f"2 x {model.kv_dim}", "throughput"),
        (5.6, 2.1, 1.4, 1.1, "Score", f"H x L = {model.num_heads} x L", "critical"),
        (7.25, 2.2, 0.95, 0.9, "Softmax", "per-head", "light"),
        (8.45, 2.1, 1.2, 1.1, "Value mix", f"ctx: {model.d_model}", "critical"),
        (9.9, 2.1, 1.4, 1.1, "O-proj + res", f"{model.d_model}", "critical"),
        (11.55, 2.2, 1.0, 0.9, "LayerNorm", f"{model.d_model}", "light"),
        (12.8, 2.1, 1.45, 1.1, "MLP SwiGLU", f"2 x {model.ffn_dim} -> {model.ffn_dim}", "critical"),
        (14.55, 2.1, 1.55, 1.1, "LM head + sample", f"vocab={model.vocab_size}", "throughput"),
    ]

    colors = {
        "critical": "#dfeaf9",
        "light": "#eef3e6",
        "throughput": "#f7eadf",
    }

    for x, y, w, h, title, subtitle, kind in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            facecolor=colors[kind],
            edgecolor="#3a3a3a",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8)

    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + boxes[idx][2]
        x2 = boxes[idx + 1][0]
        ax.add_patch(FancyArrowPatch((x1 + 0.05, 2.65), (x2 - 0.05, 2.65), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#555555"))

    ax.add_patch(Rectangle((0.15, 0.35), 5.15, 0.7, facecolor="#f4f8fd", edgecolor="#b8c7dd"))
    ax.add_patch(Rectangle((5.35, 0.35), 4.55, 0.7, facecolor="#fdf6f1", edgecolor="#d6b89f"))
    ax.add_patch(Rectangle((9.95, 0.35), 6.15, 0.7, facecolor="#f3f8ee", edgecolor="#b3c69c"))
    ax.text(2.73, 0.7, "Attention front-end and current-token projections", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(7.62, 0.7, "Sequence-dependent latency-critical attention", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(13.0, 0.7, "Output, MLP back-end, and token selection", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.text(0.2, 3.75, "Single-token decode pipeline (one layer + final projection)", fontsize=15, fontweight="bold")
    ax.text(0.2, 3.43, "Critical stages are dominated by weight streaming at short context and KV streaming at long context.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_diagram(output_path: Path, model: ModelConfig, hw: HardwareConfig, precision: PrecisionConfig) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(0.4, 7.45, "LLM inference pipeline accelerator microarchitecture", fontsize=16, fontweight="bold")
    ax.text(0.4, 7.08, f"Default datapath: {precision.activation_name} activations/KV, {precision.weight_name} weights, explicit scratchpad orchestration", fontsize=10)

    # Off-chip memory
    mem = FancyBboxPatch((0.5, 5.2), 3.3, 1.4, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#f6efe4", edgecolor="#8a6f48", linewidth=1.5)
    ax.add_patch(mem)
    ax.text(2.15, 6.18, "Off-chip memory", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(2.15, 5.82, "Model weights (streamed)\nLayered KV-cache\nPaged token blocks", ha="center", va="center", fontsize=10)

    # Scratchpad
    sp = FancyBboxPatch((5.1, 5.0), 3.8, 1.8, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#e9f3fb", edgecolor="#527aa3", linewidth=1.5)
    ax.add_patch(sp)
    ax.text(7.0, 6.35, "On-chip scratchpad + buffers", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(7.0, 5.88, f"A/B activation buffers\nWeight tiles: {hw.weight_buffer_bytes // 1024} KiB\nKV stream window: {hw.kv_stream_buffer_bytes // 1024} KiB", ha="center", va="center", fontsize=10)

    # Control plane
    ctrl = FancyBboxPatch((10.4, 5.25), 2.7, 1.3, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#eef3e6", edgecolor="#728a4d", linewidth=1.5)
    ax.add_patch(ctrl)
    ax.text(11.75, 6.12, "Decode scheduler", ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(11.75, 5.72, "Scoreboard\nToken-length counters\nDMA issue + bypass control", ha="center", va="center", fontsize=10)

    # Compute engines
    engines = [
        (0.8, 2.7, 2.5, 1.2, "Projection engine", f"{hw.projection_macs_per_cycle} MAC/cycle\nQKV + O + LM head", "#dfeaf9", "#527aa3"),
        (3.8, 2.7, 2.5, 1.2, "Attention engine", f"{hw.attention_macs_per_cycle} MAC/cycle\nQK score + alpha*V", "#fdf0e6", "#b97d4b"),
        (6.8, 2.7, 2.5, 1.2, "MLP engine", f"{hw.mlp_macs_per_cycle} MAC/cycle\nUp/gate + down", "#eef3e6", "#728a4d"),
        (9.8, 2.7, 2.5, 1.2, "Vector engine", f"{hw.vector_ops_per_cycle} ops/cycle\nLN + SwiGLU + sample", "#f3ebf8", "#8a5eb3"),
    ]
    for x, y, w, h, title, subtitle, fc, ec in engines:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center", fontsize=9.5)

    # arrows memory <-> scratchpad and scratchpad -> engines
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#5a5a5a")
    ax.add_patch(FancyArrowPatch((3.8, 5.9), (5.05, 5.9), **arrow_kw))
    ax.text(4.45, 6.12, f"DMA {hw.dma_bytes_per_cycle} B/cycle", fontsize=9, ha="center")
    ax.add_patch(FancyArrowPatch((8.95, 5.9), (10.35, 5.9), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.0, 5.0), (2.05, 3.95), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.0, 5.0), (5.05, 3.95), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.0, 5.0), (8.05, 3.95), **arrow_kw))
    ax.add_patch(FancyArrowPatch((7.0, 5.0), (11.05, 3.95), **arrow_kw))

    # Bypass path
    ax.add_patch(FancyArrowPatch((2.05, 2.7), (5.05, 2.25), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#cf5c36", connectionstyle="arc3,rad=-0.2"))
    ax.text(3.75, 1.95, "Current-token K/V bypass\navoids write-then-read stall", ha="center", va="center", fontsize=9, color="#9f3f24")

    # bottom notes
    notes = FancyBboxPatch((0.8, 0.55), 12.0, 1.0, boxstyle="round,pad=0.04,rounding_size=0.05", facecolor="#fbfbfb", edgecolor="#b0b0b0")
    ax.add_patch(notes)
    ax.text(1.05, 1.23, "Design choices:", fontsize=10.5, fontweight="bold")
    ax.text(2.55, 1.23, "Vector-style GEMV datapaths for batch=1 decode; software-managed scratchpad; explicit DMA overlap; dedicated attention reduction unit.", fontsize=9.5)
    ax.text(2.55, 0.83, f"Example baseline shown in the report: {model.num_layers}-layer decoder, d_model={model.d_model}, heads={model.num_heads}, ffn={model.ffn_dim}, max context={model.max_seq_len}.", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(output_path: Path, seq_len: int = 2048) -> Dict[str, float]:
    result = _baseline_stage_result(seq_len)
    categories = result.aggregate_category_cycles
    labels = list(categories.keys())
    values = [cycles_to_us(v, default_hardware_config().clock_ghz) for v in categories.values()]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.barh(labels, values)
    ax.set_xlabel("Latency contribution (microseconds at 1 GHz)")
    ax.set_title(f"Single-token decode latency breakdown at sequence length {seq_len}")
    for idx, val in enumerate(values):
        ax.text(val + max(values) * 0.01, idx, f"{val:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "seq_len": seq_len,
        "total_cycles": result.total_cycles,
        "total_time_us": cycles_to_us(result.total_cycles, default_hardware_config().clock_ghz),
        "overlap_reduction_pct": result.overlap_reduction * 100.0,
    }
    summary.update({f"category_us::{k}": v for k, v in zip(labels, values)})
    return summary


def plot_latency_vs_sequence(output_path: Path, seq_lens: Sequence[int], precision: PrecisionConfig | None = None) -> List[Dict[str, float]]:
    model = default_analytic_model_config()
    hw = default_hardware_config()
    prec = precision or default_precision_int8_weights()
    accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)
    rows = accel.simulate_sweep(seq_lens)

    totals = [cycles_to_us(row["total_cycles"], hw.clock_ghz) for row in rows]
    kv_score = [cycles_to_us(row.get("category::KV read + attention score", 0.0), hw.clock_ghz) for row in rows]
    mlp = [cycles_to_us(row.get("category::MLP", 0.0), hw.clock_ghz) for row in rows]
    logits = [cycles_to_us(row.get("category::Logits/sample", 0.0), hw.clock_ghz) for row in rows]

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.plot(seq_lens, totals, marker="o", linewidth=2.0, label="Total")
    ax.plot(seq_lens, kv_score, marker="s", linewidth=1.8, label="KV read + score")
    ax.plot(seq_lens, mlp, marker="^", linewidth=1.8, label="MLP")
    ax.plot(seq_lens, logits, marker="d", linewidth=1.8, label="Logits/sample")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (microseconds at 1 GHz)")
    ax.set_title(f"Latency scaling with context length ({prec.weight_name} weights)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_kv_footprint(output_path: Path, seq_lens: Sequence[int]) -> List[Dict[str, float]]:
    model = default_analytic_model_config()
    hw = default_hardware_config()
    prec = default_precision_int8_weights()
    accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)

    footprints = [mib(accel.kv_footprint_bytes(s)) for s in seq_lens]
    bw = [mib(accel.estimate_kv_bandwidth_per_token(s)) for s in seq_lens]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(seq_lens, footprints, marker="o", linewidth=2.0, label="KV footprint (MiB)")
    ax.plot(seq_lens, bw, marker="s", linewidth=2.0, label="KV read+write per token (MiB)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("MiB")
    ax.set_title("KV-cache storage and per-token bandwidth scale linearly with context")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for s, f, b in zip(seq_lens, footprints, bw):
        rows.append({"seq_len": float(s), "kv_footprint_mib": float(f), "kv_bandwidth_mib_per_token": float(b)})
    return rows


def plot_precision_comparison(output_path: Path, seq_lens: Sequence[int]) -> List[Dict[str, float]]:
    hw = default_hardware_config()
    model = default_analytic_model_config()
    int8_accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=default_precision_int8_weights())
    bf16_accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=bf16_everywhere_precision())

    int8 = int8_accel.simulate_sweep(seq_lens)
    bf16 = bf16_accel.simulate_sweep(seq_lens)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(seq_lens, [cycles_to_us(r["total_cycles"], hw.clock_ghz) for r in int8], marker="o", linewidth=2.0, label="INT8 weights")
    ax.plot(seq_lens, [cycles_to_us(r["total_cycles"], hw.clock_ghz) for r in bf16], marker="s", linewidth=2.0, label="BF16 weights")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (microseconds at 1 GHz)")
    ax.set_title("Weight precision shifts the bottleneck crossover toward KV-cache traffic")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for s, r_int8, r_bf16 in zip(seq_lens, int8, bf16):
        rows.append(
            {
                "seq_len": float(s),
                "int8_total_cycles": float(r_int8["total_cycles"]),
                "bf16_total_cycles": float(r_bf16["total_cycles"]),
            }
        )
    return rows


def plot_model_size_sensitivity(output_path: Path, d_models: Sequence[int], seq_len: int = 2048) -> List[Dict[str, float]]:
    hw = default_hardware_config()
    prec = default_precision_int8_weights()
    rows = []
    totals = []
    for d in d_models:
        model = ModelConfig(
            vocab_size=32000,
            max_seq_len=8192,
            num_layers=24,
            d_model=d,
            num_heads=max(8, d // 64),
            num_kv_heads=max(8, d // 64),
            ffn_dim=4 * d,
        )
        accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)
        result = accel.simulate_token(seq_len)
        total_us = cycles_to_us(result.total_cycles, hw.clock_ghz)
        totals.append(total_us)
        rows.append({"d_model": float(d), "ffn_dim": float(4 * d), "total_time_us": float(total_us)})

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(d_models, totals, marker="o", linewidth=2.0)
    ax.set_xlabel("Model hidden dimension (d_model)")
    ax.set_ylabel("Latency (microseconds at 1 GHz)")
    ax.set_title(f"Sensitivity to model width at sequence length {seq_len}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return rows


def _find_crossover_seq(latency_rows: Sequence[Dict[str, float]]) -> int | None:
    for row in latency_rows:
        kv = row.get("category::KV read + attention score", 0.0)
        mlp = row.get("category::MLP", 0.0)
        if kv >= mlp:
            return int(row["seq_len"])
    return None


def _find_memory_crossover_seq(model: ModelConfig, hw: HardwareConfig, prec: PrecisionConfig, seq_lens: Sequence[int]) -> int | None:
    accel = DecodeAcceleratorModel(model=model, hardware=hw, precision=prec)
    weight_stream = accel.memory_summary(seq_lens[0])["weight_stream_bytes_per_token"]
    for seq_len in seq_lens:
        if accel.estimate_kv_bandwidth_per_token(seq_len) >= weight_stream:
            return int(seq_len)
    return None


def _primary_memory_bottleneck(accel: DecodeAcceleratorModel, seq_len: int) -> str:
    memory = accel.memory_summary(seq_len)
    if memory["kv_bandwidth_bytes_per_token"] >= memory["weight_stream_bytes_per_token"]:
        return "KV-cache traffic"
    return "Weight streaming"


def run_validation(validation_dir: Path) -> Dict[str, float | int | bool | List[int]]:
    _ensure_dir(validation_dir)
    config = default_small_validation_config()
    weights = initialize_weights(config, seed=11)
    tokens = [3, 19, 7, 42, 11, 8, 14, 5, 91, 17, 22, 6]
    result = validate_incremental_decode(tokens, weights, config)
    data = {
        "config": asdict(config),
        "tokens": tokens,
        "max_abs_error": result.max_abs_error,
        "mean_abs_error": result.mean_abs_error,
        "passed": result.passed,
        "cache_lengths": result.cache_lengths,
        "sampled_tokens": result.sampled_tokens,
    }
    (validation_dir / "validation.json").write_text(json.dumps(data, indent=2))
    return data


def run_all(output_dir: Path) -> Dict[str, object]:
    _ensure_dir(output_dir)
    figs_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    validation_dir = output_dir / "validation"
    _ensure_dir(figs_dir)
    _ensure_dir(tables_dir)
    _ensure_dir(validation_dir)

    validation = run_validation(validation_dir)
    analytic_model = default_analytic_model_config()
    hw = default_hardware_config()
    prec = default_precision_int8_weights()

    plot_pipeline_diagram(figs_dir / "pipeline_diagram.png", analytic_model)
    plot_architecture_diagram(figs_dir / "architecture_diagram.png", analytic_model, hw, prec)
    breakdown_summary = plot_latency_breakdown(figs_dir / "latency_breakdown_2048.png", seq_len=2048)

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    latency_rows = plot_latency_vs_sequence(figs_dir / "latency_vs_sequence.png", seq_lens, precision=prec)
    _write_csv(tables_dir / "latency_vs_sequence.csv", latency_rows)

    kv_rows = plot_kv_footprint(figs_dir / "kv_footprint.png", seq_lens)
    _write_csv(tables_dir / "kv_footprint.csv", kv_rows)

    precision_rows = plot_precision_comparison(figs_dir / "precision_comparison.png", seq_lens)
    _write_csv(tables_dir / "precision_comparison.csv", precision_rows)

    model_size_rows = plot_model_size_sensitivity(figs_dir / "model_size_sensitivity.png", [512, 768, 1024, 1536, 2048], seq_len=2048)
    _write_csv(tables_dir / "model_size_sensitivity.csv", model_size_rows)

    baseline_accel = DecodeAcceleratorModel(analytic_model, hw, prec)
    mem_summary_512 = baseline_accel.memory_summary(512)
    mem_summary_2048 = baseline_accel.memory_summary(2048)
    mem_summary_8192 = baseline_accel.memory_summary(8192)

    stage_result_2048 = baseline_accel.simulate_token(2048)
    stage_rows = []
    for record in stage_result_2048.stage_records:
        stage_rows.append(
            {
                "label": record.label,
                "latency_cycles": float(record.latency),
                "wait_for_data_cycles": float(record.wait_for_data),
                "engine": record.engine,
            }
        )
    _write_csv(tables_dir / "stage_records_2048.csv", stage_rows)

    memory_table = [
        {
            "seq_len": 512.0,
            "kv_footprint_mib": mib(mem_summary_512["kv_footprint_bytes"]),
            "kv_bandwidth_mib_per_token": mib(mem_summary_512["kv_bandwidth_bytes_per_token"]),
            "weight_stream_mib_per_token": mib(mem_summary_512["weight_stream_bytes_per_token"]),
        },
        {
            "seq_len": 2048.0,
            "kv_footprint_mib": mib(mem_summary_2048["kv_footprint_bytes"]),
            "kv_bandwidth_mib_per_token": mib(mem_summary_2048["kv_bandwidth_bytes_per_token"]),
            "weight_stream_mib_per_token": mib(mem_summary_2048["weight_stream_bytes_per_token"]),
        },
        {
            "seq_len": 8192.0,
            "kv_footprint_mib": mib(mem_summary_8192["kv_footprint_bytes"]),
            "kv_bandwidth_mib_per_token": mib(mem_summary_8192["kv_bandwidth_bytes_per_token"]),
            "weight_stream_mib_per_token": mib(mem_summary_8192["weight_stream_bytes_per_token"]),
        },
    ]
    _write_csv(tables_dir / "memory_summary.csv", memory_table)

    crossover_seq = _find_crossover_seq(latency_rows)
    memory_crossover_seq = _find_memory_crossover_seq(analytic_model, hw, prec, seq_lens)
    bottleneck_2048 = baseline_accel.identify_primary_bottleneck(2048)
    bottleneck_8192 = baseline_accel.identify_primary_bottleneck(8192)
    primary_memory_2048 = _primary_memory_bottleneck(baseline_accel, 2048)
    primary_memory_8192 = _primary_memory_bottleneck(baseline_accel, 8192)

    summary: Dict[str, object] = {
        "validation": validation,
        "baseline_model": asdict(analytic_model),
        "hardware": asdict(hw),
        "precision": asdict(prec),
        "breakdown_summary": breakdown_summary,
        "memory_summary_rows": memory_table,
        "crossover_seq_len": crossover_seq,
        "memory_crossover_seq_len": memory_crossover_seq,
        "bottleneck_2048": bottleneck_2048,
        "bottleneck_8192": bottleneck_8192,
        "primary_memory_bottleneck_2048": primary_memory_2048,
        "primary_memory_bottleneck_8192": primary_memory_8192,
        "figure_paths": {
            "pipeline": str(figs_dir / "pipeline_diagram.png"),
            "architecture": str(figs_dir / "architecture_diagram.png"),
            "breakdown": str(figs_dir / "latency_breakdown_2048.png"),
            "latency_vs_sequence": str(figs_dir / "latency_vs_sequence.png"),
            "kv_footprint": str(figs_dir / "kv_footprint.png"),
            "precision_comparison": str(figs_dir / "precision_comparison.png"),
            "model_size_sensitivity": str(figs_dir / "model_size_sensitivity.png"),
        },
        "table_paths": {
            "latency_vs_sequence": str(tables_dir / "latency_vs_sequence.csv"),
            "kv_footprint": str(tables_dir / "kv_footprint.csv"),
            "precision_comparison": str(tables_dir / "precision_comparison.csv"),
            "model_size_sensitivity": str(tables_dir / "model_size_sensitivity.csv"),
            "memory_summary": str(tables_dir / "memory_summary.csv"),
            "stage_records_2048": str(tables_dir / "stage_records_2048.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
