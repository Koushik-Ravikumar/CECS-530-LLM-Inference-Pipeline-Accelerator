from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_inference_accel import DecodeAcceleratorModel
from llm_inference_accel.config import ModelConfig, PrecisionConfig
from llm_inference_accel.experiments import default_analytic_model_config, default_precision_int8_weights
from llm_inference_accel.host import HostCalibration, calibrate_host_numpy, detect_host_info, host_summary_rows


DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]


def cycles_to_us(cycles: float, clock_ghz: float) -> float:
    return cycles / (clock_ghz * 1_000.0)


def sanitize_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    slug = slug.strip("._-")
    return slug or "host_run"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_latency(rows: Sequence[Dict[str, object]], output_path: Path, label: str) -> None:
    seq = [int(row["seq_len"]) for row in rows]
    lat_us = [float(row["latency_us"]) for row in rows]
    tok_s = [float(row["tokens_per_s"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(seq, lat_us, marker="o", linewidth=2.0)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency per token (microseconds)")
    ax.set_title(f"Host-calibrated decode latency vs. sequence length: {label}")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(seq, tok_s, marker="s", linewidth=1.6, linestyle="--")
    ax2.set_ylabel("Estimated tokens / second")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_throughput(calibration: HostCalibration, output_path: Path, label: str) -> None:
    entries = [
        ("Projection", calibration.projection_macs_per_sec / 1e9, "GMAC/s"),
        ("MLP", calibration.mlp_macs_per_sec / 1e9, "GMAC/s"),
        ("Attention", calibration.attention_macs_per_sec / 1e9, "GMAC/s"),
        ("Vector", calibration.vector_ops_per_sec / 1e9, "Gop/s"),
        ("DMA", calibration.dma_bytes_per_sec / 1e9, "GB/s"),
    ]
    labels = [entry[0] for entry in entries]
    values = [entry[1] for entry in entries]
    units = [entry[2] for entry in entries]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Measured throughput")
    ax.set_title(f"Local host calibration kernels: {label}")
    for bar, value, unit in zip(bars, values, units):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f} {unit}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_rows(model: ModelConfig, precision: PrecisionConfig, calibration: HostCalibration, seq_lens: Iterable[int]) -> List[Dict[str, object]]:
    hardware = calibration.to_hardware_config()
    accel = DecodeAcceleratorModel(model=model, hardware=hardware, precision=precision)
    rows: List[Dict[str, object]] = []
    for seq_len in seq_lens:
        result = accel.simulate_token(int(seq_len))
        latency_us = cycles_to_us(result.total_cycles, hardware.clock_ghz)
        rows.append(
            {
                "seq_len": int(seq_len),
                "latency_us": latency_us,
                "tokens_per_s": 1e6 / latency_us if latency_us > 0 else 0.0,
                "total_cycles": result.total_cycles,
                "total_time_ns": result.total_time_ns,
                "kv_bytes_read": result.kv_bytes_read,
                "kv_bytes_write": result.kv_bytes_write,
                "weight_bytes": result.weight_bytes,
                "embedding_bytes": result.embedding_bytes,
                "overlap_reduction_pct": result.overlap_reduction * 100.0,
                "primary_bottleneck": accel.identify_primary_bottleneck(int(seq_len)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect the local machine, calibrate throughput, and run host-dependent LLM inference accelerator comparisons.")
    parser.add_argument("--label", type=str, default=None, help="Optional run label, for example m2_air or m4_pro")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "host_runs", help="Directory for host-calibrated outputs")
    parser.add_argument("--seq-lens", type=int, nargs="*", default=DEFAULT_SEQ_LENS, help="Sequence lengths to sweep")
    parser.add_argument("--benchmark-seconds", type=float, default=0.25, help="Minimum seconds per local microbenchmark")
    parser.add_argument("--reference-seq-len", type=int, default=2048, help="Representative sequence length used for attention calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = detect_host_info()
    label = args.label or sanitize_label(host.chip or host.model_identifier or host.model_name or host.machine)
    run_dir = args.output_dir / sanitize_label(label)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = default_analytic_model_config()
    precision = default_precision_int8_weights()
    calibration = calibrate_host_numpy(
        model=model,
        host=host,
        benchmark_seconds=args.benchmark_seconds,
        sequence_length_reference=args.reference_seq_len,
    )
    hardware = calibration.to_hardware_config()
    rows = build_rows(model, precision, calibration, args.seq_lens)

    write_csv(run_dir / "latency_vs_sequence.csv", rows)
    write_csv(run_dir / "host_metrics.csv", host_summary_rows(calibration))
    plot_latency(rows, run_dir / "latency_vs_sequence.png", label=label)
    plot_throughput(calibration, run_dir / "kernel_throughput.png", label=label)

    lookup_2048 = {int(row["seq_len"]): row for row in rows}
    summary = {
        "run_label": label,
        "host": host.to_dict(),
        "model": asdict(model),
        "precision": asdict(precision),
        "hardware": asdict(hardware),
        "calibration": calibration.to_dict(),
        "latency_rows": rows,
        "latency_at_2048": lookup_2048.get(2048),
        "artifacts": {
            "latency_csv": str(run_dir / "latency_vs_sequence.csv"),
            "host_metrics_csv": str(run_dir / "host_metrics.csv"),
            "latency_plot": str(run_dir / "latency_vs_sequence.png"),
            "throughput_plot": str(run_dir / "kernel_throughput.png"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote host-calibrated run to {run_dir}")
    print(f"Host: {host.label()}")
    if lookup_2048.get(2048) is not None:
        row = lookup_2048[2048]
        print(f"Latency @ 2048 tokens: {float(row['latency_us']):.2f} us/token")
        print(f"Throughput @ 2048 tokens: {float(row['tokens_per_s']):.2f} tok/s")
    print("Files:")
    print(f"  {run_dir / 'summary.json'}")
    print(f"  {run_dir / 'latency_vs_sequence.csv'}")
    print(f"  {run_dir / 'latency_vs_sequence.png'}")
    print(f"  {run_dir / 'kernel_throughput.png'}")


if __name__ == "__main__":
    main()
