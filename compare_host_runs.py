from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_summary(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "summary.json"
        if candidate.exists():
            return candidate
    return path


def _load_summary(path: Path) -> Dict[str, object]:
    resolved = _resolve_summary(path)
    return json.loads(resolved.read_text())


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def _latency_plot(summaries: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for summary in summaries:
        label = str(summary.get("run_label", "host"))
        rows = summary.get("latency_rows", [])
        seq = [int(row["seq_len"]) for row in rows]
        lat = [float(row["latency_us"]) for row in rows]
        ax.plot(seq, lat, marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency per token (microseconds)")
    ax.set_title("M2 vs M4 host-calibrated latency comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _throughput_plot(summaries: Sequence[Dict[str, object]], output_path: Path) -> None:
    metrics = [
        ("projection_macs_per_sec", "Projection GMAC/s"),
        ("mlp_macs_per_sec", "MLP GMAC/s"),
        ("attention_macs_per_sec", "Attention GMAC/s"),
        ("vector_ops_per_sec", "Vector Gop/s"),
        ("dma_bytes_per_sec", "DMA GB/s"),
    ]
    labels = [str(summary.get("run_label", "host")) for summary in summaries]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.0, 11.0))
    for ax, (metric, title) in zip(axes, metrics):
        values = []
        for summary in summaries:
            cal = summary.get("calibration", {})
            raw = float(cal.get(metric, 0.0))
            if metric.endswith("bytes_per_sec"):
                values.append(raw / 1e9)
            else:
                values.append(raw / 1e9)
        bars = ax.bar(labels, values)
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare host-calibrated runs from different Macs, for example M2 and M4.")
    parser.add_argument("runs", type=Path, nargs="+", help="Paths to summary.json files or directories that contain summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "host_comparison", help="Directory for combined comparison outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [_load_summary(path) for path in args.runs]

    headline_rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    for summary in summaries:
        host = summary.get("host", {})
        cal = summary.get("calibration", {})
        row_2048 = summary.get("latency_at_2048") or {}
        headline_rows.append(
            {
                "run_label": summary.get("run_label", "host"),
                "chip": host.get("chip", ""),
                "model_identifier": host.get("model_identifier", ""),
                "logical_cpus": host.get("total_logical_cpus", 0),
                "performance_cores": host.get("performance_cores", 0),
                "efficiency_cores": host.get("efficiency_cores", 0),
                "memory_gib": round(float(host.get("memory_bytes", 0.0)) / (1024.0 ** 3), 2),
                "latency_2048_us": row_2048.get("latency_us", 0.0),
                "tokens_per_s_2048": row_2048.get("tokens_per_s", 0.0),
                "projection_gmac_s": float(cal.get("projection_macs_per_sec", 0.0)) / 1e9,
                "mlp_gmac_s": float(cal.get("mlp_macs_per_sec", 0.0)) / 1e9,
                "attention_gmac_s": float(cal.get("attention_macs_per_sec", 0.0)) / 1e9,
                "vector_gops_s": float(cal.get("vector_ops_per_sec", 0.0)) / 1e9,
                "dma_gbytes_s": float(cal.get("dma_bytes_per_sec", 0.0)) / 1e9,
            }
        )
        for latency in summary.get("latency_rows", []):
            latency_rows.append(
                {
                    "run_label": summary.get("run_label", "host"),
                    "chip": host.get("chip", ""),
                    **latency,
                }
            )

    _write_csv(args.output_dir / "headline_comparison.csv", headline_rows)
    _write_csv(args.output_dir / "latency_comparison.csv", latency_rows)
    _latency_plot(summaries, args.output_dir / "latency_comparison.png")
    _throughput_plot(summaries, args.output_dir / "throughput_comparison.png")

    combined = {
        "runs": headline_rows,
        "artifacts": {
            "headline_csv": str(args.output_dir / "headline_comparison.csv"),
            "latency_csv": str(args.output_dir / "latency_comparison.csv"),
            "latency_plot": str(args.output_dir / "latency_comparison.png"),
            "throughput_plot": str(args.output_dir / "throughput_comparison.png"),
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(combined, indent=2))

    print(f"Wrote comparison to {args.output_dir}")
    for row in headline_rows:
        print(
            f"{row['run_label']}: {row['chip']} | "
            f"latency@2048={float(row['latency_2048_us']):.2f} us/token | "
            f"throughput@2048={float(row['tokens_per_s_2048']):.2f} tok/s"
        )


if __name__ == "__main__":
    main()
