from __future__ import annotations

import argparse
from pathlib import Path

from llm_inference_accel.experiments import run_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures, tables, and summaries for the LLM inference pipeline accelerator project.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for generated figures, tables, and summary JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all(args.output_dir)
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")
    print("Generated figures:")
    for name, path in summary["figure_paths"].items():
        print(f"  {name}: {path}")
    print("Generated tables:")
    for name, path in summary["table_paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
