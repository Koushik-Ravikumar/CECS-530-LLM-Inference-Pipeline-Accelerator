from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm_inference_accel.accelerator import DecodeAcceleratorModel
from llm_inference_accel.config import HardwareConfig, ModelConfig, PrecisionConfig, RuntimeConfig
from llm_inference_accel.experiments import default_analytic_model_config, default_precision_int8_weights
from llm_inference_accel.host import calibrate_host_numpy, detect_host_info
from llm_inference_accel.model import decode_sequence_incremental, initialize_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a functional and analytical demo of the LLM inference pipeline accelerator.")
    parser.add_argument("--tokens", type=int, nargs="*", default=[7, 11, 3, 42, 9, 5], help="Input token IDs")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--host-calibrated", action="store_true", help="Detect the local machine and calibrate the analytical hardware model from local NumPy throughput")
    parser.add_argument("--benchmark-seconds", type=float, default=0.20, help="Minimum seconds per local microbenchmark when --host-calibrated is enabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig(vocab_size=256, max_seq_len=32, num_layers=2, d_model=128, num_heads=4, num_kv_heads=4, ffn_dim=512)
    weights = initialize_weights(model_cfg, seed=17)
    runtime = RuntimeConfig(greedy=True)
    functional = decode_sequence_incremental(args.tokens, weights, model_cfg, runtime)

    analytic_cfg = default_analytic_model_config()
    prec_cfg = default_precision_int8_weights()
    hw_cfg: HardwareConfig = HardwareConfig()
    host_payload = None

    if args.host_calibrated:
        host = detect_host_info()
        calibration = calibrate_host_numpy(
            model=analytic_cfg,
            host=host,
            benchmark_seconds=args.benchmark_seconds,
            sequence_length_reference=max(len(args.tokens), 1) * 256,
        )
        hw_cfg = calibration.to_hardware_config()
        host_payload = {
            "host": host.to_dict(),
            "calibration": calibration.to_dict(),
        }

    accel = DecodeAcceleratorModel(analytic_cfg, hw_cfg, prec_cfg)
    perf = accel.simulate_token(seq_len=max(len(args.tokens), 1) * 256)

    result = {
        "input_tokens": args.tokens,
        "sampled_tokens": functional.sampled_tokens,
        "final_argmax": int(functional.sampled_tokens[-1]),
        "demo_seq_len_for_analytics": max(len(args.tokens), 1) * 256,
        "analytic_total_cycles": perf.total_cycles,
        "analytic_total_time_ns": perf.total_time_ns,
        "analytic_top_categories": perf.top_critical_stages(),
        "kv_bandwidth_bytes_per_token": accel.estimate_kv_bandwidth_per_token(max(len(args.tokens), 1) * 256),
        "hardware": asdict(hw_cfg),
        "precision": asdict(prec_cfg),
        "host_calibrated": args.host_calibrated,
    }
    if host_payload is not None:
        result["host_profile"] = host_payload

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
