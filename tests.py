from __future__ import annotations

from llm_inference_accel.accelerator import DecodeAcceleratorModel
from llm_inference_accel.config import HardwareConfig, ModelConfig, PrecisionConfig
from llm_inference_accel.host import calibrate_host_numpy, detect_host_info
from llm_inference_accel.model import initialize_weights, validate_incremental_decode


def test_incremental_matches_full() -> None:
    cfg = ModelConfig(vocab_size=128, max_seq_len=24, num_layers=2, d_model=96, num_heads=4, num_kv_heads=4, ffn_dim=384)
    weights = initialize_weights(cfg, seed=9)
    tokens = [5, 8, 2, 17, 3, 9, 21, 4, 11, 7]
    result = validate_incremental_decode(tokens, weights, cfg)
    assert result.passed, f"incremental decode mismatch: max error={result.max_abs_error}"
    assert all(length == len(tokens) for length in result.cache_lengths)


def test_latency_increases_with_seq_len() -> None:
    cfg = ModelConfig(vocab_size=32000, max_seq_len=4096, num_layers=12, d_model=512, num_heads=8, num_kv_heads=8, ffn_dim=2048)
    accel = DecodeAcceleratorModel(cfg, HardwareConfig(), PrecisionConfig())
    small = accel.simulate_token(256)
    large = accel.simulate_token(2048)
    assert large.total_cycles > small.total_cycles
    assert accel.estimate_kv_bandwidth_per_token(2048) > accel.estimate_kv_bandwidth_per_token(256)


def test_bypass_reduces_or_preserves_latency() -> None:
    cfg = ModelConfig(vocab_size=32000, max_seq_len=4096, num_layers=12, d_model=512, num_heads=8, num_kv_heads=8, ffn_dim=2048)
    base_hw = HardwareConfig(use_bypass_for_current_kv=True)
    conservative_hw = HardwareConfig(use_bypass_for_current_kv=False)
    prec = PrecisionConfig()
    base = DecodeAcceleratorModel(cfg, base_hw, prec).simulate_token(1024)
    conservative = DecodeAcceleratorModel(cfg, conservative_hw, prec).simulate_token(1024)
    assert base.total_cycles <= conservative.total_cycles


def test_host_calibration_produces_positive_rates() -> None:
    cfg = ModelConfig(vocab_size=4096, max_seq_len=512, num_layers=4, d_model=128, num_heads=4, num_kv_heads=4, ffn_dim=512)
    host = detect_host_info()
    calibration = calibrate_host_numpy(cfg, host=host, benchmark_seconds=0.01, sequence_length_reference=256)
    hw = calibration.to_hardware_config()
    assert calibration.projection_macs_per_sec > 0.0
    assert calibration.mlp_macs_per_sec > 0.0
    assert calibration.attention_macs_per_sec > 0.0
    assert calibration.vector_ops_per_sec > 0.0
    assert calibration.dma_bytes_per_sec > 0.0
    assert hw.projection_macs_per_cycle > 0.0
    assert hw.dma_bytes_per_cycle > 0.0


def main() -> None:
    test_incremental_matches_full()
    test_latency_increases_with_seq_len()
    test_bypass_reduces_or_preserves_latency()
    test_host_calibration_produces_positive_rates()
    print("All tests passed.")


if __name__ == "__main__":
    main()
