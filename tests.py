from __future__ import annotations

from llm_inference_accel.accelerator import DecodeAcceleratorModel
from llm_inference_accel.config import HardwareConfig, ModelConfig, PrecisionConfig
from llm_inference_accel.host import calibrate_host_numpy, detect_host_info
from llm_inference_accel.model import initialize_weights, validate_incremental_decode


def make_test_accel(max_seq_len: int = 8192) -> DecodeAcceleratorModel:
    cfg = ModelConfig(
        vocab_size=32000,
        max_seq_len=max_seq_len,
        num_layers=12,
        d_model=512,
        num_heads=8,
        num_kv_heads=8,
        ffn_dim=2048,
    )
    return DecodeAcceleratorModel(cfg, HardwareConfig(), PrecisionConfig())


def test_incremental_matches_full() -> None:
    cfg = ModelConfig(
        vocab_size=128,
        max_seq_len=24,
        num_layers=2,
        d_model=96,
        num_heads=4,
        num_kv_heads=4,
        ffn_dim=384,
    )
    weights = initialize_weights(cfg, seed=9)
    tokens = [5, 8, 2, 17, 3, 9, 21, 4, 11, 7]
    result = validate_incremental_decode(tokens, weights, cfg)

    assert result.passed, f"incremental decode mismatch: max error={result.max_abs_error}"
    assert all(length == len(tokens) for length in result.cache_lengths)


def test_latency_increases_with_seq_len() -> None:
    accel = make_test_accel(max_seq_len=4096)

    small = accel.simulate_token(256)
    large = accel.simulate_token(2048)

    assert large.total_cycles > small.total_cycles
    assert accel.estimate_kv_bandwidth_per_token(2048) > accel.estimate_kv_bandwidth_per_token(256)


def test_bypass_reduces_or_preserves_latency() -> None:
    cfg = ModelConfig(
        vocab_size=32000,
        max_seq_len=4096,
        num_layers=12,
        d_model=512,
        num_heads=8,
        num_kv_heads=8,
        ffn_dim=2048,
    )
    prec = PrecisionConfig()

    base = DecodeAcceleratorModel(
        cfg,
        HardwareConfig(use_bypass_for_current_kv=True),
        prec,
    ).simulate_token(1024)

    conservative = DecodeAcceleratorModel(
        cfg,
        HardwareConfig(use_bypass_for_current_kv=False),
        prec,
    ).simulate_token(1024)

    assert base.total_cycles <= conservative.total_cycles


def test_host_calibration_produces_positive_rates() -> None:
    cfg = ModelConfig(
        vocab_size=4096,
        max_seq_len=512,
        num_layers=4,
        d_model=128,
        num_heads=4,
        num_kv_heads=4,
        ffn_dim=512,
    )

    host = detect_host_info()
    calibration = calibrate_host_numpy(
        cfg,
        host=host,
        benchmark_seconds=0.01,
        sequence_length_reference=256,
    )
    hw = calibration.to_hardware_config()

    assert calibration.projection_macs_per_sec > 0.0
    assert calibration.mlp_macs_per_sec > 0.0
    assert calibration.attention_macs_per_sec > 0.0
    assert calibration.vector_ops_per_sec > 0.0
    assert calibration.dma_bytes_per_sec > 0.0
    assert hw.projection_macs_per_cycle > 0.0
    assert hw.dma_bytes_per_cycle > 0.0


def test_context_boundary_cases() -> None:
    accel = make_test_accel(max_seq_len=8192)

    for ctx in [1, 128, 2048, 8192]:
        result = accel.simulate_token(ctx)

        assert result.total_cycles > 0
        assert result.kv_bytes_read >= 0
        assert result.kv_bytes_write > 0


def test_dma_queue_ordering() -> None:
    accel = make_test_accel(max_seq_len=8192)
    result = accel.simulate_token(2048)

    dma_tasks = sorted(
        [task for task in result.tasks if task.engine == "dma"],
        key=lambda task: task.start,
    )

    assert dma_tasks, "no DMA tasks were scheduled"

    for first, second in zip(dma_tasks, dma_tasks[1:]):
        assert first.end <= second.start, (
            f"DMA overlap: {first.name} overlaps {second.name}"
        )


def test_engine_exclusivity() -> None:
    accel = make_test_accel(max_seq_len=8192)
    result = accel.simulate_token(2048)

    tasks_by_engine: dict[str, list] = {}

    for task in result.tasks:
        tasks_by_engine.setdefault(task.engine, []).append(task)

    for engine, tasks in tasks_by_engine.items():
        sorted_tasks = sorted(tasks, key=lambda task: task.start)

        for first, second in zip(sorted_tasks, sorted_tasks[1:]):
            assert first.end <= second.start, (
                f"{engine} overlap: {first.name} overlaps {second.name}"
            )


def test_deterministic_reproduction() -> None:
    accel = make_test_accel(max_seq_len=8192)

    first = accel.simulate_token(2048)
    second = accel.simulate_token(2048)

    assert first.total_cycles == second.total_cycles
    assert first.kv_bytes_read == second.kv_bytes_read
    assert first.kv_bytes_write == second.kv_bytes_write
    assert first.weight_bytes == second.weight_bytes
    assert first.aggregate_category_cycles == second.aggregate_category_cycles


def main() -> None:
    tests = [
        test_incremental_matches_full,
        test_latency_increases_with_seq_len,
        test_bypass_reduces_or_preserves_latency,
        test_host_calibration_produces_positive_rates,
        test_context_boundary_cases,
        test_dma_queue_ordering,
        test_engine_exclusivity,
        test_deterministic_reproduction,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print(f"All tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()