from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import HardwareConfig, ModelConfig


MiB = 1024 * 1024
GiB = 1024 * MiB


def _run_command(args: Sequence[str], timeout: float = 5.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    output = proc.stdout.strip()
    return output or None


def _sysctl(name: str) -> Optional[str]:
    return _run_command(["sysctl", "-n", name])


def _as_int(value: Optional[str]) -> int:
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _capture_numpy_config() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    return buf.getvalue().strip()


@dataclass
class HostInfo:
    system: str
    machine: str
    processor: str
    chip: str
    model_identifier: str
    model_name: str
    total_logical_cpus: int
    performance_cores: int
    efficiency_cores: int
    memory_bytes: int
    cache_line_bytes: int
    l2_cache_bytes: int
    l3_cache_bytes: int
    is_apple_silicon: bool
    running_under_rosetta: bool
    os_version: str
    python_version: str
    numpy_version: str
    numpy_config: str

    def label(self) -> str:
        parts = [self.chip or self.processor or self.machine]
        if self.model_name:
            parts.append(self.model_name)
        return " | ".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KernelBenchmark:
    name: str
    work_per_iteration: float
    elapsed_s: float
    iterations: int
    throughput_per_s: float
    units: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HostCalibration:
    host: HostInfo
    projection_macs_per_sec: float
    mlp_macs_per_sec: float
    attention_macs_per_sec: float
    vector_ops_per_sec: float
    dma_bytes_per_sec: float
    benchmark_seconds: float
    sequence_length_reference: int
    samples: Dict[str, KernelBenchmark] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_hardware_config(self) -> HardwareConfig:
        cache_budget = self.host.l2_cache_bytes + self.host.l3_cache_bytes
        if cache_budget <= 0:
            # Conservative fallback on machines where cache topology is hidden.
            cache_budget = 4 * MiB
        scratchpad = max(512 * 1024, min(cache_budget // 2, 8 * MiB))
        return HardwareConfig(
            projection_macs_per_cycle=max(self.projection_macs_per_sec / 1e9, 1e-6),
            mlp_macs_per_cycle=max(self.mlp_macs_per_sec / 1e9, 1e-6),
            attention_macs_per_cycle=max(self.attention_macs_per_sec / 1e9, 1e-6),
            vector_ops_per_cycle=max(self.vector_ops_per_sec / 1e9, 1e-6),
            dma_bytes_per_cycle=max(self.dma_bytes_per_sec / 1e9, 1e-6),
            compute_efficiency=1.0,
            dma_efficiency=1.0,
            scratchpad_bytes=int(scratchpad),
            weight_buffer_bytes=max(int(scratchpad // 4), 128 * 1024),
            kv_stream_buffer_bytes=max(int(scratchpad // 4), 128 * 1024),
            clock_ghz=1.0,
            use_bypass_for_current_kv=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host.to_dict(),
            "projection_macs_per_sec": self.projection_macs_per_sec,
            "mlp_macs_per_sec": self.mlp_macs_per_sec,
            "attention_macs_per_sec": self.attention_macs_per_sec,
            "vector_ops_per_sec": self.vector_ops_per_sec,
            "dma_bytes_per_sec": self.dma_bytes_per_sec,
            "benchmark_seconds": self.benchmark_seconds,
            "sequence_length_reference": self.sequence_length_reference,
            "samples": {key: value.to_dict() for key, value in self.samples.items()},
            "notes": list(self.notes),
            "derived_hardware_config": asdict(self.to_hardware_config()),
        }


def detect_host_info() -> HostInfo:
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor() or machine

    total_logical_cpus = os.cpu_count() or 0
    performance_cores = 0
    efficiency_cores = 0
    memory_bytes = 0
    cache_line_bytes = 0
    l2_cache_bytes = 0
    l3_cache_bytes = 0
    model_identifier = ""
    model_name = ""
    chip = processor
    running_under_rosetta = False

    if system == "Darwin":
        total_logical_cpus = _as_int(_sysctl("hw.logicalcpu_max")) or total_logical_cpus
        performance_cores = _as_int(_sysctl("hw.perflevel0.logicalcpu_max"))
        efficiency_cores = _as_int(_sysctl("hw.perflevel1.logicalcpu_max"))
        memory_bytes = _as_int(_sysctl("hw.memsize"))
        cache_line_bytes = _as_int(_sysctl("hw.cachelinesize"))
        l2_cache_bytes = _as_int(_sysctl("hw.l2cachesize"))
        l3_cache_bytes = _as_int(_sysctl("hw.l3cachesize"))
        model_identifier = _sysctl("hw.model") or ""
        translated = _sysctl("sysctl.proc_translated")
        running_under_rosetta = str(translated).strip() == "1"

        profiler_raw = _run_command(["system_profiler", "SPHardwareDataType", "-json"], timeout=15.0)
        if profiler_raw:
            try:
                profiler_data = json.loads(profiler_raw)
                records = profiler_data.get("SPHardwareDataType", [])
                if records:
                    record = records[0]
                    model_name = str(
                        record.get("machine_name")
                        or record.get("model_name")
                        or record.get("machine_model")
                        or ""
                    ).strip()
                    chip = str(
                        record.get("chip_type")
                        or record.get("cpu_type")
                        or record.get("chip_model")
                        or chip
                    ).strip()
            except json.JSONDecodeError:
                pass

        if not chip or chip == machine:
            chip = (
                _sysctl("machdep.cpu.brand_string")
                or _sysctl("machdep.cpu.brand_string")
                or processor
            )

    elif system == "Linux":
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) if hasattr(os, "sysconf") else 0
        model_name = platform.node()
        chip = processor or machine
    else:
        model_name = platform.node()
        chip = processor or machine

    if performance_cores <= 0 and total_logical_cpus > 0:
        performance_cores = total_logical_cpus
    if efficiency_cores < 0:
        efficiency_cores = 0

    is_apple_silicon = system == "Darwin" and machine == "arm64"

    return HostInfo(
        system=system,
        machine=machine,
        processor=processor,
        chip=chip or processor or machine,
        model_identifier=model_identifier,
        model_name=model_name,
        total_logical_cpus=total_logical_cpus,
        performance_cores=performance_cores,
        efficiency_cores=efficiency_cores,
        memory_bytes=memory_bytes,
        cache_line_bytes=cache_line_bytes,
        l2_cache_bytes=l2_cache_bytes,
        l3_cache_bytes=l3_cache_bytes,
        is_apple_silicon=is_apple_silicon,
        running_under_rosetta=running_under_rosetta,
        os_version=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        numpy_config=_capture_numpy_config(),
    )


def _benchmark(name: str, work_per_iteration: float, units: str, func, min_seconds: float, min_iterations: int = 3) -> KernelBenchmark:
    for _ in range(2):
        func()

    iterations = 0
    start = time.perf_counter()
    elapsed = 0.0
    while iterations < min_iterations or elapsed < min_seconds:
        func()
        iterations += 1
        elapsed = time.perf_counter() - start

    throughput = 0.0
    if elapsed > 0.0:
        throughput = (work_per_iteration * iterations) / elapsed

    return KernelBenchmark(
        name=name,
        work_per_iteration=work_per_iteration,
        elapsed_s=elapsed,
        iterations=iterations,
        throughput_per_s=throughput,
        units=units,
    )


def calibrate_host_numpy(
    model: ModelConfig,
    host: Optional[HostInfo] = None,
    benchmark_seconds: float = 0.20,
    sequence_length_reference: int = 2048,
) -> HostCalibration:
    host = host or detect_host_info()

    # Representative, decode-like kernels. Sizes are chosen to be large enough
    # to exercise the local BLAS/vector backend without requiring huge memory.
    proj_rows = max(2048, 2 * model.d_model)
    proj_cols = max(1024, model.d_model)
    mlp_rows = max(4096, model.ffn_dim)
    mlp_cols = max(1024, model.d_model)
    attn_seq = max(1024, min(sequence_length_reference, model.max_seq_len, 8192))
    attn_dim = model.d_model
    vec_len = max(4096, model.ffn_dim)

    rng = np.random.default_rng(123)

    proj_w = rng.standard_normal((proj_rows, proj_cols), dtype=np.float32)
    proj_x = rng.standard_normal((proj_cols,), dtype=np.float32)
    proj_y = np.empty((proj_rows,), dtype=np.float32)

    mlp_w = rng.standard_normal((mlp_rows, mlp_cols), dtype=np.float32)
    mlp_x = rng.standard_normal((mlp_cols,), dtype=np.float32)
    mlp_y = np.empty((mlp_rows,), dtype=np.float32)

    k_cache = rng.standard_normal((attn_seq, attn_dim), dtype=np.float32)
    v_cache = rng.standard_normal((attn_seq, attn_dim), dtype=np.float32)
    q = rng.standard_normal((attn_dim,), dtype=np.float32)
    alpha = rng.standard_normal((attn_seq,), dtype=np.float32)
    score_out = np.empty((attn_seq,), dtype=np.float32)
    value_out = np.empty((attn_dim,), dtype=np.float32)

    vec_a = rng.standard_normal((vec_len,), dtype=np.float32)
    vec_b = rng.standard_normal((vec_len,), dtype=np.float32)

    dma_elems = max((64 * MiB) // 4, vec_len)
    if host.memory_bytes > 0:
        dma_elems = min(dma_elems, max(host.memory_bytes // 32 // 4, vec_len))
    dma_src = rng.standard_normal((dma_elems,), dtype=np.float32)
    dma_dst = np.empty_like(dma_src)

    def proj_kernel() -> None:
        np.matmul(proj_w, proj_x, out=proj_y)

    def mlp_kernel() -> None:
        np.matmul(mlp_w, mlp_x, out=mlp_y)

    def attn_score_kernel() -> None:
        np.matmul(k_cache, q, out=score_out)

    def attn_value_kernel() -> None:
        np.matmul(alpha, v_cache, out=value_out)

    def vector_kernel() -> None:
        centered = vec_a - np.mean(vec_a)
        variance = np.mean(centered * centered)
        norm = centered / np.sqrt(variance + 1e-5)
        gate = 1.0 / (1.0 + np.exp(-vec_b))
        _ = norm + gate * vec_b

    def dma_kernel() -> None:
        np.copyto(dma_dst, dma_src)

    proj_sample = _benchmark(
        name="projection_matvec",
        work_per_iteration=float(proj_rows * proj_cols),
        units="mac/s",
        func=proj_kernel,
        min_seconds=benchmark_seconds,
    )
    mlp_sample = _benchmark(
        name="mlp_matvec",
        work_per_iteration=float(mlp_rows * mlp_cols),
        units="mac/s",
        func=mlp_kernel,
        min_seconds=benchmark_seconds,
    )
    attn_score_sample = _benchmark(
        name="attention_score",
        work_per_iteration=float(attn_seq * attn_dim),
        units="mac/s",
        func=attn_score_kernel,
        min_seconds=benchmark_seconds,
    )
    attn_value_sample = _benchmark(
        name="attention_value",
        work_per_iteration=float(attn_seq * attn_dim),
        units="mac/s",
        func=attn_value_kernel,
        min_seconds=benchmark_seconds,
    )
    vector_sample = _benchmark(
        name="vector_ops",
        work_per_iteration=float(12 * vec_len),
        units="ops/s",
        func=vector_kernel,
        min_seconds=benchmark_seconds,
    )
    dma_sample = _benchmark(
        name="dma_copy",
        work_per_iteration=float(dma_src.nbytes),
        units="bytes/s",
        func=dma_kernel,
        min_seconds=benchmark_seconds,
    )

    attention_macs_per_sec = 0.5 * (
        attn_score_sample.throughput_per_s + attn_value_sample.throughput_per_s
    )

    notes = [
        "Calibration uses NumPy microbenchmarks on the local host.",
        "Derived hardware rates reflect the installed Python/NumPy backend as well as the machine itself.",
    ]
    if host.is_apple_silicon:
        notes.append("Apple Silicon host detected; perflevel sysctls were used when available.")
    if host.running_under_rosetta:
        notes.append("Warning: process appears to be running under Rosetta, which can distort comparisons.")

    return HostCalibration(
        host=host,
        projection_macs_per_sec=proj_sample.throughput_per_s,
        mlp_macs_per_sec=mlp_sample.throughput_per_s,
        attention_macs_per_sec=attention_macs_per_sec,
        vector_ops_per_sec=vector_sample.throughput_per_s,
        dma_bytes_per_sec=dma_sample.throughput_per_s,
        benchmark_seconds=benchmark_seconds,
        sequence_length_reference=attn_seq,
        samples={
            proj_sample.name: proj_sample,
            mlp_sample.name: mlp_sample,
            attn_score_sample.name: attn_score_sample,
            attn_value_sample.name: attn_value_sample,
            vector_sample.name: vector_sample,
            dma_sample.name: dma_sample,
        },
        notes=notes,
    )


def host_summary_rows(calibration: HostCalibration) -> List[Dict[str, Any]]:
    host = calibration.host
    return [
        {
            "metric": "chip",
            "value": host.chip,
        },
        {
            "metric": "model_identifier",
            "value": host.model_identifier,
        },
        {
            "metric": "logical_cpus",
            "value": host.total_logical_cpus,
        },
        {
            "metric": "performance_cores",
            "value": host.performance_cores,
        },
        {
            "metric": "efficiency_cores",
            "value": host.efficiency_cores,
        },
        {
            "metric": "memory_gib",
            "value": round(host.memory_bytes / GiB, 2) if host.memory_bytes else 0.0,
        },
        {
            "metric": "projection_gmac_per_s",
            "value": round(calibration.projection_macs_per_sec / 1e9, 3),
        },
        {
            "metric": "mlp_gmac_per_s",
            "value": round(calibration.mlp_macs_per_sec / 1e9, 3),
        },
        {
            "metric": "attention_gmac_per_s",
            "value": round(calibration.attention_macs_per_sec / 1e9, 3),
        },
        {
            "metric": "vector_gops_per_s",
            "value": round(calibration.vector_ops_per_sec / 1e9, 3),
        },
        {
            "metric": "dma_gbytes_per_s",
            "value": round(calibration.dma_bytes_per_sec / 1e9, 3),
        },
    ]
