from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import HardwareConfig, ModelConfig, PrecisionConfig


@dataclass
class TaskSpec:
    name: str
    engine: str
    cycles: int
    deps: List[str] = field(default_factory=list)
    stage: str = ""
    layer_idx: int = -1
    task_type: str = "compute"
    input_deps: List[str] = field(default_factory=list)


@dataclass
class ScheduledTask(TaskSpec):
    start: int = 0
    end: int = 0


@dataclass
class StageRecord:
    stage_key: str
    label: str
    layer_idx: int
    start: int
    end: int
    input_ready: int
    latency: int
    wait_for_data: int
    engine: str


@dataclass
class TokenSimulationResult:
    total_cycles: int
    total_time_ns: float
    tasks: List[ScheduledTask]
    stage_records: List[StageRecord]
    aggregate_stage_cycles: Dict[str, int]
    aggregate_category_cycles: Dict[str, int]
    kv_bytes_read: int
    kv_bytes_write: int
    weight_bytes: int
    embedding_bytes: int
    serial_cycles_no_overlap: int
    overlap_reduction: float

    def top_critical_stages(self, k: int = 5) -> List[Tuple[str, int]]:
        items = sorted(self.aggregate_category_cycles.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]


def _ceil_div(num: float, denom: float) -> int:
    return int(ceil(num / denom))


class EventScheduler:
    def __init__(self) -> None:
        self.engine_available: MutableMapping[str, int] = defaultdict(int)
        self.task_end: Dict[str, int] = {}

    def schedule(self, tasks: Sequence[TaskSpec]) -> List[ScheduledTask]:
        pending: Dict[str, TaskSpec] = {task.name: task for task in tasks}
        scheduled: List[ScheduledTask] = []
        while pending:
            ready_names = [
                name
                for name, task in pending.items()
                if all(dep in self.task_end for dep in task.deps)
            ]
            if not ready_names:
                missing = {name: [dep for dep in task.deps if dep not in self.task_end] for name, task in pending.items()}
                raise RuntimeError(f"cyclic or unsatisfied dependencies in task graph: {missing}")
            for name in sorted(ready_names):
                task = pending.pop(name)
                dep_ready = max((self.task_end.get(dep, 0) for dep in task.deps), default=0)
                start = max(dep_ready, self.engine_available[task.engine])
                end = start + max(int(task.cycles), 0)
                scheduled_task = ScheduledTask(**task.__dict__, start=start, end=end)
                scheduled.append(scheduled_task)
                self.engine_available[task.engine] = end
                self.task_end[task.name] = end
        scheduled.sort(key=lambda t: (t.start, t.end, t.name))
        return scheduled


class DecodeAcceleratorModel:
    """Analytical cycle-level model for single-token decode in a decoder-only LLM.

    The scheduler models stage dependencies, dedicated engines, DMA prefetch, and
    current-token KV bypass. It is intentionally lightweight but captures the
    latency/overlap trends that dominate small-batch inference.
    """

    def __init__(
        self,
        model: ModelConfig,
        hardware: HardwareConfig,
        precision: PrecisionConfig,
    ) -> None:
        self.model = model
        self.hardware = hardware
        self.precision = precision

    # ---------- Primitive cost models ----------
    def projection_cycles(self, macs: int) -> int:
        return _ceil_div(macs, self.hardware.projection_macs_per_cycle * self.hardware.compute_efficiency)

    def mlp_cycles(self, macs: int) -> int:
        return _ceil_div(macs, self.hardware.mlp_macs_per_cycle * self.hardware.compute_efficiency)

    def attention_cycles(self, macs: int) -> int:
        return _ceil_div(macs, self.hardware.attention_macs_per_cycle * self.hardware.compute_efficiency)

    def vector_cycles(self, ops: int) -> int:
        return _ceil_div(ops, self.hardware.vector_ops_per_cycle * self.hardware.compute_efficiency)

    def dma_cycles(self, nbytes: int) -> int:
        return _ceil_div(nbytes, self.hardware.dma_bytes_per_cycle * self.hardware.dma_efficiency)

    # ---------- Byte / MAC accounting ----------
    @property
    def qkv_out_dim(self) -> int:
        return self.model.d_model + 2 * self.model.kv_dim

    def qkv_macs(self) -> int:
        return self.model.d_model * self.qkv_out_dim

    def qkv_weight_bytes(self) -> int:
        return self.qkv_macs() * self.precision.weight_bytes

    def kv_write_bytes(self) -> int:
        return 2 * self.model.kv_dim * self.precision.kv_bytes

    def kv_read_bytes(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        old_tokens = max(seq_len - 1, 0) if self.hardware.use_bypass_for_current_kv else seq_len
        return 2 * old_tokens * self.model.kv_dim * self.precision.kv_bytes

    def attn_score_macs(self, seq_len: int) -> int:
        return seq_len * self.model.d_model

    def attn_value_macs(self, seq_len: int) -> int:
        return seq_len * self.model.d_model

    def o_proj_macs(self) -> int:
        return self.model.d_model * self.model.d_model

    def o_proj_weight_bytes(self) -> int:
        return self.o_proj_macs() * self.precision.weight_bytes

    def mlp_up_gate_macs(self) -> int:
        return 2 * self.model.ffn_dim * self.model.d_model

    def mlp_up_gate_weight_bytes(self) -> int:
        return self.mlp_up_gate_macs() * self.precision.weight_bytes

    def mlp_down_macs(self) -> int:
        return self.model.ffn_dim * self.model.d_model

    def mlp_down_weight_bytes(self) -> int:
        return self.mlp_down_macs() * self.precision.weight_bytes

    def logits_macs(self) -> int:
        return self.model.vocab_size * self.model.d_model

    def logits_weight_bytes(self) -> int:
        return self.logits_macs() * self.precision.weight_bytes

    def embedding_bytes(self) -> int:
        return 2 * self.model.d_model * self.precision.activation_bytes

    def layernorm_ops(self) -> int:
        return 8 * self.model.d_model

    def residual_ops(self) -> int:
        return self.model.d_model

    def swiglu_ops(self) -> int:
        return 4 * self.model.ffn_dim

    def softmax_ops(self, seq_len: int) -> int:
        return 5 * self.model.num_heads * seq_len

    def sample_ops(self) -> int:
        return 2 * self.model.vocab_size

    def kv_footprint_bytes(self, seq_len: int) -> int:
        return (
            self.model.num_layers
            * seq_len
            * 2
            * self.model.kv_dim
            * self.precision.kv_bytes
        )

    def per_token_kv_bytes(self) -> int:
        return 2 * self.model.kv_dim * self.precision.kv_bytes * self.model.num_layers

    # ---------- Task graph ----------
    def _embedding_tasks(self) -> List[TaskSpec]:
        return [
            TaskSpec(
                name="embedding_dma",
                engine="dma",
                cycles=self.dma_cycles(self.embedding_bytes()),
                deps=[],
                stage="embedding",
                layer_idx=-1,
                task_type="dma",
                input_deps=[],
            ),
            TaskSpec(
                name="embedding",
                engine="vector",
                cycles=self.vector_cycles(self.model.d_model),
                deps=["embedding_dma"],
                stage="embedding",
                layer_idx=-1,
                task_type="compute",
                input_deps=[],
            ),
        ]

    def _layer_tasks(self, layer_idx: int, seq_len: int, input_task: str) -> List[TaskSpec]:
        qkv_name = f"L{layer_idx}_qkv"
        tasks: List[TaskSpec] = [
            TaskSpec(
                name=f"L{layer_idx}_ln1",
                engine="vector",
                cycles=self.vector_cycles(self.layernorm_ops()),
                deps=[input_task],
                stage="ln1",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[input_task],
            ),
            TaskSpec(
                name=f"L{layer_idx}_qkv_dma",
                engine="dma",
                cycles=self.dma_cycles(self.qkv_weight_bytes()),
                deps=[input_task],
                stage="qkv",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[input_task],
            ),
            TaskSpec(
                name=qkv_name,
                engine="projection",
                cycles=self.projection_cycles(self.qkv_macs()),
                deps=[f"L{layer_idx}_ln1", f"L{layer_idx}_qkv_dma"],
                stage="qkv",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_ln1"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_kv_write",
                engine="dma",
                cycles=self.dma_cycles(self.kv_write_bytes()),
                deps=[qkv_name],
                stage="kv_write",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[qkv_name],
            ),
            TaskSpec(
                name=f"L{layer_idx}_attn_dma",
                engine="dma",
                cycles=self.dma_cycles(self.kv_read_bytes(seq_len)),
                deps=[input_task],
                stage="attn_score",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[input_task],
            ),
            TaskSpec(
                name=f"L{layer_idx}_attn_score",
                engine="attention",
                cycles=self.attention_cycles(self.attn_score_macs(seq_len)),
                deps=[qkv_name, f"L{layer_idx}_attn_dma"],
                stage="attn_score",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[qkv_name],
            ),
            TaskSpec(
                name=f"L{layer_idx}_softmax",
                engine="vector",
                cycles=self.vector_cycles(self.softmax_ops(seq_len)),
                deps=[f"L{layer_idx}_attn_score"],
                stage="softmax",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_attn_score"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_attn_value",
                engine="attention",
                cycles=self.attention_cycles(self.attn_value_macs(seq_len)),
                deps=[f"L{layer_idx}_softmax"],
                stage="attn_value",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_softmax"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_o_dma",
                engine="dma",
                cycles=self.dma_cycles(self.o_proj_weight_bytes()),
                deps=[qkv_name],
                stage="o_proj",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[qkv_name],
            ),
            TaskSpec(
                name=f"L{layer_idx}_o_proj",
                engine="projection",
                cycles=self.projection_cycles(self.o_proj_macs()),
                deps=[f"L{layer_idx}_attn_value", f"L{layer_idx}_o_dma"],
                stage="o_proj",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_attn_value"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_residual1",
                engine="vector",
                cycles=self.vector_cycles(self.residual_ops()),
                deps=[f"L{layer_idx}_o_proj"],
                stage="residual1",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_o_proj"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_ln2",
                engine="vector",
                cycles=self.vector_cycles(self.layernorm_ops()),
                deps=[f"L{layer_idx}_residual1"],
                stage="ln2",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_residual1"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_mlp_up_dma",
                engine="dma",
                cycles=self.dma_cycles(self.mlp_up_gate_weight_bytes()),
                deps=[f"L{layer_idx}_attn_value"],
                stage="mlp_up_gate",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[f"L{layer_idx}_attn_value"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_mlp_up_gate",
                engine="mlp",
                cycles=self.mlp_cycles(self.mlp_up_gate_macs()),
                deps=[f"L{layer_idx}_ln2", f"L{layer_idx}_mlp_up_dma"],
                stage="mlp_up_gate",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_ln2"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_swiglu",
                engine="vector",
                cycles=self.vector_cycles(self.swiglu_ops()),
                deps=[f"L{layer_idx}_mlp_up_gate"],
                stage="swiglu",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_mlp_up_gate"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_mlp_down_dma",
                engine="dma",
                cycles=self.dma_cycles(self.mlp_down_weight_bytes()),
                deps=[f"L{layer_idx}_mlp_up_gate"],
                stage="mlp_down",
                layer_idx=layer_idx,
                task_type="dma",
                input_deps=[f"L{layer_idx}_mlp_up_gate"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_mlp_down",
                engine="mlp",
                cycles=self.mlp_cycles(self.mlp_down_macs()),
                deps=[f"L{layer_idx}_swiglu", f"L{layer_idx}_mlp_down_dma"],
                stage="mlp_down",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_swiglu"],
            ),
            TaskSpec(
                name=f"L{layer_idx}_residual2",
                engine="vector",
                cycles=self.vector_cycles(self.residual_ops()),
                deps=[f"L{layer_idx}_mlp_down"],
                stage="residual2",
                layer_idx=layer_idx,
                task_type="compute",
                input_deps=[f"L{layer_idx}_mlp_down"],
            ),
        ]

        if not self.hardware.use_bypass_for_current_kv:
            # Conservative mode: attention waits for KV cache persistence.
            for task in tasks:
                if task.name == f"L{layer_idx}_attn_score":
                    task.deps.append(f"L{layer_idx}_kv_write")
        return tasks

    def _logits_tasks(self, input_task: str) -> List[TaskSpec]:
        return [
            TaskSpec(
                name="logits_dma",
                engine="dma",
                cycles=self.dma_cycles(self.logits_weight_bytes()),
                deps=[input_task],
                stage="logits",
                layer_idx=self.model.num_layers,
                task_type="dma",
                input_deps=[input_task],
            ),
            TaskSpec(
                name="logits",
                engine="projection",
                cycles=self.projection_cycles(self.logits_macs()),
                deps=[input_task, "logits_dma"],
                stage="logits",
                layer_idx=self.model.num_layers,
                task_type="compute",
                input_deps=[input_task],
            ),
            TaskSpec(
                name="sample",
                engine="vector",
                cycles=self.vector_cycles(self.sample_ops()),
                deps=["logits"],
                stage="sample",
                layer_idx=self.model.num_layers,
                task_type="compute",
                input_deps=["logits"],
            ),
        ]

    def build_task_graph(self, seq_len: int) -> List[TaskSpec]:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        tasks = self._embedding_tasks()
        input_task = "embedding"
        for layer_idx in range(self.model.num_layers):
            layer_tasks = self._layer_tasks(layer_idx=layer_idx, seq_len=seq_len, input_task=input_task)
            tasks.extend(layer_tasks)
            input_task = f"L{layer_idx}_residual2"
        tasks.extend(self._logits_tasks(input_task))
        return tasks

    # ---------- Analysis helpers ----------
    @staticmethod
    def _stage_label(task: ScheduledTask) -> str:
        if task.layer_idx >= 0 and task.layer_idx < 10_000 and task.stage not in {"embedding", "logits", "sample"}:
            return f"L{task.layer_idx}.{task.stage}"
        return task.stage

    @staticmethod
    def _stage_category(stage: str) -> str:
        mapping = {
            "embedding": "Embedding/dispatch",
            "ln1": "Attention front-end",
            "qkv": "Attention front-end",
            "kv_write": "KV-cache write",
            "attn_score": "KV read + attention score",
            "softmax": "Softmax",
            "attn_value": "Attention value mix",
            "o_proj": "Attention output/residual",
            "residual1": "Attention output/residual",
            "ln2": "MLP front-end",
            "mlp_up_gate": "MLP",
            "swiglu": "MLP",
            "mlp_down": "MLP",
            "residual2": "MLP output/residual",
            "logits": "Logits/sample",
            "sample": "Logits/sample",
        }
        return mapping.get(stage, stage)

    def _build_stage_records(self, scheduled: Sequence[ScheduledTask]) -> List[StageRecord]:
        task_lookup = {task.name: task for task in scheduled}
        records: List[StageRecord] = []
        for task in scheduled:
            if task.task_type != "compute":
                continue
            input_ready = max((task_lookup[dep].end for dep in task.input_deps), default=0)
            latency = task.end - input_ready
            wait_for_data = task.start - input_ready
            records.append(
                StageRecord(
                    stage_key=task.name,
                    label=self._stage_label(task),
                    layer_idx=task.layer_idx,
                    start=task.start,
                    end=task.end,
                    input_ready=input_ready,
                    latency=latency,
                    wait_for_data=wait_for_data,
                    engine=task.engine,
                )
            )
        return records

    def simulate_token(self, seq_len: int) -> TokenSimulationResult:
        task_graph = self.build_task_graph(seq_len=seq_len)
        scheduled = EventScheduler().schedule(task_graph)
        total_cycles = max(task.end for task in scheduled) if scheduled else 0
        stage_records = self._build_stage_records(scheduled)

        aggregate_stage_cycles: Dict[str, int] = defaultdict(int)
        aggregate_category_cycles: Dict[str, int] = defaultdict(int)
        for record in stage_records:
            stage_name = record.label.split(".")[-1]
            aggregate_stage_cycles[record.label] += record.latency
            aggregate_category_cycles[self._stage_category(stage_name)] += record.latency

        kv_bytes_read = self.model.num_layers * self.kv_read_bytes(seq_len)
        kv_bytes_write = self.model.num_layers * self.kv_write_bytes()
        weight_bytes = self.model.num_layers * (
            self.qkv_weight_bytes()
            + self.o_proj_weight_bytes()
            + self.mlp_up_gate_weight_bytes()
            + self.mlp_down_weight_bytes()
        ) + self.logits_weight_bytes()
        embedding_bytes = self.embedding_bytes()

        serial_cycles_no_overlap = sum(task.cycles for task in task_graph)
        overlap_reduction = 0.0
        if serial_cycles_no_overlap > 0:
            overlap_reduction = 1.0 - (total_cycles / serial_cycles_no_overlap)

        return TokenSimulationResult(
            total_cycles=total_cycles,
            total_time_ns=total_cycles / self.hardware.clock_ghz,
            tasks=scheduled,
            stage_records=stage_records,
            aggregate_stage_cycles=dict(aggregate_stage_cycles),
            aggregate_category_cycles=dict(aggregate_category_cycles),
            kv_bytes_read=kv_bytes_read,
            kv_bytes_write=kv_bytes_write,
            weight_bytes=weight_bytes,
            embedding_bytes=embedding_bytes,
            serial_cycles_no_overlap=serial_cycles_no_overlap,
            overlap_reduction=overlap_reduction,
        )

    def simulate_sweep(self, seq_lens: Iterable[int]) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for seq_len in seq_lens:
            result = self.simulate_token(seq_len)
            row: Dict[str, float] = {
                "seq_len": float(seq_len),
                "total_cycles": float(result.total_cycles),
                "total_time_ns": float(result.total_time_ns),
                "kv_bytes_read": float(result.kv_bytes_read),
                "kv_bytes_write": float(result.kv_bytes_write),
                "weight_bytes": float(result.weight_bytes),
                "overlap_reduction": float(result.overlap_reduction),
                "kv_footprint_bytes": float(self.kv_footprint_bytes(seq_len)),
            }
            for key, value in result.aggregate_category_cycles.items():
                row[f"category::{key}"] = float(value)
            rows.append(row)
        return rows

    def latency_breakdown(self, seq_len: int) -> Dict[str, int]:
        result = self.simulate_token(seq_len)
        return result.aggregate_category_cycles

    def estimate_kv_bandwidth_per_token(self, seq_len: int) -> int:
        return self.model.num_layers * (
            self.kv_read_bytes(seq_len) + self.kv_write_bytes()
        )

    def identify_primary_bottleneck(self, seq_len: int) -> str:
        result = self.simulate_token(seq_len)
        dominant = max(result.aggregate_category_cycles.items(), key=lambda kv: kv[1])[0]
        return dominant

    def memory_summary(self, seq_len: int) -> Dict[str, float]:
        return {
            "seq_len": float(seq_len),
            "kv_footprint_bytes": float(self.kv_footprint_bytes(seq_len)),
            "kv_bandwidth_bytes_per_token": float(self.estimate_kv_bandwidth_per_token(seq_len)),
            "weight_stream_bytes_per_token": float(self.model.num_layers * (
                self.qkv_weight_bytes()
                + self.o_proj_weight_bytes()
                + self.mlp_up_gate_weight_bytes()
                + self.mlp_down_weight_bytes()
            ) + self.logits_weight_bytes()),
            "embedding_bytes_per_token": float(self.embedding_bytes()),
        }
