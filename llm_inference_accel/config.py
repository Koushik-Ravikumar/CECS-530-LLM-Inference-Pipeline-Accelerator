from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Configurable transformer shape used by both the functional model and the
    analytical accelerator model.
    """

    vocab_size: int = 256
    max_seq_len: int = 128
    num_layers: int = 2
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    ffn_dim: int = 512
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def parameterized_stage_names(self) -> List[str]:
        return [
            "embedding",
            "ln1",
            "qkv",
            "kv_write",
            "attn_score",
            "softmax",
            "attn_value",
            "o_proj",
            "residual1",
            "ln2",
            "mlp_up_gate",
            "swiglu",
            "mlp_down",
            "residual2",
            "logits",
            "sample",
        ]


@dataclass
class PrecisionConfig:
    activation_bytes: int = 2
    weight_bytes: int = 1
    kv_bytes: int = 2
    accumulator_bytes: int = 4
    activation_name: str = "BF16"
    weight_name: str = "INT8"
    kv_name: str = "BF16"


@dataclass
class HardwareConfig:
    """Micro-architectural knobs for the cycle-level decode model."""

    projection_macs_per_cycle: float = 2048.0
    mlp_macs_per_cycle: float = 2048.0
    attention_macs_per_cycle: float = 1024.0
    vector_ops_per_cycle: float = 256.0
    dma_bytes_per_cycle: float = 256.0
    compute_efficiency: float = 0.80
    dma_efficiency: float = 0.85
    scratchpad_bytes: int = 2 * 1024 * 1024
    weight_buffer_bytes: int = 512 * 1024
    kv_stream_buffer_bytes: int = 512 * 1024
    clock_ghz: float = 1.0
    use_bypass_for_current_kv: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.compute_efficiency <= 1.0):
            raise ValueError("compute_efficiency must be in (0, 1]")
        if not (0 < self.dma_efficiency <= 1.0):
            raise ValueError("dma_efficiency must be in (0, 1]")


@dataclass
class RuntimeConfig:
    temperature: float = 1.0
    top_k: int = 0
    greedy: bool = True


@dataclass
class BaselineScenario:
    name: str
    model: ModelConfig
    hardware: HardwareConfig
    precision: PrecisionConfig
