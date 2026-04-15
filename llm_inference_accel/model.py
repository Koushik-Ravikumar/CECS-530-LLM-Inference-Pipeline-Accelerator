from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ModelConfig, RuntimeConfig


Array = np.ndarray


@dataclass
class LayerWeights:
    ln1_gamma: Array
    ln1_beta: Array
    w_q: Array
    w_k: Array
    w_v: Array
    w_o: Array
    ln2_gamma: Array
    ln2_beta: Array
    w_gate: Array
    w_up: Array
    w_down: Array


@dataclass
class TransformerWeights:
    token_embedding: Array
    position_embedding: Array
    layers: List[LayerWeights]
    lm_head: Array


@dataclass
class KVCache:
    keys: List[Array]
    values: List[Array]
    length: int = 0

    @classmethod
    def allocate(cls, config: ModelConfig) -> "KVCache":
        key_shape = (config.max_seq_len, config.num_kv_heads, config.head_dim)
        keys = [np.zeros(key_shape, dtype=np.float32) for _ in range(config.num_layers)]
        values = [np.zeros(key_shape, dtype=np.float32) for _ in range(config.num_layers)]
        return cls(keys=keys, values=values, length=0)

    def clone_prefix(self, length: int) -> "KVCache":
        return KVCache(
            keys=[k[:length].copy() for k in self.keys],
            values=[v[:length].copy() for v in self.values],
            length=length,
        )


@dataclass
class DecodeResult:
    logits_per_step: List[Array]
    hidden_per_step: List[Array]
    sampled_tokens: List[int]
    cache: KVCache


@dataclass
class ValidationResult:
    max_abs_error: float
    mean_abs_error: float
    passed: bool
    cache_lengths: List[int]
    sampled_tokens: List[int]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _scaled_matrix(rng: np.random.Generator, shape: Tuple[int, ...], scale: float) -> Array:
    return rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float32)


def initialize_weights(config: ModelConfig, seed: int = 7) -> TransformerWeights:
    rng = _rng(seed)
    d = config.d_model
    h = config.ffn_dim
    kv_dim = config.kv_dim

    token_embedding = _scaled_matrix(rng, (config.vocab_size, d), 0.02)
    position_embedding = _scaled_matrix(rng, (config.max_seq_len, d), 0.01)

    layers: List[LayerWeights] = []
    attn_scale = 1.0 / np.sqrt(d)
    mlp_scale = 1.0 / np.sqrt(h)
    out_scale = 1.0 / np.sqrt(max(d, h))

    for _ in range(config.num_layers):
        layers.append(
            LayerWeights(
                ln1_gamma=np.ones((d,), dtype=np.float32),
                ln1_beta=np.zeros((d,), dtype=np.float32),
                w_q=_scaled_matrix(rng, (d, d), attn_scale),
                w_k=_scaled_matrix(rng, (kv_dim, d), attn_scale),
                w_v=_scaled_matrix(rng, (kv_dim, d), attn_scale),
                w_o=_scaled_matrix(rng, (d, d), attn_scale),
                ln2_gamma=np.ones((d,), dtype=np.float32),
                ln2_beta=np.zeros((d,), dtype=np.float32),
                w_gate=_scaled_matrix(rng, (h, d), mlp_scale),
                w_up=_scaled_matrix(rng, (h, d), mlp_scale),
                w_down=_scaled_matrix(rng, (d, h), out_scale),
            )
        )

    lm_head = _scaled_matrix(rng, (config.vocab_size, d), 0.02)
    return TransformerWeights(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        layers=layers,
        lm_head=lm_head,
    )


def layer_norm(x: Array, gamma: Array, beta: Array, eps: float) -> Array:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


def silu(x: Array) -> Array:
    return x / (1.0 + np.exp(-x))


def _expand_kv(kv: Array, group_size: int) -> Array:
    if group_size == 1:
        return kv
    return np.repeat(kv, repeats=group_size, axis=1)


def _single_token_attention(
    q: Array,
    k_cache: Array,
    v_cache: Array,
    config: ModelConfig,
) -> Array:
    """q shape [num_heads, head_dim], k/v cache shape [T, num_kv_heads, head_dim]."""
    k_expanded = _expand_kv(k_cache, config.kv_group_size)
    v_expanded = _expand_kv(v_cache, config.kv_group_size)
    scores = np.einsum("hd,thd->ht", q, k_expanded, optimize=True)
    scores = scores / np.sqrt(config.head_dim)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    context = np.einsum("ht,thd->hd", probs, v_expanded, optimize=True)
    return context.reshape(-1)


def _full_causal_attention(q: Array, k: Array, v: Array, config: ModelConfig) -> Array:
    """Tensor shapes: q [T, H, D], k/v [T, KVH, D]. Returns [T, d_model]."""
    k_expanded = _expand_kv(k, config.kv_group_size)
    v_expanded = _expand_kv(v, config.kv_group_size)
    t = q.shape[0]
    scale = 1.0 / np.sqrt(config.head_dim)
    scores = np.einsum("thd,shd->ths", q, k_expanded, optimize=True) * scale
    causal_mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores = np.where(causal_mask[:, None, :], -1e30, scores)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores - max_scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    context = np.einsum("ths,shd->thd", probs, v_expanded, optimize=True)
    return context.reshape(t, -1)


def decode_next_token(
    token_id: int,
    position: int,
    cache: KVCache,
    weights: TransformerWeights,
    config: ModelConfig,
    runtime: RuntimeConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[Array, int, Array]:
    """Performs single-token autoregressive decode through the configured model.

    Returns logits, sampled_token, final hidden state.
    """
    if position >= config.max_seq_len:
        raise ValueError("position exceeds configured max_seq_len")

    if runtime is None:
        runtime = RuntimeConfig()
    if rng is None:
        rng = _rng(0)

    x = weights.token_embedding[token_id] + weights.position_embedding[position]
    x = x.astype(np.float32, copy=False)

    for layer_idx, layer in enumerate(weights.layers):
        x_norm = layer_norm(x, layer.ln1_gamma, layer.ln1_beta, config.eps)

        q = layer.w_q @ x_norm
        k = layer.w_k @ x_norm
        v = layer.w_v @ x_norm

        q = q.reshape(config.num_heads, config.head_dim)
        k = k.reshape(config.num_kv_heads, config.head_dim)
        v = v.reshape(config.num_kv_heads, config.head_dim)

        cache.keys[layer_idx][position] = k
        cache.values[layer_idx][position] = v

        k_prefix = cache.keys[layer_idx][: position + 1]
        v_prefix = cache.values[layer_idx][: position + 1]
        context = _single_token_attention(q, k_prefix, v_prefix, config)
        attn_out = layer.w_o @ context
        x = x + attn_out

        x_norm2 = layer_norm(x, layer.ln2_gamma, layer.ln2_beta, config.eps)
        gate = layer.w_gate @ x_norm2
        up = layer.w_up @ x_norm2
        mlp_hidden = silu(gate) * up
        mlp_out = layer.w_down @ mlp_hidden
        x = x + mlp_out

    cache.length = max(cache.length, position + 1)
    logits = weights.lm_head @ x
    sampled_token = sample_from_logits(logits, runtime=runtime, rng=rng)
    return logits.astype(np.float32), sampled_token, x.astype(np.float32)


def decode_sequence_incremental(
    token_ids: Sequence[int],
    weights: TransformerWeights,
    config: ModelConfig,
    runtime: RuntimeConfig | None = None,
    seed: int = 0,
) -> DecodeResult:
    cache = KVCache.allocate(config)
    logits_per_step: List[Array] = []
    hidden_per_step: List[Array] = []
    sampled_tokens: List[int] = []
    rng = _rng(seed)

    for pos, token_id in enumerate(token_ids):
        logits, sampled, hidden = decode_next_token(
            token_id=token_id,
            position=pos,
            cache=cache,
            weights=weights,
            config=config,
            runtime=runtime,
            rng=rng,
        )
        logits_per_step.append(logits)
        hidden_per_step.append(hidden)
        sampled_tokens.append(sampled)

    return DecodeResult(
        logits_per_step=logits_per_step,
        hidden_per_step=hidden_per_step,
        sampled_tokens=sampled_tokens,
        cache=cache,
    )


def forward_full_sequence(
    token_ids: Sequence[int],
    weights: TransformerWeights,
    config: ModelConfig,
) -> List[Array]:
    t = len(token_ids)
    if t > config.max_seq_len:
        raise ValueError("sequence length exceeds configured max_seq_len")

    positions = np.arange(t)
    x = weights.token_embedding[np.asarray(token_ids)] + weights.position_embedding[positions]
    x = x.astype(np.float32, copy=False)

    for layer in weights.layers:
        x_norm = layer_norm(x, layer.ln1_gamma, layer.ln1_beta, config.eps)
        q = (x_norm @ layer.w_q.T).reshape(t, config.num_heads, config.head_dim)
        k = (x_norm @ layer.w_k.T).reshape(t, config.num_kv_heads, config.head_dim)
        v = (x_norm @ layer.w_v.T).reshape(t, config.num_kv_heads, config.head_dim)
        context = _full_causal_attention(q, k, v, config)
        attn_out = context @ layer.w_o.T
        x = x + attn_out

        x_norm2 = layer_norm(x, layer.ln2_gamma, layer.ln2_beta, config.eps)
        gate = x_norm2 @ layer.w_gate.T
        up = x_norm2 @ layer.w_up.T
        mlp_hidden = silu(gate) * up
        mlp_out = mlp_hidden @ layer.w_down.T
        x = x + mlp_out

    logits = x @ weights.lm_head.T
    return [row.astype(np.float32) for row in logits]


def sample_from_logits(
    logits: Array,
    runtime: RuntimeConfig,
    rng: np.random.Generator,
) -> int:
    if runtime.greedy:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64)
    if runtime.temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = scaled / runtime.temperature

    if runtime.top_k > 0 and runtime.top_k < scaled.size:
        top_idx = np.argpartition(scaled, -runtime.top_k)[-runtime.top_k :]
        masked = np.full_like(scaled, fill_value=-1e30, dtype=np.float64)
        masked[top_idx] = scaled[top_idx]
        scaled = masked

    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(logits.size), p=probs))


def validate_incremental_decode(
    token_ids: Sequence[int],
    weights: TransformerWeights,
    config: ModelConfig,
) -> ValidationResult:
    full_logits = forward_full_sequence(token_ids, weights, config)
    incremental = decode_sequence_incremental(token_ids, weights, config)

    errors = []
    for full, inc in zip(full_logits, incremental.logits_per_step):
        errors.append(np.abs(full - inc))
    all_errors = np.concatenate(errors) if errors else np.zeros((1,), dtype=np.float32)
    max_abs_error = float(np.max(all_errors))
    mean_abs_error = float(np.mean(all_errors))
    cache_lengths = [incremental.cache.length for _ in range(config.num_layers)]

    return ValidationResult(
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        passed=max_abs_error < 1e-4,
        cache_lengths=cache_lengths,
        sampled_tokens=incremental.sampled_tokens,
    )


def summarize_kv_cache_shapes(config: ModelConfig) -> Dict[str, int]:
    return {
        "num_layers": config.num_layers,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "max_seq_len": config.max_seq_len,
        "per_layer_slots": config.max_seq_len * config.num_kv_heads * config.head_dim,
        "total_slots": config.num_layers * config.max_seq_len * config.num_kv_heads * config.head_dim * 2,
    }
