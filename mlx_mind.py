"""
MIND — Mixture of Intelligent Neural Dynamics  (MLX Port)
==========================================================

Apple-Silicon-native reimplementation of the MIND ~1B Mixture-of-Experts
causal language model using the `MLX <https://ml-explore.github.io/mlx>`_
framework for accelerated pretraining, finetuning, and inference on Metal
GPUs with unified memory.

Every module, hyperparameter, and architectural decision mirrors the
PyTorch reference ``mind.py``; only the backend differs.  Consult the
original for detailed docstrings and mathematical background.

Key MLX-Specific Adaptations
-----------------------------
* ``nn.Parameter`` → public ``self.name = mx.array(...)`` so MLX's
  auto-discovery treats them as trainable.  Non-trainable buffers use
  ``self._name`` (private, skipped by ``nn.Module.parameters()``).
* ``register_buffer("inv_freq")`` → ``self._inv_freq`` (prefixed ``_``).
* ``nn.ModuleList`` → plain Python ``list`` (MLX auto-discovers attrs).
* ``nn.SiLU()`` → ``nn.silu`` (function) or inline ``mx.sigmoid(x) * x``.
* ``F.one_hot(x, C)`` → ``mx.one_hot(x, C)``, but note: mlx one_hot
  expects integer input and produces float output.
* ``F.softmax(..., dtype=float32)`` → ``mx.softmax(x.astype(mx.float32))``.
* ``Tensor.detach()`` → ``mx.stop_gradient()``.
* ``torch.where(mask)`` → explicit boolean indexing or gather patterns.
* ``index_add_`` → functional ``mx.zeros(...).at[idx].add(vals)`` or
  loop-based accumulation (MLX supports slice-based indexing).
* Weight tying (embed ↔ LM head) handled by referencing the same array.
* All ``forward()`` methods become ``__call__()`` in MLX ``nn.Module``.

Architecture (identical to PyTorch ``mind.py``)
------------------------------------------------
  Tier 1 — SENSORY   (layers 0–5):   GQA Attention + Dense SwiGLU FFN
  Tier 2 — ASSOCIATIVE (layers 6–17): GQA Attention + Cognitive MoE
  Tier 3 — EXECUTIVE (layers 18–23):  GQA Attention + Cognitive MoE + FracToM

  ~1.006B total parameters, ~576M active per token (57%).

Requirements
------------
    Python ≥ 3.9,  mlx ≥ 0.20

Usage
-----
    import mlx.core as mx
    from mlx_mind import MindForCausalLM, MindConfig

    config = MindConfig()
    model = MindForCausalLM(config)
    input_ids = mx.random.randint(0, 32000, shape=(2, 128))
    output = model(input_ids)
    logits = output.logits   # (2, 128, 32000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


COGNITIVE_MODULE_NAMES = [
    "Analytical",    # Logic, maths, structured reasoning
    "Linguistic",    # Grammar, semantics, pragmatics
    "Associative",   # Pattern matching, analogy, memory retrieval
    "Social",        # ToM, empathy, social inference
]

COGNITIVE_TIER_LABELS = {
    "sensory": "Sensory — primary feature extraction",
    "associative": "Associative — multi-modal cognitive MoE",
    "executive": "Executive — FracToM ToM + causal reasoning",
}

PEARL_HIERARCHY_LABELS = {
    0: "Association (seeing)",
    1: "Intervention (doing)",
    2: "Counterfactual (imagining)",
}


@dataclass
class MindConfig:
    """Configuration for the MIND (Mixture of Intelligent Neural Dynamics)
    architecture.

    Default values yield a ~1.0B parameter model.
    """

    # ── Token Embedding ──────────────────────────────────────────────
    vocab_size: int = 32_000
    hidden_size: int = 1536
    max_position_embeddings: int = 32_768

    # ── Transformer Backbone ─────────────────────────────────────────
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 96          # hidden_size // num_attention_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_dropout: float = 0.0
    hidden_act: str = "silu"

    # ── Dense FFN (Sensory Layers) ───────────────────────────────────
    dense_intermediate_size: int = 4096

    # ── Cognitive MoE ────────────────────────────────────────────────
    num_cognitive_modules: int = 4          # analytical, linguistic, ...
    experts_per_module: int = 4             # 4 modules × 4 = 16 total
    expert_intermediate_size: int = 384     # small per-expert FFN
    shared_expert_intermediate_size: int = 2048  # always-active expert
    num_experts_per_tok: int = 4            # active experts per token
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001

    # ── Cognitive Tier Boundaries ────────────────────────────────────
    num_sensory_layers: int = 6             # bottom:  dense FFN
    num_associative_layers: int = 12        # middle:  cognitive MoE
    num_executive_layers: int = 6           # top:     MoE + FracToM

    # ── FracToM Integration (Executive Layers) ──────────────────────
    bdi_factor_dim: int = 256               # per BDI component dimension
    fractom_dropout: float = 0.1

    # ── Initialisation ───────────────────────────────────────────────
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    # ── Derived properties ───────────────────────────────────────────

    @property
    def total_experts(self) -> int:
        return self.num_cognitive_modules * self.experts_per_module

    @property
    def sensory_layer_range(self) -> range:
        return range(0, self.num_sensory_layers)

    @property
    def associative_layer_range(self) -> range:
        s = self.num_sensory_layers
        return range(s, s + self.num_associative_layers)

    @property
    def executive_layer_range(self) -> range:
        s = self.num_sensory_layers + self.num_associative_layers
        return range(s, s + self.num_executive_layers)

    def layer_tier(self, layer_idx: int) -> str:
        if layer_idx in self.sensory_layer_range:
            return "sensory"
        elif layer_idx in self.associative_layer_range:
            return "associative"
        else:
            return "executive"

    def executive_depth(self, layer_idx: int) -> int:
        """Mentalizing depth (0-indexed) for an executive layer."""
        return layer_idx - (self.num_sensory_layers + self.num_associative_layers)

    def __post_init__(self):
        assert self.num_hidden_layers == (
            self.num_sensory_layers
            + self.num_associative_layers
            + self.num_executive_layers
        ), (
            f"Layer counts must sum to num_hidden_layers: "
            f"{self.num_sensory_layers} + {self.num_associative_layers} + "
            f"{self.num_executive_layers} ≠ {self.num_hidden_layers}"
        )
        assert self.hidden_size == self.num_attention_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal "
            f"num_attention_heads ({self.num_attention_heads}) × "
            f"head_dim ({self.head_dim})"
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      OUTPUT CONTAINERS                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


class BDITensor:
    """Sequence-aware Belief-Desire-Intention container.

    All three arrays share shape ``(batch, seq_len, factor_dim)`` (or
    ``(batch, factor_dim)`` for pooled representations).
    """

    __slots__ = ("belief", "desire", "intention")

    def __init__(self, belief: mx.array, desire: mx.array, intention: mx.array):
        self.belief = belief
        self.desire = desire
        self.intention = intention

    def pack(self) -> mx.array:
        """Pack → ``(…, 3 × factor_dim)``."""
        return mx.concatenate([self.belief, self.desire, self.intention], axis=-1)

    @staticmethod
    def unpack(x: mx.array, factor_dim: int) -> "BDITensor":
        b = x[..., :factor_dim]
        d = x[..., factor_dim:2 * factor_dim]
        i = x[..., 2 * factor_dim:]
        return BDITensor(b, d, i)

    def stop_gradient(self) -> "BDITensor":
        """MLX equivalent of ``detach()``."""
        return BDITensor(
            mx.stop_gradient(self.belief),
            mx.stop_gradient(self.desire),
            mx.stop_gradient(self.intention),
        )


@dataclass
class MindOutput:
    """Output from the ``MindModel`` backbone (no LM head)."""

    last_hidden_state: mx.array
    past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None
    router_probs: Optional[Tuple[mx.array, ...]] = None
    bdi_states: Optional[List[BDITensor]] = None
    cognitive_stats: Optional[Dict[str, Any]] = None


@dataclass
class MindCausalLMOutput:
    """Output from ``MindForCausalLM``."""

    loss: Optional[mx.array] = None
    aux_loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None
    router_probs: Optional[Tuple[mx.array, ...]] = None
    bdi_states: Optional[List[BDITensor]] = None
    cognitive_stats: Optional[Dict[str, Any]] = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     UTILITY FUNCTIONS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


def clip_grad_norm(grads, max_norm: float):
    """Clip gradient tree by global L2 norm."""
    from mlx.utils import tree_flatten, tree_map

    flat = tree_flatten(grads)
    total_sq = sum(mx.sum(g * g).item() for _, g in flat)
    total_norm = math.sqrt(total_sq)
    coeff = max_norm / (total_norm + 1e-6)
    if coeff < 1.0:
        grads = tree_map(lambda g: g * coeff, grads)
    return grads


def _one_hot(indices: mx.array, num_classes: int) -> mx.array:
    """One-hot encoding (manual implementation for MLX compatibility).

    Parameters
    ----------
    indices : integer array of any shape.
    num_classes : number of classes.

    Returns
    -------
    Float array with shape ``(*indices.shape, num_classes)``.
    """
    return (mx.expand_dims(indices, axis=-1) == mx.arange(num_classes)).astype(mx.float32)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     CORE BUILDING BLOCKS                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Used throughout in place of LayerNorm, following Qwen3/LLaMA convention.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self._eps)
        return (self.weight * x).astype(dtype)


# ──────────────── Rotary Position Embedding ──────────────────────────


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021) with configurable θ.

    Implements the same RoPE variant used in Qwen3-MoE.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32_768,
        theta: float = 1_000_000.0,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )
        # Non-trainable buffer: underscore prefix
        self._inv_freq = inv_freq
        self._max_seq_len = max_seq_len

    def __call__(
        self, x: mx.array, position_ids: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Parameters
        ----------
        x : (B, S, D) — used only for dtype.
        position_ids : (B, S) — absolute position indices.

        Returns
        -------
        cos, sin : each (B, S, head_dim).
        """
        B = position_ids.shape[0]
        # (B, head_dim//2, 1)
        inv_freq = mx.broadcast_to(
            self._inv_freq[None, :, None],
            (B, self._inv_freq.shape[0], 1),
        ).astype(mx.float32)
        pos = position_ids[:, None, :].astype(mx.float32)   # (B, 1, S)
        freqs = mx.transpose(inv_freq @ pos, axes=(0, 2, 1))  # (B, S, hd//2)
        emb = mx.concatenate([freqs, freqs], axis=-1)         # (B, S, hd)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the second half of the last dimension."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
    unsqueeze_dim: int = 1,
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE to query and key arrays."""
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)  # (B, 1, S, D) for head dim
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_emb = q * cos + _rotate_half(q) * sin
    k_emb = k * cos + _rotate_half(k) * sin
    return q_emb, k_emb


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads for grouped-query attention."""
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    x = mx.expand_dims(x, axis=2)       # (B, H, 1, S, D)
    x = mx.broadcast_to(x, (B, H, n_rep, S, D))
    return mx.reshape(x, (B, H * n_rep, S, D))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  GROUPED-QUERY ATTENTION                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveAttention(nn.Module):
    """Grouped-Query Attention with QK-norm and RoPE.

    Structurally identical to Qwen3MoeAttention: the same attention is
    used across all three cognitive tiers.
    """

    def __init__(self, config: MindConfig, layer_idx: int):
        super().__init__()
        self._hidden_size = config.hidden_size
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        self._num_kv_groups = self._num_heads // self._num_kv_heads
        self._scaling = self._head_dim ** -0.5
        self._layer_idx = layer_idx

        self.q_proj = nn.Linear(
            self._hidden_size, self._num_heads * self._head_dim, bias=False,
        )
        self.k_proj = nn.Linear(
            self._hidden_size, self._num_kv_heads * self._head_dim, bias=False,
        )
        self.v_proj = nn.Linear(
            self._hidden_size, self._num_kv_heads * self._head_dim, bias=False,
        )
        self.o_proj = nn.Linear(
            self._num_heads * self._head_dim, self._hidden_size, bias=False,
        )

        # QK-norm on head_dim (Qwen3 innovation — stabilises training)
        self.q_norm = RMSNorm(self._head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self._head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Parameters
        ----------
        hidden_states : (B, S, D)
        position_embeddings : (cos, sin), each (B, S, head_dim)
        attention_mask : (1, 1, S, T) additive mask (0 = attend, -inf = mask)
        past_key_value : optional (K_past, V_past)

        Returns
        -------
        output : (B, S, D)
        present_kv : (K, V) — updated KV cache arrays.
        """
        B, S, _ = hidden_states.shape

        # Project & reshape
        q = mx.reshape(
            self.q_proj(hidden_states),
            (B, S, self._num_heads, self._head_dim),
        )
        k = mx.reshape(
            self.k_proj(hidden_states),
            (B, S, self._num_kv_heads, self._head_dim),
        )
        v = mx.reshape(
            self.v_proj(hidden_states),
            (B, S, self._num_kv_heads, self._head_dim),
        )

        # QK-norm (applied before transpose, on head_dim axis)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, heads, S, head_dim)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache concatenation
        if past_key_value is not None:
            k = mx.concatenate([past_key_value[0], k], axis=2)
            v = mx.concatenate([past_key_value[1], v], axis=2)
        present_kv = (k, v)

        # GQA: expand KV heads to match query heads
        k = repeat_kv(k, self._num_kv_groups)
        v = repeat_kv(v, self._num_kv_groups)

        # Scaled dot-product attention
        attn_weights = (q @ mx.transpose(k, axes=(0, 1, 3, 2))) * self._scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = mx.softmax(
            attn_weights.astype(mx.float32), axis=-1,
        ).astype(q.dtype)

        out = attn_weights @ v
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = mx.reshape(out, (B, -1, self._hidden_size))
        out = self.o_proj(out)

        return out, present_kv


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   FEED-FORWARD NETWORKS                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveMLP(nn.Module):
    """SwiGLU MLP (Shazeer, 2020) for dense (sensory) layers and the
    shared cognition expert.

    Architecture: output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CognitiveExpert(nn.Module):
    """Individual SwiGLU expert MLP.

    Architecturally identical to ``CognitiveMLP`` but with smaller
    ``intermediate_size``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              HIERARCHICAL COGNITIVE ROUTER                         ║
# ║                                                                    ║
# ║  Two-stage routing: module-level then expert-level.                ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveRouter(nn.Module):
    """Hierarchical two-level cognitive router.

    Stage 1: Module routing — softmax over cognitive modules.
    Stage 2: Expert routing — intra-module softmax over experts.
    Combined: P(j, m | x) = P(m | x) · P(j | m, x).
    Top-k selection over the flat combined distribution.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self._num_modules = config.num_cognitive_modules
        self._experts_per_module = config.experts_per_module
        self._total_experts = config.total_experts
        self._num_experts_per_tok = config.num_experts_per_tok
        self._norm_topk_prob = config.norm_topk_prob

        # Stage 1: module-level gate
        self.module_gate = nn.Linear(
            config.hidden_size, self._num_modules, bias=False,
        )
        # Stage 2: expert-level gate (flat, then reshaped by module)
        self.expert_gate = nn.Linear(
            config.hidden_size, self._total_experts, bias=False,
        )

    def __call__(
        self, hidden_states: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Parameters
        ----------
        hidden_states : (num_tokens, hidden_size) — flattened over batch×seq.

        Returns
        -------
        combined_probs : (num_tokens, total_experts)
        routing_weights : (num_tokens, top_k) — normalised weights.
        selected_experts : (num_tokens, top_k) — expert indices.
        """
        T = hidden_states.shape[0]

        # Stage 1: Module probabilities
        module_logits = self.module_gate(hidden_states)  # (T, M)
        module_probs = mx.softmax(
            module_logits.astype(mx.float32), axis=-1,
        )  # (T, M)

        # Stage 2: Expert probabilities (intra-module softmax)
        expert_logits = self.expert_gate(hidden_states)  # (T, E_total)
        expert_logits = mx.reshape(
            expert_logits,
            (T, self._num_modules, self._experts_per_module),
        )  # (T, M, E_per_M)
        expert_probs = mx.softmax(
            expert_logits.astype(mx.float32), axis=-1,
        )  # (T, M, E_per_M) — normalised within each module

        # Combined factored distribution
        combined_probs = mx.reshape(
            mx.expand_dims(module_probs, axis=-1) * expert_probs,
            (T, self._total_experts),
        )  # (T, E_total)

        # Top-k selection
        # MLX topk returns (values, indices) along last axis
        top_k_weights = mx.zeros((T, self._num_experts_per_tok))
        top_k_indices = mx.zeros((T, self._num_experts_per_tok), dtype=mx.int32)

        # Use argsort-based top-k (MLX does not have a dedicated topk)
        sorted_indices = mx.argsort(combined_probs, axis=-1)  # ascending
        # Take last k (highest)
        top_k_indices = sorted_indices[:, -self._num_experts_per_tok:]  # (T, k)
        # Flip to descending order
        top_k_indices = top_k_indices[:, ::-1]

        # Gather the corresponding weights
        # Use advanced indexing: combined_probs[row, col]
        row_idx = mx.broadcast_to(
            mx.arange(T)[:, None], top_k_indices.shape,
        )
        top_k_weights = combined_probs[row_idx, top_k_indices]  # (T, k)

        if self._norm_topk_prob:
            top_k_weights = top_k_weights / (
                mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-8
            )

        top_k_weights = top_k_weights.astype(hidden_states.dtype)
        return combined_probs, top_k_weights, top_k_indices


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 COGNITIVE MoE BLOCK                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveMoEBlock(nn.Module):
    """Cognitive Mixture-of-Experts feedforward block.

    16 specialised experts (4 modules × 4 experts) + shared cognition
    expert.  Expert dispatch uses loop-based token gathering + scattering.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self._total_experts = config.total_experts
        self._num_experts_per_tok = config.num_experts_per_tok

        # Specialised experts (stored as a list; MLX auto-discovers)
        self.experts = [
            CognitiveExpert(config.hidden_size, config.expert_intermediate_size)
            for _ in range(self._total_experts)
        ]

        # Shared cognition expert (always-active, larger capacity)
        self.shared_expert = CognitiveMLP(
            config.hidden_size, config.shared_expert_intermediate_size,
        )

        # Hierarchical router
        self.router = CognitiveRouter(config)

    def __call__(
        self, hidden_states: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Parameters
        ----------
        hidden_states : (B, S, D)

        Returns
        -------
        output : (B, S, D)
        router_probs : (B*S, total_experts) — for load balancing loss.
        """
        B, S, D = hidden_states.shape
        x_flat = mx.reshape(hidden_states, (-1, D))  # (T, D)

        # Route tokens to experts
        combined_probs, routing_weights, selected_experts = self.router(x_flat)

        # ── Expert dispatch ──────────────────────────────────────────
        # For each expert, compute the combined routing weight across
        # all top-k slots, then apply the expert to all tokens and mask.
        # This is a branchless, MLX-friendly dispatch pattern that avoids
        # dynamic indexing.  The per-expert FFNs are small (intermediate=384)
        # so the overhead of processing unrouted tokens is modest.
        final_hidden = mx.zeros_like(x_flat)

        for e in range(self._total_experts):
            # Aggregate routing weight for expert e across all top-k slots
            # selected_experts: (T, k), routing_weights: (T, k)
            mask_e = (selected_experts == e).astype(x_flat.dtype)  # (T, k)
            weight_e = mx.sum(
                routing_weights * mask_e, axis=-1, keepdims=True,
            )  # (T, 1) — zero for tokens not routed to this expert

            expert_out = self.experts[e](x_flat)    # (T, D)
            final_hidden = final_hidden + expert_out * weight_e

        # ── Shared cognition expert (always-active) ──────────────────
        shared_out = self.shared_expert(x_flat)
        final_hidden = final_hidden + shared_out

        return mx.reshape(final_hidden, (B, S, D)), combined_probs


# ╔══════════════════════════════════════════════════════════════════════╗
# ║            FracToM INTEGRATION (Executive Layers)                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class _SiLU(nn.Module):
    """SiLU activation as a module (for use in Sequential)."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class _Softplus(nn.Module):
    """Softplus activation as a module (for use in Sequential)."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.softplus(x)


class _Sigmoid(nn.Module):
    """Sigmoid activation as a module (for use in Sequential)."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(x)


class FracToMIntegration(nn.Module):
    """Integrates FracToM's BDI causal reasoning into a single decoder layer.

    Each executive layer at mentalizing depth *d* performs:

    1. **Observation projection** — ``hidden_states → obs ∈ ℝ^F``
    2. **Causal BDI extraction** — structural equations with learnable
       causal edge strengths.
    3. **Epistemic gating** — belief confidence modulates BDI enrichment.
    4. **Perspective shifting** (depth > 0) — gated aggregation of the
       prior executive layer's BDI.
    5. **BDI-to-hidden enrichment** — project refined BDI back to
       ``hidden_size`` via a learned output gate (initialised near zero).
    """

    def __init__(self, config: MindConfig, mentalizing_depth: int):
        super().__init__()
        D = config.hidden_size
        F = config.bdi_factor_dim
        self._mentalizing_depth = mentalizing_depth
        self._factor_dim = F

        # ── Observation → factor_dim projection ─────────────────────
        self.obs_proj = nn.Linear(D, F, bias=False)

        # ── Causal structural equations (Obs → B → D → I) ──────────
        self.belief_eq = nn.Sequential(
            nn.Linear(F, F), _SiLU(), nn.Linear(F, F),
        )
        self.desire_eq = nn.Sequential(
            nn.Linear(F, F), _SiLU(), nn.Linear(F, F),
        )
        self.intention_eq = nn.Sequential(
            nn.Linear(F, F), _SiLU(), nn.Linear(F, F),
        )

        # Learnable causal edge strengths (differentiable SCM)
        # Edges: [obs→B, B→D, obs→D, B→I, D→I]
        self.causal_strengths = mx.array([2.0, 1.5, 0.5, 1.5, 2.0])

        # ── Epistemic confidence (from belief fidelity) ─────────────
        self.epistemic_head = nn.Sequential(
            nn.Linear(F, F // 4),
            _SiLU(),
            nn.Linear(F // 4, 1),
            _Softplus(),  # σ > 0
        )

        # ── BDI → hidden enrichment ────────────────────────────────
        self.bdi_to_hidden = nn.Linear(3 * F, D, bias=False)

        # Output gate (initialised near zero → identity at init)
        self.output_gate_linear = nn.Linear(D, D)
        # Sigmoid applied in forward.  Weight init to zeros and bias
        # to -3.0 is done in MindForCausalLM._init_weights().

        # ── Perspective shifting (cross-depth, depth > 0 only) ──────
        self._has_perspective = mentalizing_depth > 0
        if self._has_perspective:
            self.prior_bdi_proj = nn.Linear(3 * F, D, bias=False)
            self.perspective_gate_net = nn.Sequential(
                nn.Linear(D + D, D // 4),
                _SiLU(),
                nn.Linear(D // 4, 1),
                _Sigmoid(),
            )

    def __call__(
        self,
        hidden_states: mx.array,
        prior_bdi: Optional[BDITensor] = None,
    ) -> Tuple[mx.array, BDITensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        hidden_states : (B, S, D) — output of MoE + residual.
        prior_bdi : BDITensor from the previous executive layer, or None.

        Returns
        -------
        enriched : (B, S, D) — hidden states enriched with BDI information.
        bdi : BDITensor — this layer's BDI state.
        info : dict — interpretability signals.
        """
        # 1. Observation projection
        obs = self.obs_proj(hidden_states)           # (B, S, F)

        # 2. Causal structural equations
        s = mx.sigmoid(self.causal_strengths)        # normalise to [0, 1]

        belief = self.belief_eq(obs * s[0])          # Obs → Belief

        desire_input = belief * s[1] + obs * s[2]    # B→D + Obs→D
        desire = self.desire_eq(desire_input)

        intention_input = belief * s[3] + desire * s[4]  # B→I + D→I
        intention = self.intention_eq(intention_input)

        bdi = BDITensor(belief, desire, intention)

        # 3. Epistemic gating
        sigma = self.epistemic_head(belief)          # (B, S, 1)
        confidence = 1.0 / (1.0 + sigma)             # ∈ (0, 1]

        # 4. BDI → hidden enrichment (gated)
        bdi_packed = bdi.pack() * confidence          # scale by confidence
        enrichment = self.bdi_to_hidden(bdi_packed)   # (B, S, D)
        gate = mx.sigmoid(self.output_gate_linear(enrichment))  # (B, S, D)
        hidden_states = hidden_states + gate * enrichment

        # 5. Perspective shifting (cross-depth)
        if self._has_perspective and prior_bdi is not None:
            prior_h = self.prior_bdi_proj(prior_bdi.pack())  # (B, S, D)
            p_gate = self.perspective_gate_net(
                mx.concatenate([hidden_states, prior_h], axis=-1),
            )  # (B, S, 1)
            hidden_states = hidden_states + p_gate * prior_h

        info = {
            "confidence": mx.stop_gradient(confidence).mean().item(),
            "causal_strengths": mx.stop_gradient(self.causal_strengths).tolist(),
            "pearl_level": PEARL_HIERARCHY_LABELS.get(
                min(self._mentalizing_depth, 2), "Counterfactual (imagining)",
            ),
        }

        return hidden_states, bdi, info


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     DECODER LAYER                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MindDecoderLayer(nn.Module):
    """A single layer of the MIND decoder.

    Architecture (pre-norm):
        1. residual + RMSNorm → GQA Attention
        2. residual + RMSNorm → FFN (Dense or Cognitive MoE)
        3. [executive only] FracToM BDI Integration
    """

    def __init__(self, config: MindConfig, layer_idx: int):
        super().__init__()
        self._layer_idx = layer_idx
        self._tier = config.layer_tier(layer_idx)
        self._hidden_size = config.hidden_size

        # ── Attention ────────────────────────────────────────────────
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CognitiveAttention(config, layer_idx)

        # ── FFN / MoE ───────────────────────────────────────────────
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
        )
        if self._tier == "sensory":
            self.mlp = CognitiveMLP(
                config.hidden_size, config.dense_intermediate_size,
            )
        else:
            self.mlp = CognitiveMoEBlock(config)

        # ── FracToM Integration (executive only) ────────────────────
        self.fractom: Optional[FracToMIntegration] = None
        if self._tier == "executive":
            depth = config.executive_depth(layer_idx)
            self.fractom = FracToMIntegration(config, mentalizing_depth=depth)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        prior_bdi: Optional[BDITensor] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Optional[mx.array], Optional[BDITensor], Dict]:
        """
        Returns
        -------
        hidden_states : (B, S, D)
        present_kv : (K, V) — updated KV cache.
        router_probs : (B*S, total_experts) or None (sensory layers).
        bdi_state : BDITensor or None (non-executive layers).
        fractom_info : dict — FracToM interpretability info.
        """
        # 1. Pre-norm + Self Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states, position_embeddings, attention_mask, past_key_value,
        )
        hidden_states = residual + hidden_states

        # 2. Pre-norm + FFN/MoE + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_probs = None
        if self._tier == "sensory":
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states, router_probs = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 3. FracToM Integration (executive only)
        bdi_state = None
        fractom_info: Dict[str, Any] = {}
        if self.fractom is not None:
            hidden_states, bdi_state, fractom_info = self.fractom(
                hidden_states, prior_bdi,
            )

        return hidden_states, present_kv, router_probs, bdi_state, fractom_info


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        MIND MODEL                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MindModel(nn.Module):
    """MIND decoder backbone: embedding → N decoder layers → final norm.

    Manages the three cognitive tiers, KV cache, and BDI state propagation
    across executive layers.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MindDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )

    def _make_causal_mask(
        self,
        seq_len: int,
        past_len: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        """Create additive causal attention mask.

        Returns (1, 1, seq_len, past_len + seq_len) where 0 = attend
        and -inf = masked (future positions).
        """
        total_len = past_len + seq_len
        mask = mx.full(
            (seq_len, total_len), float("-inf"), dtype=dtype,
        )
        # triu with diagonal=past_len+1: positions j > past_len + i are masked
        # MLX doesn't have triu directly; build from indices.
        rows = mx.arange(seq_len)[:, None]      # (S, 1)
        cols = mx.arange(total_len)[None, :]     # (1, T)
        # Attend where col <= row + past_len  (i.e., not future)
        causal = (cols <= rows + past_len)
        mask = mx.where(causal, mx.zeros_like(mask), mask)
        # (1, 1, S, T)
        return mx.reshape(mask, (1, 1, seq_len, total_len))

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> MindOutput:
        """
        Parameters
        ----------
        input_ids : (B, S) — input token IDs.
        attention_mask : (B, T) — 1 = real token, 0 = padding.
        position_ids : (B, S) — absolute position indices.
        past_key_values : list of (K, V) per layer.
        use_cache : bool — whether to return updated KV cache.

        Returns
        -------
        MindOutput with last_hidden_state, past_key_values, router_probs,
        bdi_states, cognitive_stats.
        """
        B, S = input_ids.shape

        # Determine past sequence length
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]

        # Position IDs
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(past_len, past_len + S)[None, :],
                (B, S),
            )

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        causal_mask = self._make_causal_mask(S, past_len, hidden_states.dtype)
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) where T = past_len + S, 1=attend, 0=pad
            # Use mx.where to avoid 0 * -inf = NaN
            pad_positions = (attention_mask[:, None, None, :] == 0)  # (B,1,1,T)
            causal_mask = mx.where(
                pad_positions,
                mx.array(float("-inf"), dtype=causal_mask.dtype),
                causal_mask,
            )

        # ── Layer-by-layer forward ───────────────────────────────────
        all_router_probs: List[mx.array] = []
        all_bdi_states: List[BDITensor] = []
        all_fractom_info: Dict[str, Any] = {}
        new_kv: List[Tuple[mx.array, mx.array]] = []

        # BDI state propagation across executive layers
        current_bdi: Optional[BDITensor] = None

        for i, layer in enumerate(self.layers):
            past_kv_i = None
            if past_key_values is not None and i < len(past_key_values):
                past_kv_i = past_key_values[i]

            (
                hidden_states,
                present_kv,
                router_probs,
                bdi_state,
                fractom_info,
            ) = layer(
                hidden_states,
                position_embeddings,
                causal_mask,
                past_kv_i,
                prior_bdi=current_bdi,
            )

            if use_cache:
                new_kv.append(present_kv)

            if router_probs is not None:
                all_router_probs.append(router_probs)

            if bdi_state is not None:
                all_bdi_states.append(bdi_state)
                current_bdi = bdi_state  # propagate to next executive layer

            if fractom_info:
                all_fractom_info[f"layer_{i}"] = fractom_info

        # Final normalisation
        hidden_states = self.norm(hidden_states)

        return MindOutput(
            last_hidden_state=hidden_states,
            past_key_values=new_kv if use_cache else None,
            router_probs=tuple(all_router_probs) if all_router_probs else None,
            bdi_states=all_bdi_states if all_bdi_states else None,
            cognitive_stats=all_fractom_info if all_fractom_info else None,
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   MIND FOR CAUSAL LM                               ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MindForCausalLM(nn.Module):
    """MIND model with language modelling head.

    Combines the ``MindModel`` backbone with:
    - Linear LM head (optionally tied with embedding weights).
    - Cross-entropy language modelling loss.
    - Cognitive load balancing auxiliary loss.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self._config = config
        self.model = MindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Weight tying: share the embedding weight with LM head
            self.lm_head.weight = self.model.embed_tokens.weight

        self._router_aux_loss_coef = config.router_aux_loss_coef
        self._total_experts = config.total_experts
        self._num_experts_per_tok = config.num_experts_per_tok

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following Qwen3 convention."""
        std = self._config.initializer_range
        from mlx.utils import tree_map_with_path

        def _init_param(path: str, param: mx.array) -> mx.array:
            # RMSNorm / layernorm weights → ones (1-D, named 'weight')
            if param.ndim == 1 and ("layernorm" in path or "norm" in path):
                if "weight" in path.split(".")[-1]:
                    return mx.ones_like(param)
            # Biases → zeros (except output gate, handled below)
            if param.ndim == 1 and "bias" in path.split(".")[-1]:
                return mx.zeros_like(param)
            # 2-D+ parameters (Linear / Embedding weights) → normal
            if param.ndim >= 2:
                return mx.random.normal(param.shape) * std
            # Other 1-D params (e.g. causal_strengths) → keep as-is
            return param

        new_params = tree_map_with_path(_init_param, self.parameters())
        self.update(new_params)

        # Re-apply FracToM output gate initialisation
        for layer in self.model.layers:
            if layer.fractom is not None:
                layer.fractom.output_gate_linear.weight = mx.zeros_like(
                    layer.fractom.output_gate_linear.weight,
                )
                layer.fractom.output_gate_linear.bias = mx.full(
                    layer.fractom.output_gate_linear.bias.shape,
                    -3.0,
                )

        # Re-tie weights after init
        if self._config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        labels: Optional[mx.array] = None,
        use_cache: bool = False,
    ) -> MindCausalLMOutput:
        """
        Parameters
        ----------
        input_ids : (B, S)
        attention_mask : (B, T) — 1 = attend, 0 = pad.
        position_ids : (B, S)
        past_key_values : list of (K, V) per layer.
        labels : (B, S) — for computing LM loss (-100 = ignore).
        use_cache : bool

        Returns
        -------
        MindCausalLMOutput with loss, aux_loss, logits, etc.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # ── Language modelling loss ──────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]        # (B, S-1, V)
            shift_labels = labels[:, 1:]             # (B, S-1)
            # Cross-entropy: flatten and compute
            V = self._config.vocab_size
            flat_logits = mx.reshape(shift_logits, (-1, V))
            flat_labels = mx.reshape(shift_labels, (-1,))
            # Mask out ignore_index=-100
            valid = (flat_labels != -100)
            # MLX cross_entropy
            log_probs = mx.softmax(flat_logits.astype(mx.float32), axis=-1)
            log_probs = mx.log(log_probs + 1e-12)
            # Gather log-probs for true labels (clamp labels to valid range)
            safe_labels = mx.clip(flat_labels, a_min=0, a_max=V - 1)
            # Advanced gather: log_probs[i, safe_labels[i]]
            row_idx = mx.arange(flat_logits.shape[0])
            nll = -log_probs[row_idx, safe_labels]  # (B*(S-1),)
            # Apply valid mask
            nll = nll * valid.astype(nll.dtype)
            num_valid = mx.sum(valid).astype(mx.float32)
            loss = mx.sum(nll) / mx.maximum(num_valid, mx.array(1.0))

        # ── Cognitive load balancing loss ────────────────────────────
        aux_loss = None
        if outputs.router_probs is not None:
            aux_loss = cognitive_load_balancing_loss(
                outputs.router_probs,
                self._total_experts,
                self._num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss = loss + self._router_aux_loss_coef * aux_loss

        return MindCausalLMOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            router_probs=outputs.router_probs,
            bdi_states=outputs.bdi_states,
            cognitive_stats=outputs.cognitive_stats,
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    LOSS FUNCTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


def cognitive_load_balancing_loss(
    router_probs: Tuple[mx.array, ...],
    num_experts: int,
    top_k: int,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """Load balancing auxiliary loss for cognitive MoE routing.

    Switch Transformer (Fedus et al., 2021) style:
        L = num_experts × Σ_i (f_i × p_i)
    """
    if not router_probs:
        return mx.array(0.0)

    # Concatenate across layers: (total_tokens_across_layers, num_experts)
    concatenated = mx.concatenate(list(router_probs), axis=0)

    # Top-k selection for "tokens assigned to each expert"
    sorted_idx = mx.argsort(concatenated, axis=-1)
    selected = sorted_idx[:, -top_k:]  # (N, k)

    expert_mask = _one_hot(selected, num_experts)  # (N, k, E)

    if attention_mask is None:
        # f_i: mean fraction of assignments per expert
        tokens_per_expert = mx.mean(expert_mask, axis=(0, 1))
        # p_i: mean routing probability per expert
        router_prob_per_expert = mx.mean(concatenated, axis=0)
    else:
        # Expand attention mask for weighted computation
        B = attention_mask.shape[0]
        tokens_per_layer = concatenated.shape[0] // len(router_probs)
        S_new = tokens_per_layer // B
        current_mask = attention_mask[:, -S_new:] if attention_mask.shape[1] > S_new else attention_mask
        num_layers = len(router_probs)
        per_token_mask = mx.reshape(current_mask, (-1,))          # (B*S_new,)
        per_token_mask = mx.tile(per_token_mask, (num_layers,))   # (N,)
        per_token_mask = per_token_mask.astype(mx.float32)

        mask_sum = mx.maximum(mx.sum(per_token_mask), mx.array(1.0))
        tokens_per_expert = (
            mx.sum(
                expert_mask * per_token_mask[:, None, None],
                axis=(0, 1),
            ) / (mask_sum * top_k)
        )
        router_prob_per_expert = (
            mx.sum(
                concatenated * per_token_mask[:, None],
                axis=0,
            ) / mask_sum
        )

    return mx.sum(tokens_per_expert * router_prob_per_expert) * num_experts


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                ANALYSIS & INTERPRETABILITY UTILITIES                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    from mlx.utils import tree_flatten
    params = tree_flatten(model.trainable_parameters() if trainable_only else model.parameters())
    return sum(p.size for _, p in params)


def count_active_parameters(config: MindConfig) -> int:
    """Estimate active parameters per token.

    For MoE layers, only ``num_experts_per_tok`` experts + shared expert
    are active per token.
    """
    D = config.hidden_size
    H = config.num_attention_heads
    KV = config.num_key_value_heads
    hd = config.head_dim

    # Attention (same for all layers)
    attn_params = (H * hd + 2 * KV * hd + H * hd) * D + 2 * hd
    norms = 2 * D

    # Sensory dense FFN
    dense_ffn = 3 * D * config.dense_intermediate_size

    # Active MoE FFN (4 active experts + shared)
    active_expert_ffn = (
        config.num_experts_per_tok * 3 * D * config.expert_intermediate_size
    )
    shared_ffn = 3 * D * config.shared_expert_intermediate_size
    moe_ffn = active_expert_ffn + shared_ffn

    # FracToM per executive layer (rough estimate)
    F_ = config.bdi_factor_dim
    fractom = D * F_ + 3 * 2 * F_ * F_ + 3 * F_ * D

    sensory = config.num_sensory_layers * (attn_params + norms + dense_ffn)
    associative = config.num_associative_layers * (attn_params + norms + moe_ffn)
    executive = config.num_executive_layers * (attn_params + norms + moe_ffn + fractom)

    embedding = config.vocab_size * D  # LM head is tied

    return int(sensory + associative + executive + embedding)


def analyse_cognitive_architecture(output: MindCausalLMOutput) -> str:
    """Generate a human-readable analysis of the model's cognitive activity."""
    lines = []
    lines.append("=" * 70)
    lines.append("MIND (MLX) — Cognitive Architecture Analysis")
    lines.append("=" * 70)

    # Router statistics
    if output.router_probs:
        lines.append("\n─── Cognitive Module Routing ───")
        all_probs = mx.concatenate(list(output.router_probs), axis=0)
        total_experts = all_probs.shape[-1]
        num_modules = len(COGNITIVE_MODULE_NAMES)
        experts_per_module = total_experts // num_modules
        module_usage = mx.mean(
            mx.reshape(all_probs, (-1, num_modules, experts_per_module)).sum(-1),
            axis=0,
        )
        for m in range(num_modules):
            name = COGNITIVE_MODULE_NAMES[m] if m < len(COGNITIVE_MODULE_NAMES) else f"Module {m}"
            val = module_usage[m].item()
            bar = "█" * int(val * 80) if not math.isnan(val) else "(NaN)"
            lines.append(f"  {name:14s}: {val:.3f}  {bar}")

        expert_usage = mx.mean(all_probs, axis=0)
        lines.append("\n  Per-expert utilisation:")
        for m in range(num_modules):
            name = COGNITIVE_MODULE_NAMES[m] if m < len(COGNITIVE_MODULE_NAMES) else f"Module {m}"
            start = m * experts_per_module
            vals = "  ".join(
                f"e{j}={expert_usage[start + j].item():.3f}"
                for j in range(experts_per_module)
            )
            lines.append(f"    {name:14s}: {vals}")

    # BDI states
    if output.bdi_states:
        lines.append("\n─── FracToM BDI States (Executive Layers) ───")
        for i, bdi in enumerate(output.bdi_states):
            b_norm = mx.sqrt(mx.sum(
                mx.stop_gradient(bdi.belief) ** 2, axis=-1,
            )).mean().item()
            d_norm = mx.sqrt(mx.sum(
                mx.stop_gradient(bdi.desire) ** 2, axis=-1,
            )).mean().item()
            i_norm = mx.sqrt(mx.sum(
                mx.stop_gradient(bdi.intention) ** 2, axis=-1,
            )).mean().item()
            pearl = PEARL_HIERARCHY_LABELS.get(min(i, 2), "Counterfactual")
            lines.append(
                f"  Depth {i} ({pearl:25s}): "
                f"‖B‖={b_norm:.3f}  ‖D‖={d_norm:.3f}  ‖I‖={i_norm:.3f}"
            )

    # FracToM info
    if output.cognitive_stats:
        lines.append("\n─── FracToM Causal Integration ───")
        for layer_key, info in sorted(output.cognitive_stats.items()):
            conf = info.get("confidence", "N/A")
            pearl = info.get("pearl_level", "N/A")
            strengths = info.get("causal_strengths", [])
            edge_names = ["Obs→B", "B→D", "Obs→D", "B→I", "D→I"]
            s_str = "  ".join(
                f"{edge_names[i]}={strengths[i]:.2f}"
                for i in range(min(len(strengths), len(edge_names)))
            ) if strengths else "N/A"
            lines.append(
                f"  {layer_key:10s} | {pearl:30s} | conf={conf:.3f} | {s_str}"
            )

    # Auxiliary loss
    if output.aux_loss is not None:
        val = output.aux_loss.item()
        lines.append(f"\n  Load balancing loss: {val:.6f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def get_tier_summary(config: MindConfig) -> str:
    """Print a summary of the cognitive tier architecture."""
    lines = []
    lines.append("MIND Cognitive Tier Architecture")
    lines.append("─" * 50)
    for i in range(config.num_hidden_layers):
        tier = config.layer_tier(i)
        label = COGNITIVE_TIER_LABELS[tier]
        extras = ""
        if tier == "executive":
            depth = config.executive_depth(i)
            pearl = PEARL_HIERARCHY_LABELS.get(min(depth, 2), "Counterfactual")
            extras = f" | FracToM depth {depth} ({pearl})"
        lines.append(f"  Layer {i:2d}: {tier:12s} — {label}{extras}")
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   DEMO / SMOKE TEST                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def demo_forward_backward() -> None:
    """Smoke test: build model, forward pass, backward pass, print stats."""
    print("=" * 70)
    print("MIND (MLX) — ~1B Cognitive MoE Language Model: Smoke Test")
    print("=" * 70)

    config = MindConfig()
    model = MindForCausalLM(config)

    total = count_parameters(model)
    active = count_active_parameters(config)
    print(f"\nConfiguration:")
    print(f"  Hidden size:           {config.hidden_size}")
    print(f"  Layers:                {config.num_hidden_layers}")
    print(f"  Attention heads:       {config.num_attention_heads} (KV: {config.num_key_value_heads})")
    print(f"  Cognitive modules:     {config.num_cognitive_modules}")
    print(f"  Experts per module:    {config.experts_per_module}")
    print(f"  Total experts:         {config.total_experts}")
    print(f"  Active experts/token:  {config.num_experts_per_tok}")
    print(f"  Vocab size:            {config.vocab_size}")
    print()

    print(f"Parameters:")
    print(f"  Total:              {total:>12,}")
    print(f"  Active per token:   {active:>12,}  ({100*active/total:.1f}%)")
    print()

    # Tier summary
    print(get_tier_summary(config))
    print()

    print("Device: Apple Silicon (Metal via MLX)")

    # Forward pass
    B, S = 2, 64
    input_ids = mx.random.randint(0, config.vocab_size, shape=(B, S))
    labels = mx.random.randint(0, config.vocab_size, shape=(B, S))
    attention_mask = mx.ones((B, S), dtype=mx.int32)

    print(f"\nForward pass: batch_size={B}, seq_len={S}")

    # Use value_and_grad for backward pass test
    def loss_fn(model, ids, mask, labs):
        out = model(ids, attention_mask=mask, labels=labs)
        return out.loss, out

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    (loss_val, output), grads = loss_and_grad(model, input_ids, attention_mask, labels)
    mx.eval(loss_val, output.logits)

    print(f"  Logits shape:    {output.logits.shape}")
    print(f"  Loss:            {loss_val.item():.4f}")
    if output.aux_loss is not None:
        mx.eval(output.aux_loss)
        print(f"  Aux loss:        {output.aux_loss.item():.6f}")
    if output.bdi_states:
        print(f"  BDI states:      {len(output.bdi_states)} (executive layers)")

    # Gradient norm
    print("\nBackward pass...")
    from mlx.utils import tree_flatten
    mx.eval(grads)
    grad_norm_sq = sum(
        mx.sum(g * g).item() for _, g in tree_flatten(grads)
    )
    grad_norm = math.sqrt(grad_norm_sq)
    print(f"  Gradient norm:   {grad_norm:.4f}")
    print(f"  Status:          OK ✓")

    # Cognitive analysis (eval mode — no grad needed)
    output_eval = model(input_ids, attention_mask=attention_mask)
    mx.eval(output_eval.logits)
    print()
    print(analyse_cognitive_architecture(output_eval))

    # KV cache test
    print("\n─── KV Cache Test (Autoregressive Generation) ───")
    # Prefill
    prefill_ids = input_ids[:, :32]
    prefill_mask = attention_mask[:, :32]
    prefill_out = model.model(
        prefill_ids, attention_mask=prefill_mask, use_cache=True,
    )
    past_kv = prefill_out.past_key_values
    mx.eval(prefill_out.last_hidden_state)
    print(f"  Prefill: {prefill_ids.shape[1]} tokens → "
          f"KV cache: {len(past_kv)} layers × K shape {past_kv[0][0].shape}")

    # Decode one token
    next_id = input_ids[:, 32:33]
    decode_mask = mx.ones((B, 33), dtype=mx.int32)
    decode_out = model.model(
        next_id,
        attention_mask=decode_mask,
        past_key_values=past_kv,
        use_cache=True,
    )
    mx.eval(decode_out.last_hidden_state)
    decode_logits = model.lm_head(decode_out.last_hidden_state)
    mx.eval(decode_logits)
    print(f"  Decode:  1 token → logits {decode_logits.shape}")
    print(f"  Updated KV cache: K shape {decode_out.past_key_values[0][0].shape}")
    print(f"  Status:  OK ✓")

    print("\n" + "=" * 70)
    print(f"MIND (MLX) smoke test passed | {total:,} parameters (~{total/1e9:.2f}B)")
    print("=" * 70)


if __name__ == "__main__":
    demo_forward_backward()
