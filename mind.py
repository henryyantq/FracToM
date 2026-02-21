"""
MIND — Mixture of Intelligent Neural Dynamics
================================================

A ~1B-parameter Mixture-of-Experts causal language model that organises
computation into parallel and hierarchical cognitive processing streams,
using FracToM's causal Theory-of-Mind decoder as the integrative backbone.

Architecture draws structural inspiration from Qwen3-MoE (Yang et al., 2025)
— RMSNorm, RoPE, GQA with QK-norm, SwiGLU, top-k softmax routing — while
making deep modifications grounded in cognitive science.


Cognitive Design Philosophy
----------------------------
The human brain processes information through parallel, specialised cortical
modules that are hierarchically organised (Mesulam, 1998):

  1. **Sensory cortex** — fast, feedforward feature extraction.
  2. **Association cortex** — multi-modal integration, pattern matching.
  3. **Prefrontal cortex** — executive control, planning, Theory-of-Mind.

MIND mirrors this hierarchy in three tiers of decoder layers:

  ┌────────────────────────────────────────────────────────────────┐
  │  Tier 3: EXECUTIVE (layers 18–23)                             │
  │  Cognitive MoE + FracToM BDI/SCM causal reasoning             │
  │  ≈ Prefrontal cortex: ToM, metacognition, planning            │
  ├────────────────────────────────────────────────────────────────┤
  │  Tier 2: ASSOCIATIVE (layers 6–17)                            │
  │  Cognitive MoE: parallel expert modules for different          │
  │  cognitive functions (analytical, linguistic, associative,     │
  │  social)                                                       │
  │  ≈ Temporal / parietal association cortex                      │
  ├────────────────────────────────────────────────────────────────┤
  │  Tier 1: SENSORY (layers 0–5)                                 │
  │  Dense SwiGLU FFN: basic token processing, no MoE overhead    │
  │  ≈ Primary sensory cortex: rapid feature extraction            │
  └────────────────────────────────────────────────────────────────┘


Mixture-of-Experts Design (Modified from Qwen3-MoE)
-----------------------------------------------------
Standard MoE (Qwen3) uses a flat pool of undifferentiated experts with
a single-level top-k router.  MIND introduces:

  1. **Cognitive Module Grouping** — Experts are organised into functionally
     specialised *cognitive modules* (4 modules × 4 experts = 16 experts).

       Module 0 "Analytical":  logic, maths, structured reasoning
       Module 1 "Linguistic":  grammar, semantics, pragmatics
       Module 2 "Associative": pattern matching, analogy, memory retrieval
       Module 3 "Social":      Theory-of-Mind, empathy, social inference

  2. **Hierarchical Routing** — A two-stage router first selects cognitive
     modules, then selects experts within each module:

       score(expert j in module m | token x)
         = P(module m | x) × P(expert j | module m, x)

     This factored distribution induces functional specialisation: experts
     within a module share cognitive-domain relevance while differing in
     fine-grained specialisation.

  3. **Shared Cognition Expert** — An always-active dense expert that
     processes every token, regardless of routing.  Analogous to general
     intelligence / working memory (Cattell, 1963): a domain-general
     cognitive resource that supports all specialised processing.

  4. **Cognitive Load Balancing** — Auxiliary loss encourages balanced
     utilisation at both module and expert levels, preventing cognitive-
     module collapse.


FracToM Integration (Executive Layers)
----------------------------------------
The top 6 decoder layers incorporate FracToM's Theory-of-Mind mechanisms:

  - **BDI State Extraction**: Each executive layer projects hidden states
    into Belief-Desire-Intention triples via causal structural equations
    (Obs→B→D→I), following Bratman's (1987) BDI framework.

  - **Causal Hierarchy**: Executive layers are mapped to Pearl's (2009)
    three levels of causal reasoning:
    * Layer 18 (depth 0): Association — observe and predict
    * Layer 19 (depth 1): Intervention — "what if I change my belief?"
    * Layers 20–23 (depths 2–5): Counterfactual — "what would they
      believe if they'd seen X?"

  - **Epistemic Gating**: A learned confidence gate modulates BDI
    enrichment, suppressing hallucinated mental-state attributions from
    uncertain or noisy inputs.

  - **Cross-Depth Perspective Shifting**: Higher executive layers attend
    to prior executive layers' BDI states through a gated mechanism,
    modelling the cognitive operation of adopting another's viewpoint.


Parameter Budget (~1.0B)
-------------------------
  Embedding:                 49.2M
  24× Attention (GQA):     141.6M
  6× Sensory Dense FFN:    113.2M
  18× Cognitive MoE FFN:   680.0M
  6× FracToM Integration:   21.2M
  Final Norm + LM Head:      1.5M  (tied)
  ─────────────────────────────────
  Total:                  ~1.006B

  Active params / token:  ~576M  (57% active ratio)


Usage
-----
    from mind import MindForCausalLM, MindConfig

    config = MindConfig()          # default ~1B config
    model = MindForCausalLM(config)
    input_ids = torch.randint(0, 32000, (2, 128))
    output = model(input_ids)
    logits = output.logits         # (2, 128, 32000)


Architecture Diagram
--------------------
    Input Token IDs
          │
          ▼
    ┌─────────────┐
    │  Embedding   │  + RoPE positional encoding
    └──────┬──────┘
           │
    ═══════╪═══════════════ Tier 1: SENSORY (layers 0–5) ═══════
           │
     ┌─────┴─────┐
     │  RMSNorm   │
     │  GQA Attn  │─── QK-norm, RoPE, GQA (16 heads, 4 KV heads)
     │  + Residual│
     ├───────────┤
     │  RMSNorm   │
     │  SwiGLU    │─── Dense FFN (1536 → 4096 → 1536)
     │  + Residual│
     └─────┬─────┘  × 6 layers
           │
    ═══════╪═══════════════ Tier 2: ASSOCIATIVE (layers 6–17) ══
           │
     ┌─────┴─────┐
     │  RMSNorm   │
     │  GQA Attn  │─── same attention as Tier 1
     │  + Residual│
     ├───────────┤
     │  RMSNorm   │
     │  Cognitive │      ┌─── Module Router (4 modules)
     │    MoE     │──────┤    Expert Router (4 experts/module)
     │  + Shared  │      └─── Shared Cognition Expert
     │  + Residual│
     └─────┬─────┘  × 12 layers
           │
    ═══════╪═══════════════ Tier 3: EXECUTIVE (layers 18–23) ═══
           │
     ┌─────┴─────┐
     │  RMSNorm   │
     │  GQA Attn  │─── same attention
     │  + Residual│
     ├───────────┤
     │  RMSNorm   │
     │  Cognitive │─── same MoE as Tier 2
     │    MoE     │
     │  + Residual│
     ├───────────┤
     │  FracToM   │      ┌─── BDI Extraction (Obs→B→D→I)
     │ Integration│──────┤    Epistemic Gating (confidence)
     │            │      └─── Perspective Shift (cross-depth)
     └─────┬─────┘  × 6 layers
           │
     ┌─────┴─────┐
     │  RMSNorm   │  Final layer normalisation
     └──────┬─────┘
            │
     ┌──────┴──────┐
     │   LM Head   │  Linear projection → vocab logits
     └─────────────┘


References
----------
  - Qwen3-MoE: Yang et al. (2025) — base MoE architecture
  - FracToM (this work) — fractal Theory-of-Mind backbone
  - BDI: Bratman (1987) — Belief-Desire-Intention framework
  - SCM: Pearl (2009) — Structural Causal Models
  - Mesulam (1998) — cortical processing hierarchy
  - Cattell (1963) — fluid vs. crystallised intelligence
  - Switch Transformer: Fedus et al. (2021) — MoE load balancing

Requirements
------------
  Python ≥ 3.9,  PyTorch ≥ 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

    All three tensors share shape ``(batch, seq_len, factor_dim)`` (or
    ``(batch, factor_dim)`` for pooled representations).
    """

    __slots__ = ("belief", "desire", "intention")

    def __init__(self, belief: Tensor, desire: Tensor, intention: Tensor):
        self.belief = belief
        self.desire = desire
        self.intention = intention

    def pack(self) -> Tensor:
        """Pack → ``(…, 3 × factor_dim)``."""
        return torch.cat([self.belief, self.desire, self.intention], dim=-1)

    @staticmethod
    def unpack(x: Tensor, factor_dim: int) -> "BDITensor":
        b, d, i = x.split(factor_dim, dim=-1)
        return BDITensor(b, d, i)

    def detach(self) -> "BDITensor":
        return BDITensor(
            self.belief.detach(),
            self.desire.detach(),
            self.intention.detach(),
        )


@dataclass
class MindOutput:
    """Output from the ``MindModel`` backbone (no LM head)."""

    last_hidden_state: Tensor
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
    router_probs: Optional[Tuple[Tensor, ...]] = None
    bdi_states: Optional[List[BDITensor]] = None
    cognitive_stats: Optional[Dict[str, Any]] = None


@dataclass
class MindCausalLMOutput:
    """Output from ``MindForCausalLM``."""

    loss: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
    router_probs: Optional[Tuple[Tensor, ...]] = None
    bdi_states: Optional[List[BDITensor]] = None
    cognitive_stats: Optional[Dict[str, Any]] = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     CORE BUILDING BLOCKS                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Used throughout in place of LayerNorm, following Qwen3/LLaMA convention.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


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
            theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(
        self, x: Tensor, position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, S, D) — used only for device / dtype.
        position_ids : (B, S) — absolute position indices.

        Returns
        -------
        cos, sin : each (B, S, head_dim).
        """
        # (1, head_dim//2, 1)
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1,
        ).to(x.device)
        pos = position_ids[:, None, :].float()          # (B, 1, S)
        freqs = (inv_freq @ pos).transpose(1, 2)        # (B, S, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (B, S, head_dim)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)  # (B, 1, S, D) for head dim
    sin = sin.unsqueeze(unsqueeze_dim)
    q_emb = q * cos + _rotate_half(q) * sin
    k_emb = k * cos + _rotate_half(k) * sin
    return q_emb, k_emb


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads for grouped-query attention."""
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  GROUPED-QUERY ATTENTION                           ║
# ║                                                                    ║
# ║  Follows Qwen3-MoE: GQA with QK-norm (RMSNorm on head_dim),      ║
# ║  RoPE, no attention bias.                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveAttention(nn.Module):
    """Grouped-Query Attention with QK-norm and RoPE.

    Structurally identical to Qwen3MoeAttention: the same attention is
    used across all three cognitive tiers (sensory, associative,
    executive), mirroring the neuroscience finding that thalamo-cortical
    attention gating is shared infrastructure across cortical hierarchies.

    Modifications from standard Qwen3:
    - No sliding-window support (simplicity).
    - Explicit KV-cache tuple interface.
    """

    def __init__(self, config: MindConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False,
        )

        # QK-norm on head_dim (Qwen3 innovation — stabilises training)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
        present_kv : (K, V) — updated KV cache tensors.
        """
        B, S, _ = hidden_states.shape

        # Project & reshape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        # QK-norm (applied before transpose, on head_dim axis)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache concatenation
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present_kv = (k, v)

        # GQA: expand KV heads to match query heads
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32,
        ).to(q.dtype)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, -1, self.hidden_size)
        out = self.o_proj(out)

        return out, present_kv


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   FEED-FORWARD NETWORKS                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveMLP(nn.Module):
    """SwiGLU MLP (Shazeer, 2020) for dense (sensory) layers and the
    shared cognition expert.

    Follows the Qwen3MoeMLP architecture:
        output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class CognitiveExpert(nn.Module):
    """Individual SwiGLU expert MLP.

    Architecturally identical to ``CognitiveMLP`` but with smaller
    ``intermediate_size``, representing a specialised cognitive sub-process
    within a cognitive module.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              HIERARCHICAL COGNITIVE ROUTER                         ║
# ║                                                                    ║
# ║  Two-stage routing: module-level then expert-level.                ║
# ║                                                                    ║
# ║  This is a key architectural departure from Qwen3-MoE, which uses ║
# ║  flat top-k routing.  The factored distribution:                   ║
# ║                                                                    ║
# ║    P(expert j ∈ module m | x) = P(m | x) · P(j | m, x)           ║
# ║                                                                    ║
# ║  induces cognitive-module specialisation through the hierarchical  ║
# ║  structure, while retaining differentiability.                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveRouter(nn.Module):
    """Hierarchical two-level cognitive router.

    Stage 1: *Module routing* — scores each cognitive module (analytical,
    linguistic, associative, social) via a learned linear gate.

    Stage 2: *Expert routing* — within all modules, scores individual
    experts via a second linear gate that produces logits grouped by
    module (intra-module softmax normalisation).

    The combined routing probability for expert *j* in module *m* is:

        P(j, m | x) = softmax_m(module_logits)[m]
                     × softmax_j(expert_logits[m])[j]

    Top-k selection is then performed over the flat combined distribution.

    Parameters
    ----------
    config : MindConfig
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self.num_modules = config.num_cognitive_modules
        self.experts_per_module = config.experts_per_module
        self.total_experts = config.total_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        # Stage 1: module-level gate
        self.module_gate = nn.Linear(
            config.hidden_size, self.num_modules, bias=False,
        )
        # Stage 2: expert-level gate (flat, then reshaped by module)
        self.expert_gate = nn.Linear(
            config.hidden_size, self.total_experts, bias=False,
        )

    def forward(
        self, hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        hidden_states : (num_tokens, hidden_size) — flattened over batch×seq.

        Returns
        -------
        combined_probs : (num_tokens, total_experts) — full routing distribution
            (for load balancing loss computation).
        routing_weights : (num_tokens, top_k) — normalised weights for selected
            experts.
        selected_experts : (num_tokens, top_k) — indices of selected experts.
        """
        T, D = hidden_states.shape

        # Stage 1: Module probabilities
        module_logits = self.module_gate(hidden_states)   # (T, M)
        module_probs = F.softmax(
            module_logits, dim=-1, dtype=torch.float32,
        )  # (T, M)

        # Stage 2: Expert probabilities (intra-module softmax)
        expert_logits = self.expert_gate(hidden_states)   # (T, E_total)
        expert_logits = expert_logits.view(
            T, self.num_modules, self.experts_per_module,
        )  # (T, M, E_per_M)
        expert_probs = F.softmax(
            expert_logits, dim=-1, dtype=torch.float32,
        )  # (T, M, E_per_M) — normalised within each module

        # Combined factored distribution
        # P(expert j in module m | x) = P(m | x) · P(j | m, x)
        combined_probs = (
            module_probs.unsqueeze(-1) * expert_probs
        ).view(T, self.total_experts)  # (T, E_total)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            combined_probs, self.num_experts_per_tok, dim=-1,
        )  # each (T, k)

        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
            )

        top_k_weights = top_k_weights.to(hidden_states.dtype)
        return combined_probs, top_k_weights, top_k_indices


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 COGNITIVE MoE BLOCK                                ║
# ║                                                                    ║
# ║  Complete MoE feedforward layer with:                              ║
# ║  (1) Hierarchical cognitive routing                                ║
# ║  (2) 16 specialised experts (4 modules × 4 experts)               ║
# ║  (3) Shared cognition expert (always-active)                       ║
# ║                                                                    ║
# ║  Structurally replaces Qwen3MoeSparseMoeBlock.                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class CognitiveMoEBlock(nn.Module):
    """Cognitive Mixture-of-Experts feedforward block.

    Replaces the dense SwiGLU FFN in associative and executive layers with
    a routed ensemble of specialised experts plus an always-active shared
    expert ("general cognition" pathway).

    Expert Dispatch
    ---------------
    Follows the Qwen3 dispatch pattern: for each expert that has at least
    one token routed to it, gather the relevant tokens, apply the expert,
    and scatter the weighted outputs back.

    Shared Expert
    -------------
    The shared cognition expert runs on **every** token and its output is
    added to the routed expert output (DeepSeek-V2 style). Cognitively,
    this represents domain-general working memory / fluid intelligence that
    supports all specialised processing.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self.total_experts = config.total_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Specialised experts (grouped by cognitive module, but stored flat)
        self.experts = nn.ModuleList([
            CognitiveExpert(config.hidden_size, config.expert_intermediate_size)
            for _ in range(self.total_experts)
        ])

        # Shared cognition expert (always-active, larger capacity)
        self.shared_expert = CognitiveMLP(
            config.hidden_size, config.shared_expert_intermediate_size,
        )

        # Hierarchical router
        self.router = CognitiveRouter(config)

    def forward(
        self, hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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
        x_flat = hidden_states.view(-1, D)   # (T, D) where T = B * S

        # Route tokens to experts
        combined_probs, routing_weights, selected_experts = self.router(x_flat)

        # ── Expert dispatch ──────────────────────────────────────────
        final_hidden = torch.zeros_like(x_flat)

        # Build expert assignment mask:  (num_experts, top_k, num_tokens)
        with torch.no_grad():
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.total_experts,
            ).float()                                       # (T, k, E)
            expert_mask = expert_mask.permute(2, 1, 0)      # (E, k, T)
            # Find which experts have ≥1 token assigned
            expert_hit = (
                expert_mask.sum(dim=(-1, -2)) > 0
            ).nonzero(as_tuple=False)

        for idx in expert_hit:
            e = idx[0].item()
            # Positions where this expert is selected
            top_k_pos, token_idx = torch.where(expert_mask[e])
            if token_idx.numel() == 0:
                continue
            current_state = x_flat[token_idx]
            expert_out = self.experts[e](current_state)
            expert_out = expert_out * routing_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, expert_out.to(final_hidden.dtype))

        # ── Shared cognition expert (always-active) ──────────────────
        shared_out = self.shared_expert(x_flat)
        final_hidden = final_hidden + shared_out

        return final_hidden.view(B, S, D), combined_probs


# ╔══════════════════════════════════════════════════════════════════════╗
# ║            FracToM INTEGRATION (Executive Layers)                  ║
# ║                                                                    ║
# ║  Lightweight integration of FracToM's BDI causal reasoning into    ║
# ║  the decoder.  Each executive layer:                               ║
# ║  1. Extracts BDI from hidden states via causal structural eqs.     ║
# ║  2. Applies epistemic gating (confidence-based scaling).           ║
# ║  3. Aggregates prior-layer BDI via perspective shift.              ║
# ║  4. Enriches hidden states with causally-refined BDI info.         ║
# ║                                                                    ║
# ║  The sequential stack of executive layers forms a mentalizing      ║
# ║  hierarchy: deeper layers = higher-order ToM.                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FracToMIntegration(nn.Module):
    """Integrates FracToM's BDI causal reasoning into a single decoder layer.

    Each executive layer at mentalizing depth *d* performs:

    1. **Observation projection** — ``hidden_states → obs ∈ ℝ^F``
    2. **Causal BDI extraction** — structural equations:
       ``B = f_B(obs)``, ``D = f_D(B + obs)``, ``I = f_I(B + D + obs)``
       with learnable causal edge strengths.
    3. **Epistemic gating** — ``confidence = 1 / (1 + σ_B)``, scaling
       the BDI enrichment by belief confidence.
    4. **Perspective shifting** (depth > 0) — gated aggregation of the
       prior executive layer's BDI, modelling the cognitive operation
       of "putting yourself in someone else's shoes".
    5. **BDI-to-hidden enrichment** — project refined BDI back to
       ``hidden_size`` and add via a learned output gate (initialised
       near zero for stable training startup).

    The stack of executive layers together implements Pearl's causal
    hierarchy: depth 0 → Association, depth 1 → Intervention,
    depth 2+ → Counterfactual.

    Parameters
    ----------
    config : MindConfig
    mentalizing_depth : int
        0-indexed depth within the executive tier.
    """

    def __init__(self, config: MindConfig, mentalizing_depth: int):
        super().__init__()
        D = config.hidden_size
        F = config.bdi_factor_dim
        self.mentalizing_depth = mentalizing_depth
        self.factor_dim = F

        # ── Observation → factor_dim projection ─────────────────────
        self.obs_proj = nn.Linear(D, F, bias=False)

        # ── Causal structural equations (Obs → B → D → I) ──────────
        # Each is a 2-layer MLP: Linear → SiLU → Linear
        self.belief_eq = nn.Sequential(
            nn.Linear(F, F), nn.SiLU(), nn.Linear(F, F),
        )
        self.desire_eq = nn.Sequential(
            nn.Linear(F, F), nn.SiLU(), nn.Linear(F, F),
        )
        self.intention_eq = nn.Sequential(
            nn.Linear(F, F), nn.SiLU(), nn.Linear(F, F),
        )

        # Learnable causal edge strengths (differentiable SCM)
        # Edges: [obs→B, B→D, obs→D, B→I, D→I]
        self.causal_strengths = nn.Parameter(
            torch.tensor([2.0, 1.5, 0.5, 1.5, 2.0]),
        )

        # ── Epistemic confidence (from belief fidelity) ─────────────
        self.epistemic_head = nn.Sequential(
            nn.Linear(F, F // 4),
            nn.SiLU(),
            nn.Linear(F // 4, 1),
            nn.Softplus(),  # σ > 0
        )

        # ── BDI → hidden enrichment ────────────────────────────────
        self.bdi_to_hidden = nn.Linear(3 * F, D, bias=False)

        # Output gate (initialised near zero → identity at init)
        self.output_gate = nn.Sequential(
            nn.Linear(D, D),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.output_gate[0].weight)
        nn.init.constant_(self.output_gate[0].bias, -3.0)  # sigmoid(-3) ≈ 0.05

        # ── Perspective shifting (cross-depth, depth > 0 only) ──────
        self.has_perspective = mentalizing_depth > 0
        if self.has_perspective:
            # Project prior BDI (3F) → hidden_size for gated addition
            self.prior_bdi_proj = nn.Linear(3 * F, D, bias=False)
            self.perspective_gate_net = nn.Sequential(
                nn.Linear(D + D, D // 4),
                nn.SiLU(),
                nn.Linear(D // 4, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        hidden_states: Tensor,
        prior_bdi: Optional[BDITensor] = None,
    ) -> Tuple[Tensor, BDITensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        hidden_states : (B, S, D) — output of MoE + residual.
        prior_bdi : BDITensor from the previous executive layer, or None.

        Returns
        -------
        enriched : (B, S, D) — hidden states enriched with BDI information.
        bdi : BDITensor — this layer's BDI state.
        info : dict — interpretability signals (confidence, causal strengths).
        """
        F_dim = self.factor_dim

        # 1. Observation projection
        obs = self.obs_proj(hidden_states)           # (B, S, F)

        # 2. Causal structural equations
        s = torch.sigmoid(self.causal_strengths)     # normalise to [0, 1]

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
        gate = self.output_gate(enrichment)           # (B, S, D)
        hidden_states = hidden_states + gate * enrichment

        # 5. Perspective shifting (cross-depth)
        if self.has_perspective and prior_bdi is not None:
            prior_h = self.prior_bdi_proj(prior_bdi.pack())  # (B, S, D)
            # Compute scalar perspective gate from current + prior
            p_gate = self.perspective_gate_net(
                torch.cat([hidden_states, prior_h], dim=-1),
            )  # (B, S, 1)
            hidden_states = hidden_states + p_gate * prior_h

        info = {
            "confidence": confidence.detach().mean().item(),
            "causal_strengths": self.causal_strengths.detach().tolist(),
            "pearl_level": PEARL_HIERARCHY_LABELS.get(
                min(self.mentalizing_depth, 2), "Counterfactual (imagining)",
            ),
        }

        return hidden_states, bdi, info


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     DECODER LAYER                                  ║
# ║                                                                    ║
# ║  A single layer of the MIND decoder, parameterised by its tier:    ║
# ║  - sensory:     Attn + Dense FFN                                   ║
# ║  - associative: Attn + Cognitive MoE                               ║
# ║  - executive:   Attn + Cognitive MoE + FracToM Integration         ║
# ║                                                                    ║
# ║  Follows Qwen3's pre-norm convention (RMSNorm before each sub-     ║
# ║  layer) with residual connections.                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MindDecoderLayer(nn.Module):
    """A single layer of the MIND decoder.

    Architecture (pre-norm):

        1. residual + RMSNorm → GQA Attention
        2. residual + RMSNorm → FFN (Dense or Cognitive MoE)
        3. [executive only] FracToM BDI Integration

    The tier (sensory / associative / executive) determines which FFN
    variant is used and whether FracToM integration is present.
    """

    def __init__(self, config: MindConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tier = config.layer_tier(layer_idx)
        self.hidden_size = config.hidden_size

        # ── Attention ────────────────────────────────────────────────
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CognitiveAttention(config, layer_idx)

        # ── FFN / MoE ───────────────────────────────────────────────
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
        )
        if self.tier == "sensory":
            self.mlp = CognitiveMLP(
                config.hidden_size, config.dense_intermediate_size,
            )
        else:
            self.mlp = CognitiveMoEBlock(config)

        # ── FracToM Integration (executive only) ────────────────────
        self.fractom: Optional[FracToMIntegration] = None
        if self.tier == "executive":
            depth = config.executive_depth(layer_idx)
            self.fractom = FracToMIntegration(config, mentalizing_depth=depth)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        prior_bdi: Optional[BDITensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Optional[Tensor], Optional[BDITensor], Dict]:
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
        if self.tier == "sensory":
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
# ║                                                                    ║
# ║  The full decoder backbone (without LM head).                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MindModel(nn.Module):
    """MIND decoder backbone: embedding → N decoder layers → final norm.

    Manages the three cognitive tiers, KV cache, and BDI state propagation
    across executive layers.
    """

    def __init__(self, config: MindConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MindDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )

    def _make_causal_mask(
        self,
        seq_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create additive causal attention mask.

        Returns (1, 1, seq_len, past_len + seq_len) where 0 = attend
        and -inf = masked (future positions).
        """
        total_len = past_len + seq_len
        mask = torch.full(
            (seq_len, total_len), float("-inf"), device=device, dtype=dtype,
        )
        # triu with diagonal=past_len+1: positions j > past_len + i are masked
        mask = torch.triu(mask, diagonal=past_len + 1)
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, T)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> MindOutput:
        """
        Parameters
        ----------
        input_ids : (B, S) — input token IDs.
        attention_mask : (B, T) — 1 = real token, 0 = padding.
            T = past_len + S (or just S if no cache).
        position_ids : (B, S) — absolute position indices.
        past_key_values : list of (K, V) per layer (length = num_layers).
        use_cache : bool — whether to return updated KV cache.

        Returns
        -------
        MindOutput with last_hidden_state, past_key_values, router_probs,
        bdi_states, cognitive_stats.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Determine past sequence length
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + S, device=device,
            ).unsqueeze(0).expand(B, -1)

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        causal_mask = self._make_causal_mask(
            S, past_len, device, hidden_states.dtype,
        )
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) where T = past_len + S, 1=attend, 0=pad
            # Use masked_fill to avoid 0.0 * -inf = NaN
            pad_positions = ~attention_mask[:, None, None, :].bool()
            causal_mask = causal_mask.masked_fill(pad_positions, float("-inf"))

        # ── Layer-by-layer forward ───────────────────────────────────
        all_router_probs: List[Tensor] = []
        all_bdi_states: List[BDITensor] = []
        all_fractom_info: Dict[str, Any] = {}
        new_kv: List[Tuple[Tensor, Tensor]] = []

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
# ║                                                                    ║
# ║  Full model with language modelling head and loss computation.     ║
# ║  Follows Qwen3MoeForCausalLM structure.                            ║
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
        self.config = config
        self.model = MindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.total_experts = config.total_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following Qwen3 convention."""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # Re-apply FracToM output gate initialisation (overridden above)
        for layer in self.model.layers:
            if layer.fractom is not None:
                nn.init.zeros_(layer.fractom.output_gate[0].weight)
                nn.init.constant_(layer.fractom.output_gate[0].bias, -3.0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        labels: Optional[Tensor] = None,
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # ── Cognitive load balancing loss ────────────────────────────
        aux_loss = None
        if outputs.router_probs is not None:
            aux_loss = cognitive_load_balancing_loss(
                outputs.router_probs,
                self.total_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss

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
    router_probs: Tuple[Tensor, ...],
    num_experts: int,
    top_k: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Load balancing auxiliary loss for cognitive MoE routing.

    Adapted from Switch Transformer (Fedus et al., 2021) as implemented
    in Qwen3-MoE.  Encourages balanced token assignment across all experts.

    The loss is:
        L = num_experts × Σ_i (f_i × p_i)

    where f_i is the fraction of tokens routed to expert i and p_i is the
    average routing probability for expert i.

    Parameters
    ----------
    router_probs : tuple of (num_tokens, num_experts) from each MoE layer.
    num_experts : total number of experts.
    top_k : number of experts selected per token.
    attention_mask : (B, S) — 1 = real, 0 = pad.  Used to exclude padding
        tokens from the load computation.
    """
    if not router_probs:
        return torch.tensor(0.0)

    device = router_probs[0].device
    # Concatenate across layers: (total_tokens_across_layers, num_experts)
    concatenated = torch.cat(
        [rp.to(device) for rp in router_probs], dim=0,
    )

    # Top-k selection for "tokens assigned to each expert"
    _, selected = torch.topk(concatenated, top_k, dim=-1)
    expert_mask = F.one_hot(selected, num_experts).float()

    if attention_mask is None:
        # f_i: mean fraction of assignments per expert
        tokens_per_expert = expert_mask.mean(dim=(0, 1))
        # p_i: mean routing probability per expert
        router_prob_per_expert = concatenated.mean(dim=0)
    else:
        # Expand attention mask to match concatenated shape.
        # Router probs have shape (B*S_new, E) per layer where S_new is the
        # number of NEW tokens processed (may differ from attention_mask width
        # when KV cache is in use).
        B = attention_mask.shape[0]
        tokens_per_layer = concatenated.shape[0] // len(router_probs)
        S_new = tokens_per_layer // B
        # Use only the last S_new columns of attention_mask (current tokens)
        current_mask = attention_mask[:, -S_new:] if attention_mask.shape[1] > S_new else attention_mask
        num_layers = len(router_probs)
        per_token_mask = current_mask.reshape(-1)                # (B*S_new,)
        per_token_mask = per_token_mask.repeat(num_layers)       # (num_layers*B*S_new,)
        per_token_mask = per_token_mask.to(device).float()

        # Weighted means excluding padding
        mask_sum = per_token_mask.sum().clamp(min=1)
        tokens_per_expert = (
            expert_mask * per_token_mask[:, None, None]
        ).sum(dim=(0, 1)) / (mask_sum * top_k)
        router_prob_per_expert = (
            concatenated * per_token_mask[:, None]
        ).sum(dim=0) / mask_sum

    return (tokens_per_expert * router_prob_per_expert).sum() * num_experts


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                ANALYSIS & INTERPRETABILITY UTILITIES                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_active_parameters(config: MindConfig) -> int:
    """Estimate active parameters per token (params actually computed).

    For MoE layers, only ``num_experts_per_tok`` experts + shared expert
    are active per token.
    """
    D = config.hidden_size
    H = config.num_attention_heads
    KV = config.num_key_value_heads
    hd = config.head_dim

    # Attention (same for all layers)
    attn_params = (H * hd + 2 * KV * hd + H * hd) * D + 2 * hd  # QKV+O + norms
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
    fractom = D * F_ + 3 * 2 * F_ * F_ + 3 * F_ * D  # obs_proj + 3 eqs + bdi_to_hidden

    sensory = config.num_sensory_layers * (attn_params + norms + dense_ffn)
    associative = config.num_associative_layers * (attn_params + norms + moe_ffn)
    executive = config.num_executive_layers * (attn_params + norms + moe_ffn + fractom)

    embedding = config.vocab_size * D  # LM head is tied

    return int(sensory + associative + executive + embedding)


def analyse_cognitive_architecture(output: MindCausalLMOutput) -> str:
    """Generate a human-readable analysis of the model's cognitive activity.

    Reports routing statistics, BDI states, and FracToM interpretability
    signals from a forward pass.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MIND — Cognitive Architecture Analysis")
    lines.append("=" * 70)

    # Router statistics
    if output.router_probs:
        lines.append("\n─── Cognitive Module Routing ───")
        all_probs = torch.cat(output.router_probs, dim=0)  # (total_tokens, E)
        # Reshape to (total_tokens, M, E_per_M) and sum over experts
        total_experts = all_probs.shape[-1]
        num_modules = len(COGNITIVE_MODULE_NAMES)
        experts_per_module = total_experts // num_modules
        module_usage = (
            all_probs.view(-1, num_modules, experts_per_module)
            .sum(-1)
            .mean(0)
        )  # (M,)
        for m in range(num_modules):
            name = COGNITIVE_MODULE_NAMES[m] if m < len(COGNITIVE_MODULE_NAMES) else f"Module {m}"
            val = module_usage[m].item()
            bar = "█" * int(val * 80) if not math.isnan(val) else "(NaN)"
            lines.append(f"  {name:14s}: {val:.3f}  {bar}")

        # Per-expert utilisation
        expert_usage = all_probs.mean(0)
        lines.append("\n  Per-expert utilisation:")
        for m in range(num_modules):
            name = COGNITIVE_MODULE_NAMES[m] if m < len(COGNITIVE_MODULE_NAMES) else f"Module {m}"
            start = m * experts_per_module
            end = start + experts_per_module
            vals = "  ".join(
                f"e{j}={expert_usage[start + j]:.3f}"
                for j in range(experts_per_module)
            )
            lines.append(f"    {name:14s}: {vals}")

    # BDI states
    if output.bdi_states:
        lines.append("\n─── FracToM BDI States (Executive Layers) ───")
        for i, bdi in enumerate(output.bdi_states):
            b_norm = bdi.belief.detach().norm(dim=-1).mean().item()
            d_norm = bdi.desire.detach().norm(dim=-1).mean().item()
            i_norm = bdi.intention.detach().norm(dim=-1).mean().item()
            depth = i
            pearl = PEARL_HIERARCHY_LABELS.get(
                min(depth, 2), "Counterfactual",
            )
            lines.append(
                f"  Depth {depth} ({pearl:25s}): "
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
        val = output.aux_loss.item() if isinstance(output.aux_loss, Tensor) else output.aux_loss
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
    print("MIND — ~1B Cognitive MoE Language Model: Smoke Test")
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

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Forward pass
    B, S = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    labels = torch.randint(0, config.vocab_size, (B, S), device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)

    print(f"\nForward pass: batch_size={B}, seq_len={S}")
    output = model(input_ids, attention_mask=attention_mask, labels=labels)

    print(f"  Logits shape:    {output.logits.shape}")
    print(f"  Loss:            {output.loss.item():.4f}")
    if output.aux_loss is not None:
        print(f"  Aux loss:        {output.aux_loss.item():.6f}")
    if output.bdi_states:
        print(f"  BDI states:      {len(output.bdi_states)} (executive layers)")

    # Backward pass
    print("\nBackward pass...")
    output.loss.backward()
    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  Gradient norm:   {grad_norm:.4f}")
    print(f"  Status:          OK ✓")

    # Cognitive analysis
    model.eval()
    with torch.no_grad():
        output_eval = model(input_ids, attention_mask=attention_mask)
    print()
    print(analyse_cognitive_architecture(output_eval))

    # KV cache test
    print("\n─── KV Cache Test (Autoregressive Generation) ───")
    model.eval()
    with torch.no_grad():
        # Prefill
        prefill_ids = input_ids[:, :32]
        prefill_mask = attention_mask[:, :32]
        prefill_out = model(
            prefill_ids, attention_mask=prefill_mask, use_cache=True,
        )
        past_kv = prefill_out.past_key_values
        print(f"  Prefill: {prefill_ids.shape[1]} tokens → "
              f"KV cache: {len(past_kv)} layers × K shape {past_kv[0][0].shape}")

        # Decode one token
        next_id = input_ids[:, 32:33]
        # Full attention mask covering past + current
        decode_mask = torch.ones(B, 33, dtype=torch.long, device=device)
        decode_out = model(
            next_id,
            attention_mask=decode_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        print(f"  Decode:  1 token → logits {decode_out.logits.shape}")
        print(f"  Updated KV cache: K shape {decode_out.past_key_values[0][0].shape}")
        print(f"  Status:  OK ✓")

    print("\n" + "=" * 70)
    print(f"MIND smoke test passed | {total:,} parameters (~{total/1e9:.2f}B)")
    print("=" * 70)


if __name__ == "__main__":
    demo_forward_backward()
