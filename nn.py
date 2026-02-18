"""
FracToM — Fractal Theory-of-Mind Neural Network
================================================

A novel neural architecture that unifies **fractal self-similarity** with
**Theory-of-Mind (ToM)** inspired hierarchical mentalizing and
**Structural Causal Models (SCM)**, producing a network that is
simultaneously more interpretable, more theoretically grounded, and
structurally richer than standard deep architectures.

Theoretical Foundations
-----------------------
1. **Fractal Self-Similarity** (Mandelbrot, 1982; Larsson et al., 2017):
   Processing blocks at every mentalizing depth share ***the same
   architectural template*** (an iterated function system on
   representations), differing only in learned parameters.

2. **BDI Architecture** (Bratman, 1987):  Mental states are factored into
   **B**eliefs (epistemic: what the agent thinks is true), **D**esires
   (motivational: what it wants), and **I**ntentions (conative: what it
   plans to do).  Each mentalizing level produces an explicit BDI triple.

3. **Hierarchical Mentalizing** (Premack & Woodruff, 1978):
   ToM is inherently recursive — "I believe that *you* believe that *I*
   intend …".  The recursion depth maps directly to fractal column depth.

4. **Bayesian Brain / Active Inference** (Friston, 2010):  Belief
   revision modules implement approximate Bayesian updating, maintaining
   calibrated epistemic uncertainty.

5. **Structural Causal Models** (Pearl, 2009):  BDI factors are linked
   by a *learnable causal graph* with edges Obs→B, B→D, B→I, D→I.
   The graph is discovered end-to-end via differentiable structure
   learning (NOTEARS; Zheng et al., 2018) while encoding ToM-specific
   causal priors from the BDI framework.

6. **Pearl's Causal Hierarchy**:  Mentalizing depths are mapped to
   Pearl's three levels of causal reasoning:
   - Level 1 (Association / seeing): depth 0 — direct perception.
   - Level 2 (Intervention / doing): depth 1 — metacognition.
   - Level 3 (Counterfactual / imagining): depth 2+ — other-modelling.
   This mapping reflects cognitive development: children first learn
   associations, then causal interventions, and finally counterfactuals
   (Gopnik & Wellman, 2012).

7. **Fractal Drop-Path** (adapted from FractalNet):  Entire mentalizing
   columns can be stochastically dropped during training, which
   (a) regularises, (b) forces every depth to carry meaning independently,
   and (c) mirrors cognitive development where deeper mentalizing emerges
   gradually.

Mathematical Formulation
------------------------
Let x denote the input observation and M_k the mental model at depth k:

    M_0(x) = φ(x)                                  — direct perception
    M_k(x) = ψ_k(x, M_0, M_1, …, M_{k-1})        — k-th order mentalizing

    ψ_k is *structurally self-similar* to ψ_j for all j ≠ k (same layer
    topology, different learned weights).

    Final output  = Σ_k  α_k(x) · M_k(x)

    where α_k(x) are input-dependent, normalised attention weights
    (soft mentalizing-depth selection).

Structural Causal Model:
    B = f_B(Obs, ε_B)          — belief from observation
    D = f_D(B, Obs, ε_D)       — desire from belief + context
    I = f_I(B, D, ε_I)         — intention from belief + desire

    Intervention:  do(B = b) severs Obs→B, recomputes D, I downstream.
    Counterfactual: abduct ε from (B,D,I)_obs, predict with Obs' + ε.

Key Innovations over Prior Art
------------------------------
* **Fractal mentalizing columns** — unlike vanilla FractalNet columns
  (which differ only in *depth*), each column corresponds to a
  *cognitive mentalizing order*, giving every column a clear
  interpretation.

* **BDI-factored latent space** — every intermediate representation is
  a structured (Belief, Desire, Intention) triple, enabling direct
  probing and mechanistic interpretability.

* **Causal BDI graph** — a differentiable, learnable DAG over BDI
  variables replaces arbitrary neural connections with causally-
  structured information flow, ensuring BDI representations respect
  the causal semantics of mental-state attribution.

* **Pearl hierarchy routing** — each mentalizing depth is softly routed
  to a causal reasoning level (association / intervention /
  counterfactual), grounding fractal depth in Pearl's causal hierarchy.

* **Counterfactual ToM** — the network can answer "what would they
  believe if they'd seen X?" via abduction-action-prediction,
  supporting false-belief reasoning and strategic deception detection.

* **Cross-depth causal discovery** — a differentiable structure learning
  module discovers which mentalizing levels causally influence others,
  providing scientific insight into ToM's computational structure.

* **Perspective-shifting cross-depth attention** — higher mentalizing
  levels *attend* to lower ones through learned perspective transforms,
  modelling the cognitive operation "simulate what *they* would
  perceive".

* **Epistemic gating** — a learned gate modulates information flow from
  each mentalizing level proportionally to its epistemic confidence,
  suppressing hallucinated mental-state attributions.

* **Developmental drop-path curriculum** — drop probability decreases
  over training for deeper columns, emulating the developmental
  timeline of ToM acquisition in children.

Architecture Overview
---------------------
    Input
      │
      ▼
  ┌────────────────────┐
  │  MentalStateEncoder │  (observation → initial BDI embedding)
  └────────┬───────────┘
           │
     ┌─────┴─────┬─────────┬── … ──┐
     ▼           ▼         ▼       ▼
  Column_0   Column_1  Column_2  Column_K    ← FractalMentalizingColumn
  (depth 0)  (depth 1) (depth 2) (depth K)     (self-similar ψ blocks)
     │           │         │       │
     ▼           ▼         ▼       ▼
  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │ SCM  │  │ SCM  │  │ SCM  │  │ SCM  │   ← StructuralCausalModel
  │ L1   │  │ L2   │  │ L3   │  │ L3   │     (Pearl hierarchy levels)
  │Assoc │  │Inter │  │ CF   │  │ CF   │
  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
     │         │         │         │
     └────┬────┴─────────┴── … ───┘
          │
          ▼
  ┌──────────────────────────┐
  │  Cross-Depth Causal      │  (discover inter-column causal links)
  │     Discovery            │
  └──────────┬───────────────┘
             │
             ▼
  ┌────────────────────────┐
  │   Attention-Weighted   │  α_k(x) soft join
  │        Join            │
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │  BeliefRevisionModule  │  Bayesian gated update
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │     Task Head          │  Classification / Prediction / ToM
  └────────────────────────┘

Usage
-----
    model = FracToMNet(
        input_dim=128,
        hidden_dim=256,
        mentalizing_depth=4,   # 0..3 orders of ToM
        num_bdi_factors=3,     # Belief, Desire, Intention
        num_heads=8,
        dropout=0.1,
        causal_model=True,     # enable SCM integration
        causal_noise_dim=16,   # exogenous noise dimensionality
    )

    x = torch.randn(32, 128)                     # batch of observations
    out, report = model(x, return_interpretability=True)
    # report contains per-level attention weights, BDI activations,
    # causal graph adjacency, Pearl hierarchy weights,
    # cross-depth causal structure, counterfactual distances, etc.

    # Inspect learned causal graph:
    causal = extract_causal_graph(report)
    print(causal["bdi_edges"])        # e.g., [("Obs", "Belief", 0.92), ...]
    print(causal["hierarchy_weights"])  # Pearl level weights per column

Requirements
------------
    Python ≥ 3.9,  PyTorch ≥ 2.0
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        UTILITY MODULES                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalDropPath(nn.Module):
    """Stochastic depth / drop-path adapted for mentalizing columns.

    During training each column (mentalizing depth) can be entirely
    dropped with probability *p*.  Supports a **developmental
    curriculum**: ``column_index`` controls a linear scaling so that
    deeper mentalizing levels start with higher drop probability and
    are gradually "unlocked" as training progresses.

    Parameters
    ----------
    drop_prob : float
        Base drop probability (applied to depth-0 column).
    max_depth : int
        Maximum mentalizing depth in the network.
    developmental : bool
        If True, scale drop probability linearly with depth:
        p_k = drop_prob + (1 − drop_prob) × (k / max_depth) × curriculum_factor.
    """

    def __init__(
        self,
        drop_prob: float = 0.15,
        max_depth: int = 4,
        developmental: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_depth = max(max_depth, 1)
        self.developmental = developmental
        # curriculum_factor ∈ [0, 1]; start at 1 (high drop for deep columns),
        # anneal toward 0 during training.
        self.register_buffer(
            "curriculum_factor", torch.tensor(1.0)
        )

    def set_curriculum(self, factor: float) -> None:
        """Call during training to anneal deep-column drop rates.

        ``factor`` should go from 1.0 (beginning) → 0.0 (end).
        """
        self.curriculum_factor.fill_(max(0.0, min(1.0, factor)))

    def forward(self, x: Tensor, column_index: int = 0) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        p = self.drop_prob
        if self.developmental:
            depth_ratio = column_index / self.max_depth
            p = p + (1.0 - p) * depth_ratio * self.curriculum_factor.item()
        p = min(p, 0.999)

        keep = 1.0 - p
        # Bernoulli mask per sample in the batch (structured drop)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep  # rescale to maintain expectation


class GatedResidual(nn.Module):
    """Gated Residual Unit (GRU-style gate on residual branch).

    Computes:  output = gate ⊙ f(x) + (1 − gate) ⊙ x
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.gate = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        g = torch.sigmoid(self.gate(h))
        h = self.act(h)
        h = self.drop(self.fc2(h))
        return g * h + (1.0 - g) * x


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     BDI MENTAL-STATE ENCODER                       ║
# ╚══════════════════════════════════════════════════════════════════════╝


class BDIState:
    """Structured container for a Belief-Desire-Intention triple.

    All three tensors share shape (batch, factor_dim).
    Provides named access and packs/unpacks to a single tensor.
    """

    __slots__ = ("belief", "desire", "intention")

    def __init__(self, belief: Tensor, desire: Tensor, intention: Tensor):
        self.belief = belief
        self.desire = desire
        self.intention = intention

    def pack(self) -> Tensor:
        """→ (batch, 3 × factor_dim)"""
        return torch.cat([self.belief, self.desire, self.intention], dim=-1)

    @staticmethod
    def unpack(x: Tensor, factor_dim: int) -> "BDIState":
        b, d, i = x.split(factor_dim, dim=-1)
        return BDIState(b, d, i)

    def detach(self) -> "BDIState":
        return BDIState(
            self.belief.detach(),
            self.desire.detach(),
            self.intention.detach(),
        )


class MentalStateEncoder(nn.Module):
    """Encodes raw observations into an initial BDI mental-state triple.

    Architecture
    ------------
    Shared trunk (2-layer MLP) → three parallel projection heads, one
    per BDI factor.  An optional *epistemic uncertainty head* outputs a
    scalar σ indicating how confident the belief component is.

    Why BDI?  The Belief-Desire-Intention framework (Bratman, 1987)
    decomposes mental states into *epistemic* (what is believed true),
    *motivational* (what is desired), and *conative* (what is intended)
    components.  This factoring enables targeted probing: we can read off
    a network's "beliefs" or "intentions" without training an external
    classifier.
    """

    def __init__(
        self,
        input_dim: int,
        factor_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = factor_dim * 4
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_belief = nn.Linear(hidden, factor_dim)
        self.proj_desire = nn.Linear(hidden, factor_dim)
        self.proj_intention = nn.Linear(hidden, factor_dim)
        # epistemic confidence (scalar per sample)
        self.proj_uncertainty = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tuple[BDIState, Tensor]:
        """
        Returns
        -------
        bdi : BDIState
        uncertainty : Tensor, shape (batch, 1)
            σ > 0 — higher means *more* uncertain.
        """
        h = self.trunk(x)
        bdi = BDIState(
            belief=self.proj_belief(h),
            desire=self.proj_desire(h),
            intention=self.proj_intention(h),
        )
        sigma = F.softplus(self.proj_uncertainty(h))  # ensure > 0
        return bdi, sigma


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               SELF-SIMILAR MENTALIZING BLOCK (ψ)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SelfSimilarBlock(nn.Module):
    """The ***fractal primitive***: a single mentalizing transformation.

    Every mentalizing column, regardless of depth *k*, is composed of
    stacked instances of this block.  The blocks share the same
    ***template*** (topology, activation structure, normalisation scheme)
    but have **independent weights**, which is exactly the definition of a
    self-similar iterated function system applied to neural representation
    space.

    Architecture
    ------------
    LayerNorm → Multi-Head Self-Attention → Residual₁
              → Feed-Forward (GEGLU) → Residual₂
              → BDI Re-Factoring projection

    Compared to a vanilla Transformer block the only additions are:
      (a) BDI re-factoring at the output, and
      (b) an optional *context* tensor from shallower mentalizing levels
          injected via cross-attention.
    """

    def __init__(
        self,
        dim: int,
        factor_dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.factor_dim = factor_dim

        # --- self-attention ---
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )

        # --- optional cross-attention (for perspective shifting) ---
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_ca = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True,
            )

        # --- feed-forward with GEGLU ---
        self.norm_ff = nn.LayerNorm(dim)
        inner = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, inner * 2),
            _GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

        # --- BDI re-factoring ---
        self.bdi_proj = nn.Linear(dim, factor_dim * 3)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, BDIState]:
        """
        Parameters
        ----------
        x : (batch, seq, dim) or (batch, dim) — if 2-D, unsqueezes to
            (batch, 1, dim) internally.
        context : optional (batch, ctx_len, dim) from shallower level.

        Returns
        -------
        h : Tensor, same shape as x.
        bdi : BDIState from the re-factoring head.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)  # (B, 1, D)

        # self-attention
        h = self.norm_sa(x)
        h_sa, _ = self.self_attn(h, h, h)
        x = x + h_sa

        # cross-attention (perspective shifting)
        if self.use_cross_attn and context is not None:
            h = self.norm_ca(x)
            ctx = context.unsqueeze(1) if context.ndim == 2 else context
            h_ca, _ = self.cross_attn(h, ctx, ctx)
            x = x + h_ca

        # feed-forward
        x = x + self.ff(self.norm_ff(x))

        # BDI re-factoring: project hidden state → structured BDI
        bdi_raw = self.bdi_proj(x.mean(dim=1))  # pool over seq
        bdi = BDIState.unpack(bdi_raw, self.factor_dim)

        if squeeze:
            x = x.squeeze(1)
        return x, bdi


class _GEGLU(nn.Module):
    """Gated GELU activation (Shazeer, 2020)."""

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║           PERSPECTIVE-SHIFTING CROSS-DEPTH ATTENTION               ║
# ╚══════════════════════════════════════════════════════════════════════╝


class PerspectiveShiftAttention(nn.Module):
    """Models the cognitive operation of *adopting another's viewpoint*.

    When mentalizing at depth k, the agent needs to "imagine" what a
    simpler agent (depth < k) would perceive.  This module takes the
    current-level representation as *query* and all lower-level
    representations as *keys/values*, applying a learned perspective
    transform before the attention operation.

    Cognitive analogy
    -----------------
    The perspective transform rotates the representational coordinate
    system — analogous to "putting yourself in someone else's shoes"
    before reading off their likely beliefs/desires/intentions.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.perspective_transform = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.perspective_transform.weight)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        query: Tensor,
        lower_level_states: List[Tensor],
    ) -> Tensor:
        """
        Parameters
        ----------
        query : (batch, dim) — current mentalizing level.
        lower_level_states : list of (batch, dim) from shallower levels.

        Returns
        -------
        (batch, dim) — perspective-shifted representation.
        """
        if not lower_level_states:
            return query

        # stack lower-level states → (batch, num_lower, dim)
        kv = torch.stack(lower_level_states, dim=1)
        kv = self.perspective_transform(kv)  # rotate into "their" frame
        kv = self.norm_kv(kv)

        q = self.norm_q(query).unsqueeze(1)  # (B, 1, D)
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(1)  # (B, D)

        # gated residual — lets the network ignore perspective info
        g = self.gate(out)
        return query + g * out


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    FRACTAL MENTALIZING COLUMN                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalMentalizingColumn(nn.Module):
    """A single *column* in the fractal architecture.

    Column *k* represents **k-th order mentalizing**:
    - Column 0: direct stimulus→response (no mentalizing)
    - Column 1: self-model / metacognition
    - Column 2: basic other-modelling ("I think you think …")
    - Column k: k-th order mentalizing

    Each column stacks ``num_blocks`` SelfSimilarBlocks.  Columns with
    k > 0 receive cross-level context from all columns 0..k−1 via
    PerspectiveShiftAttention, implementing the recursive nature of
    mentalizing.

    The architecture within every column is ***identical in topology***
    (self-similarity), differing only in learned parameters.
    """

    def __init__(
        self,
        depth_index: int,
        dim: int,
        factor_dim: int,
        num_blocks: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth_index = depth_index

        self.blocks = nn.ModuleList([
            SelfSimilarBlock(
                dim=dim,
                factor_dim=factor_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                use_cross_attn=(depth_index > 0),
            )
            for _ in range(num_blocks)
        ])

        # perspective-shift attention (only for depth > 0)
        self.perspective_attn: Optional[PerspectiveShiftAttention] = None
        if depth_index > 0:
            self.perspective_attn = PerspectiveShiftAttention(
                dim, num_heads, dropout,
            )

        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        lower_level_outputs: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[BDIState]]:
        """
        Parameters
        ----------
        x : (batch, dim) — encoded observation.
        lower_level_outputs : list of (batch, dim) from columns 0..k−1.

        Returns
        -------
        h : (batch, dim) — this column's output representation.
        bdis : list of BDIState, one per block in this column.
        """
        bdis: List[BDIState] = []

        # perspective shift: enrich input with lower-level context
        if self.perspective_attn is not None and lower_level_outputs:
            x = self.perspective_attn(x, lower_level_outputs)

        h = x
        for block in self.blocks:
            ctx = None
            if lower_level_outputs and block.use_cross_attn:
                # use the final lower-level output as cross-attn context
                ctx = lower_level_outputs[-1]
            h, bdi = block(h, context=ctx)
            bdis.append(bdi)

        h = self.output_norm(h)
        return h, bdis


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   EPISTEMIC GATING MODULE                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


class EpistemicGate(nn.Module):
    """Modulates column outputs by epistemic confidence.

    Each mentalizing column's output is scaled by
    ``confidence = 1 / (1 + σ_k)`` where σ_k is the epistemic
    uncertainty of that column's belief component.

    Purpose: deeper mentalizing (higher k) is inherently noisier because
    it requires modelling another agent's internal state — something for
    which direct evidence is scarce.  The epistemic gate gracefully
    degrades the influence of unreliable high-order attributions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus(),  # σ > 0
        )

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        gated_h : (batch, dim) — scaled by confidence.
        sigma : (batch, 1) — epistemic uncertainty.
        """
        sigma = self.uncertainty_proj(h)           # (B, 1)
        confidence = 1.0 / (1.0 + sigma)           # ∈ (0, 1]
        return h * confidence, sigma


# ╔══════════════════════════════════════════════════════════════════════╗
# ║       GUIDING BELIEF MODULE  (FractalGen-inspired)                 ║
# ║                                                                    ║
# ║  Analogous to FractalGen's "guiding pixel" (Li et al., 2025):      ║
# ║  before fine-grained mentalizing, produce a coarse gist belief     ║
# ║  that conditions every SelfSimilarBlock via FiLM modulation.       ║
# ║                                                                    ║
# ║  Cognitive grounding: "gist processing" (Oliva & Torralba, 2006)   ║
# ║  — humans form rapid holistic impressions before detailed analysis.║
# ╚══════════════════════════════════════════════════════════════════════╝


class GuidingBeliefModule(nn.Module):
    """Coarse-to-fine conditioning via FiLM modulation.

    Inspired by FractalGen's *guiding pixel* mechanism (Li et al., 2025),
    this module predicts a low-dimensional "gist belief" from the input
    representation *before* a mentalizing column processes it.  The gist
    is then injected into each SelfSimilarBlock via Feature-wise Linear
    Modulation (FiLM; Perez et al., 2018):

        h' = γ(gist) ⊙ h + β(gist)

    This gives each column a coarse prior over the cognitive situation
    before running expensive multi-head attention.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input representation.
    gist_dim : int
        Dimensionality of the gist belief vector.
    output_dim : int
        Dimensionality of the column hidden states being modulated.
    """

    def __init__(
        self,
        input_dim: int,
        gist_dim: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()
        self.gist_encoder = nn.Sequential(
            nn.Linear(input_dim, gist_dim),
            nn.GELU(),
            nn.Linear(gist_dim, gist_dim),
        )
        # FiLM parameters: γ and β
        self.film_gamma = nn.Linear(gist_dim, output_dim)
        self.film_beta = nn.Linear(gist_dim, output_dim)

        # Initialise γ → 1, β → 0 so the module is identity at init
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.bias)
        nn.init.zeros_(self.film_beta.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gist belief and FiLM parameters.

        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        gamma : (batch, output_dim) — multiplicative modulation.
        beta  : (batch, output_dim) — additive modulation.
        """
        gist = self.gist_encoder(x)  # (B, gist_dim)
        gamma = self.film_gamma(gist)  # (B, output_dim)
        beta = self.film_beta(gist)    # (B, output_dim)
        return gamma, beta

    @staticmethod
    def modulate(h: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """Apply FiLM modulation: h' = γ ⊙ h + β."""
        if h.ndim == 3:  # (B, S, D)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * h + beta


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    BAYESIAN BELIEF REVISION                        ║
# ╚══════════════════════════════════════════════════════════════════════╝


class BeliefRevisionModule(nn.Module):
    """Approximate Bayesian belief updating.

    Inspired by the **Bayesian brain hypothesis** (Friston, 2010), this
    module maintains a *prior belief* state and updates it with incoming
    *evidence* (the joined column representations) via a gated mechanism
    that approximates:

        posterior ∝ prior × likelihood

    In neural-network terms:

        belief_new = gate × evidence + (1 − gate) × prior

    The gate is computed from both prior and evidence, functioning as an
    approximate log-likelihood ratio.

    This module is applied ***after*** the column join, providing a final
    "sanity check" that integrates the aggregated mental-state
    information into a coherent belief before handing off to the task
    head.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.prior_proj = nn.Linear(dim, dim)
        self.evidence_proj = nn.Linear(dim, dim)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, prior: Tensor, evidence: Tensor) -> Tensor:
        prior_h = self.prior_proj(prior)
        evidence_h = self.evidence_proj(evidence)
        gate = self.gate_net(torch.cat([prior_h, evidence_h], dim=-1))
        posterior = gate * evidence_h + (1.0 - gate) * prior_h
        return self.norm(posterior)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          STRUCTURAL CAUSAL MODEL (SCM) MODULES                     ║
# ║                                                                    ║
# ║  Integrates Pearl's Causal Hierarchy (Association, Intervention,   ║
# ║  Counterfactual) with the fractal mentalizing architecture.        ║
# ║                                                                    ║
# ║  References:                                                       ║
# ║  - Pearl (2009), "Causality" — SCM framework & do-calculus         ║
# ║  - Zheng et al. (2018), "DAGs with NO TEARS" — differentiable DAG ║
# ║  - Bratman (1987) — BDI causal structure                           ║
# ║  - Premack & Woodruff (1978) — ToM as causal mental-state         ║
# ║    reasoning about hidden variables                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


class LearnableCausalGraph(nn.Module):
    """Differentiable Directed Acyclic Graph for causal structure learning.

    Parameterises a continuous adjacency matrix A ∈ ℝ^{d×d} and enforces
    the DAG constraint via the NOTEARS trace-exponential penalty
    (Zheng et al., 2018):

        h(A) = tr(e^{A ⊙ A}) − d = 0

    Entry A[i,j] > 0 means variable i causally influences variable j.

    Parameters
    ----------
    num_variables : int
        Number of causal variables (e.g., 4 for Obs, B, D, I).
    init_sparsity : float
        Scaling of initial random weights (lower → sparser prior).
    """

    def __init__(self, num_variables: int, init_sparsity: float = 0.3):
        super().__init__()
        self.num_variables = num_variables
        self.raw_adjacency = nn.Parameter(
            torch.randn(num_variables, num_variables) * init_sparsity
        )
        self.register_buffer(
            "diag_mask", 1.0 - torch.eye(num_variables),
        )

    @property
    def adjacency(self) -> Tensor:
        """Weighted adjacency matrix ∈ [0, 1]^{d×d}, no self-loops."""
        return torch.sigmoid(self.raw_adjacency) * self.diag_mask

    def dag_penalty(self) -> Tensor:
        """NOTEARS acyclicity constraint: h(A) = tr(e^{A⊙A}) − d.

        Returns 0 iff the graph is a DAG.
        """
        A = self.adjacency
        M = A * A  # element-wise square ensures non-negativity
        # torch.matrix_exp is not implemented on MPS; fall back to CPU.
        orig_device = M.device
        expm = torch.matrix_exp(M.cpu()).to(orig_device)
        return torch.trace(expm) - self.num_variables

    def forward(self) -> Tensor:
        return self.adjacency


class StructuralEquationNetwork(nn.Module):
    """Neural structural equation for one endogenous variable.

    Implements  X_j = f_j(Pa(X_j), ε_j)  where Pa(X_j) are the weighted
    causal parents and ε_j is an exogenous noise term.

    The function f_j is a gated MLP: the gate controls how much of the
    structural output vs. the raw parent signal to use, providing a smooth
    interpolation between fully-causal and pass-through modes.

    Parameters
    ----------
    dim : int
        Dimensionality of each variable representation.
    noise_dim : int
        Dimensionality of the exogenous noise vector ε.
    dropout : float
        Dropout rate.
    """

    def __init__(self, dim: int, noise_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.noise_dim = noise_dim
        self.parent_proj = nn.Linear(dim, dim)
        self.noise_proj = nn.Linear(noise_dim, dim)
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        parent_repr: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        parent_repr : (batch, dim) — aggregated parent representation.
        noise : (batch, noise_dim) — exogenous noise (sampled or inferred).

        Returns
        -------
        (batch, dim) — structural equation output.
        """
        h_parent = self.parent_proj(parent_repr)
        if noise is None:
            noise = torch.randn(
                parent_repr.shape[0], self.noise_dim,
                device=parent_repr.device,
            )
        h_noise = self.noise_proj(noise)
        combined = self.combine(torch.cat([h_parent, h_noise], dim=-1))
        g = self.gate(combined)
        return g * combined + (1.0 - g) * h_parent


class StructuralCausalModel(nn.Module):
    """Differentiable Structural Causal Model over BDI mental-state variables.

    Encodes the causal structure of Theory-of-Mind reasoning:

        Observation → Belief   (epistemic update)
        Belief + Context → Desire   (motivational computation)
        Belief + Desire → Intention   (deliberation)

    The causal graph is **learnable** — initialised from a theoretically-
    motivated BDI prior but refined through end-to-end gradient descent,
    achieving causal *discovery* alongside causal *reasoning*.

    Supports all three levels of **Pearl's Causal Hierarchy**:

    Level 1 (Association):  P(I | B, D)
        Standard forward pass through structural equations.

    Level 2 (Intervention):  P(I | do(B = b), D)
        ``do``-operator severs incoming edges to the intervention target.

    Level 3 (Counterfactual):  P(I_cf | B_obs, D_obs)
        Abduction → Action → Prediction cycle (Pearl, 2009).

    Mathematical Formulation
    ------------------------
    The SCM is defined by the tuple ⟨U, V, F, P(U)⟩ where:
    - U = {ε_B, ε_D, ε_I} are exogenous noise variables
    - V = {Obs, B, D, I} are endogenous variables
    - F = {f_B, f_D, f_I} are structural equations (neural networks)
    - P(U) is the distribution over exogenous noise

    Structural equations:
        B = f_B(Obs, ε_B)          — belief from observation
        D = f_D(B, Obs, ε_D)       — desire from belief + context
        I = f_I(B, D, ε_I)         — intention from belief + desire

    Intervention do(B = b):
        Ĩ = f_I(b, f_D(b, Obs, ε_D), ε_I)   — with Obs→B edge severed

    Counterfactual (given observed BDI, alternative observation Obs'):
        1. Abduct: ε̂ = g(B_obs, D_obs, I_obs)
        2. Predict: B_cf = f_B(Obs', ε̂_B), D_cf = f_D(B_cf, Obs', ε̂_D), ...

    Parameters
    ----------
    factor_dim : int
        Dimensionality of each BDI factor.
    noise_dim : int
        Exogenous noise dimensionality.
    dropout : float
        Dropout rate in structural equations.
    """

    # Variable indices in the causal graph
    VAR_OBS = 0
    VAR_BELIEF = 1
    VAR_DESIRE = 2
    VAR_INTENTION = 3
    NUM_VARS = 4

    def __init__(
        self,
        factor_dim: int,
        noise_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.factor_dim = factor_dim
        self.noise_dim = noise_dim

        # Learnable causal graph over {Obs, B, D, I}
        self.causal_graph = LearnableCausalGraph(
            num_variables=self.NUM_VARS,
            init_sparsity=0.3,
        )

        # Initialise with BDI prior (Bratman, 1987):
        #   Obs → B (strong), Obs → D (weak)
        #   B → D,  B → I
        #   D → I
        with torch.no_grad():
            prior = torch.tensor([
                #  O     B     D     I
                [0.0,  2.0,  0.5,  0.0],   # from Obs
                [0.0,  0.0,  1.5,  2.0],   # from Belief
                [0.0,  0.0,  0.0,  1.5],   # from Desire
                [0.0,  0.0,  0.0,  0.0],   # from Intention
            ])
            self.causal_graph.raw_adjacency.copy_(prior)

        # Structural equations for each endogenous BDI variable
        self.eq_belief = StructuralEquationNetwork(
            factor_dim, noise_dim, dropout,
        )
        self.eq_desire = StructuralEquationNetwork(
            factor_dim, noise_dim, dropout,
        )
        self.eq_intention = StructuralEquationNetwork(
            factor_dim, noise_dim, dropout,
        )

        # Noise encoder for abduction (counterfactual reasoning step 1)
        # Infers exogenous noise ε from observed BDI values
        self.noise_encoder = nn.Sequential(
            nn.Linear(factor_dim * 3, factor_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim * 2, noise_dim * 3),
        )

        # Parent aggregators: combine parent signals weighted by edge strength
        self.parent_agg = nn.ModuleList([
            nn.Linear(factor_dim * self.NUM_VARS, factor_dim)
            for _ in range(self.NUM_VARS)
        ])

    # -- helpers ----------------------------------------------------------

    def _aggregate_parents(
        self,
        var_idx: int,
        variables: List[Tensor],
        adjacency: Tensor,
    ) -> Tensor:
        """Weight parent representations by causal edge strength.

        For variable j, computes  Σ_i A[i,j] · proj(X_i)  where A is the
        adjacency matrix.
        """
        parent_weights = adjacency[:, var_idx]  # (NUM_VARS,)
        weighted = [v * parent_weights[i] for i, v in enumerate(variables)]
        stacked = torch.cat(weighted, dim=-1)   # (B, factor_dim * NUM_VARS)
        return self.parent_agg[var_idx](stacked)

    # -- Level 1: Associational (seeing) ---------------------------------

    def forward(
        self,
        observation: Tensor,
        bdi_init: "BDIState",
    ) -> Tuple["BDIState", Dict[str, Tensor]]:
        """Forward (associational) causal pass — P(BDI | Obs).

        Parameters
        ----------
        observation : (batch, factor_dim) — encoded observation.
        bdi_init : initial BDI from the MentalStateEncoder.

        Returns
        -------
        bdi_causal : BDIState — causally-refined BDI triple.
        info : dict with ``adjacency`` (Tensor) and ``dag_penalty`` (Tensor).
        """
        A = self.causal_graph()
        variables = [
            observation,
            bdi_init.belief,
            bdi_init.desire,
            bdi_init.intention,
        ]

        # Causally-ordered generation following the graph:
        # Belief ← parents(Obs, ...)
        parent_b = self._aggregate_parents(self.VAR_BELIEF, variables, A)
        belief = self.eq_belief(parent_b)
        variables[self.VAR_BELIEF] = belief

        # Desire ← parents(Obs, Belief, ...)
        parent_d = self._aggregate_parents(self.VAR_DESIRE, variables, A)
        desire = self.eq_desire(parent_d)
        variables[self.VAR_DESIRE] = desire

        # Intention ← parents(Belief, Desire, ...)
        parent_i = self._aggregate_parents(self.VAR_INTENTION, variables, A)
        intention = self.eq_intention(parent_i)

        return BDIState(belief, desire, intention), {
            "adjacency": A,
            "dag_penalty": self.causal_graph.dag_penalty(),
        }

    # -- Level 2: Interventional (doing) — do-operator -------------------

    def intervene(
        self,
        observation: Tensor,
        bdi_init: "BDIState",
        target: int,
        value: Tensor,
    ) -> "BDIState":
        """Pearl's do(X_target = value).

        Fixes a variable to a specific value and recomputes all downstream
        variables through the structural equations, while **severing**
        incoming causal edges to the target.

        This answers questions like:
        - "If I *set* the agent's belief to b, what would their intention be?"
        - "If I *force* a desire, how does that change intentions?"

        Parameters
        ----------
        target : VAR_BELIEF, VAR_DESIRE, or VAR_INTENTION.
        value : (batch, factor_dim) — the intervention value.
        """
        A = self.causal_graph().clone()
        A[:, target] = 0.0  # sever incoming edges (do-operator)

        variables = [
            observation,
            bdi_init.belief,
            bdi_init.desire,
            bdi_init.intention,
        ]
        variables[target] = value

        # Regenerate all downstream variables in causal order
        for var_idx in range(self.VAR_BELIEF, self.NUM_VARS):
            if var_idx == target:
                continue  # skip the intervened variable
            parent = self._aggregate_parents(var_idx, variables, A)
            eq = [self.eq_belief, self.eq_desire, self.eq_intention][
                var_idx - 1
            ]
            variables[var_idx] = eq(parent)

        return BDIState(
            variables[self.VAR_BELIEF],
            variables[self.VAR_DESIRE],
            variables[self.VAR_INTENTION],
        )

    # -- Level 3: Counterfactual (imagining) ------------------------------

    def counterfactual(
        self,
        observation: Tensor,
        bdi_observed: "BDIState",
        cf_observation: Tensor,
    ) -> "BDIState":
        """Counterfactual reasoning via abduction-action-prediction (Pearl, 2009).

        Three steps:
        1. **Abduction** — infer exogenous noise ε from observed BDI.
        2. **Action** — substitute the counterfactual observation.
        3. **Prediction** — forward pass with inferred ε + new observation.

        Answers: "If the observation had been Obs' instead of Obs,
        what would the BDI states have been?"

        This is the core of ToM counterfactual reasoning:
        "If Sally *had* seen Anne move the ball, would she look in the
        new location?"  (Answer: yes → the model needs counterfactual
        capacity to distinguish this from the false-belief case.)
        """
        # Step 1: Abduction — infer exogenous noise from observed BDI
        bdi_packed = bdi_observed.pack()
        noise_all = self.noise_encoder(bdi_packed)
        noise_b, noise_d, noise_i = noise_all.split(self.noise_dim, dim=-1)

        # Step 2 & 3: Action + Prediction with counterfactual observation
        A = self.causal_graph()
        variables = [
            cf_observation,
            torch.zeros_like(bdi_observed.belief),
            torch.zeros_like(bdi_observed.desire),
            torch.zeros_like(bdi_observed.intention),
        ]

        parent_b = self._aggregate_parents(self.VAR_BELIEF, variables, A)
        cf_belief = self.eq_belief(parent_b, noise_b)
        variables[self.VAR_BELIEF] = cf_belief

        parent_d = self._aggregate_parents(self.VAR_DESIRE, variables, A)
        cf_desire = self.eq_desire(parent_d, noise_d)
        variables[self.VAR_DESIRE] = cf_desire

        parent_i = self._aggregate_parents(self.VAR_INTENTION, variables, A)
        cf_intention = self.eq_intention(parent_i, noise_i)

        return BDIState(cf_belief, cf_desire, cf_intention)


class CausalHierarchyRouter(nn.Module):
    """Routes mentalizing columns through Pearl's 3-level causal hierarchy.

    Maps fractal mentalizing depth to causal reasoning mode:

    - **Depth 0** (Direct perception) → Level 1: *Association*
      P(Y|X) — observe and predict.

    - **Depth 1** (Metacognition) → Level 2: *Intervention*
      P(Y|do(X)) — what if I change my belief?

    - **Depth 2+** (Other-modelling) → Level 3: *Counterfactual*
      P(Y_{x'}|X,Y) — what would they believe if they'd seen X'?

    This mapping reflects the cognitive development of causal reasoning:
    children first learn associations, then interventions, and finally
    counterfactual reasoning — paralleling ToM development stages
    (Gopnik & Wellman, 2012).

    Routing is **soft**: a learned gate blends all three levels with
    depth-dependent priors biasing toward the theoretically appropriate
    level.

    Parameters
    ----------
    dim : int
        Hidden dimensionality.
    max_depth : int
        Maximum mentalizing depth.
    """

    def __init__(self, dim: int, max_depth: int):
        super().__init__()
        self.max_depth = max(max_depth, 1)
        self.num_levels = 3  # association, intervention, counterfactual
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, self.num_levels),
        )
        self.register_buffer("_dummy", torch.empty(0))

    def _depth_prior(self, depth_index: int) -> Tensor:
        """Soft prior over causal levels for mentalizing depth *k*."""
        p = torch.zeros(self.num_levels, device=self._dummy.device)
        if depth_index == 0:
            p[0], p[1], p[2] = 2.0, 0.5, 0.0   # association-dominated
        elif depth_index == 1:
            p[0], p[1], p[2] = 0.5, 2.0, 0.5   # intervention-dominated
        else:
            ratio = min(depth_index / self.max_depth, 1.0)
            p[0] = 0.2
            p[1] = 1.0 - 0.5 * ratio
            p[2] = 1.0 + ratio                   # counterfactual-dominated
        return p

    def forward(self, h: Tensor, depth_index: int) -> Tensor:
        """
        Returns
        -------
        (batch, 3) — soft weights over [association, intervention, counterfactual].
        """
        logits = self.router(h) + self._depth_prior(depth_index)
        return F.softmax(logits, dim=-1)


class CausalDiscoveryModule(nn.Module):
    """Discovers cross-depth causal structure from BDI representations.

    Scores potential causal edges *between* mentalizing depths using a
    pairwise neural scorer, enforcing:
    1. A **depth-ordering prior** — shallower depths causally precede
       deeper ones (reflecting cognitive development).
    2. A **DAG constraint** — the discovered cross-depth graph must be
       acyclic.
    3. A **sparsity prior** — only the most informative causal links
       are retained.

    This enables the network to learn *which* mentalizing levels
    causally influence others — e.g., discovering that depth-0 beliefs
    structurally inform depth-2 other-modelling.

    The discovered structure can be read off for scientific interpretation:
    a strong edge from column 1 → column 3 would mean "metacognitive
    beliefs causally drive second-order ToM".

    Parameters
    ----------
    factor_dim : int
        BDI factor dimensionality.
    max_depth : int
        Maximum mentalizing depth.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        factor_dim: int,
        max_depth: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_depth = max_depth

        self.edge_scorer = nn.Sequential(
            nn.Linear(factor_dim * 2, factor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim, 1),
        )

        # Depth-ordering prior: shallower → deeper preferred
        self.depth_bias = nn.Parameter(
            torch.zeros(max_depth + 1, max_depth + 1),
        )
        with torch.no_grad():
            for i in range(max_depth + 1):
                for j in range(max_depth + 1):
                    if i < j:
                        self.depth_bias[i, j] = 1.0   # forward causation
                    elif i > j:
                        self.depth_bias[i, j] = -2.0   # discourage backward

    def discover(
        self,
        bdi_per_depth: Dict[int, "BDIState"],
        observation: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Discover cross-depth causal adjacency.

        Parameters
        ----------
        bdi_per_depth : dict mapping depth index → BDIState.
        observation : (batch, factor_dim).

        Returns
        -------
        cross_adj : (K+1, K+1) — discovered cross-depth adjacency.
        penalty : scalar — DAG + sparsity penalty.
        """
        depth_reps = []
        for k in sorted(bdi_per_depth.keys()):
            bdi = bdi_per_depth[k]
            rep = (bdi.belief + bdi.desire + bdi.intention) / 3.0
            depth_reps.append(rep)

        K = len(depth_reps)
        device = observation.device

        adj = torch.zeros(K, K, device=device)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                pair = torch.cat([
                    depth_reps[i].mean(0, keepdim=True),
                    depth_reps[j].mean(0, keepdim=True),
                ], dim=-1)
                score = self.edge_scorer(pair).squeeze()
                bias = self.depth_bias[i, j] if (
                    i < self.depth_bias.shape[0] and
                    j < self.depth_bias.shape[1]
                ) else 0.0
                adj[i, j] = torch.sigmoid(score + bias)

        # DAG penalty (NOTEARS)
        M = adj * adj
        orig_device = M.device
        expm = torch.matrix_exp(M.cpu()).to(orig_device)
        dag_pen = torch.trace(expm) - K
        sparsity = adj.sum()
        penalty = dag_pen + 0.01 * sparsity

        return adj, penalty


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 ATTENTION-WEIGHTED COLUMN JOIN                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MentalizingJoin(nn.Module):
    """Input-dependent soft selection over mentalizing depths.

    Given output vectors h_0 … h_K from the columns, this module
    computes input-conditioned attention weights α_k(x) and returns:

        joined = Σ_k α_k h_k

    The α weights are directly interpretable: they reveal which
    mentalizing depth the network relies on for a given input.  High
    α_0 means the task was solvable without mentalizing; high α_2 means
    second-order ToM was required.
    """

    def __init__(self, dim: int, max_depth: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)
        self.max_depth = max_depth

    def forward(
        self,
        column_outputs: List[Tensor],
        input_rep: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        joined : (batch, dim)
        alpha : (batch, K+1) — interpretable mentalizing-depth weights.
        """
        q = self.query(input_rep).unsqueeze(1)          # (B, 1, D)
        keys = torch.stack(column_outputs, dim=1)       # (B, K+1, D)
        k = self.key(keys)
        scores = (q * k).sum(-1) / self.scale           # (B, K+1)
        alpha = F.softmax(scores, dim=-1)               # (B, K+1)
        joined = (alpha.unsqueeze(-1) * keys).sum(1)    # (B, D)
        return joined, alpha


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  INTERPRETABILITY REPORT                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


@dataclass
class InterpretabilityReport:
    """Collects all interpretable signals produced during a forward pass.

    Attributes
    ----------
    depth_weights : Tensor (batch, K+1)
        α_k — attention weight assigned to each mentalizing depth.
    bdi_states : dict[int, list[BDIState]]
        Per-column, per-block BDI triples (key = column depth index).
    column_uncertainties : dict[int, Tensor]
        Per-column epistemic uncertainty σ_k (batch, 1).
    belief_revision_gate : Tensor | None
        Gate values from the BeliefRevisionModule (batch, dim).
    causal_adjacency : Tensor | None
        Learned BDI causal graph adjacency matrix (NUM_VARS, NUM_VARS).
        Entry [i,j] indicates causal influence strength from variable i
        to variable j, where 0=Obs, 1=Belief, 2=Desire, 3=Intention.
    causal_hierarchy_weights : dict[int, Tensor] | None
        Per-column soft weights over Pearl's 3 causal levels
        (association, intervention, counterfactual). Shape (batch, 3).
    cross_depth_adjacency : Tensor | None
        Discovered cross-depth causal structure (K+1, K+1).
        Entry [i,j] indicates causal influence from column i to column j.
    dag_penalty : Tensor | None
        NOTEARS DAG constraint value (0 = perfect DAG).  Differentiable.
    counterfactual_distances : dict[int, float] | None
        Per-column L2 distance between factual and counterfactual BDI.
        Larger values at deeper columns indicate richer counterfactual
        reasoning.
    """

    depth_weights: Tensor
    bdi_states: Dict[int, List[BDIState]] = field(default_factory=dict)
    column_uncertainties: Dict[int, Tensor] = field(default_factory=dict)
    belief_revision_gate: Optional[Tensor] = None
    causal_adjacency: Optional[Tensor] = None
    causal_hierarchy_weights: Optional[Dict[int, Tensor]] = None
    cross_depth_adjacency: Optional[Tensor] = None
    dag_penalty: Optional[Tensor] = None
    counterfactual_distances: Optional[Dict[int, float]] = None
    # --- FractalGen-inspired enhancements ---
    auxiliary_logits: Optional[Dict[int, Tensor]] = None
    guiding_gists: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None
    column_dims: Optional[List[int]] = None
    projected_bdi_states: Optional[Dict[int, List[BDIState]]] = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        TASK HEADS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class ClassificationHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class ToMPredictionHead(nn.Module):
    """Head specialised for Theory-of-Mind prediction tasks.

    Outputs a BDI prediction for what another agent is thinking,
    using the network's own mentalizing columns.

    Suitable for tasks like:
    - Sally-Anne false-belief tasks
    - Predicting an opponent's next move from their perspective
    - Inferring hidden goals from observed behaviour
    """

    def __init__(self, dim: int, factor_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, factor_dim * 3),
        )
        self.factor_dim = factor_dim

    def forward(self, x: Tensor) -> BDIState:
        raw = self.proj(x)
        return BDIState.unpack(raw, self.factor_dim)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              COMPLETE FracToM ARCHITECTURE                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FracToMNet(nn.Module):
    """Fractal Theory-of-Mind Network — the complete architecture.

    Combines all sub-modules into a single differentiable forward pass:

        Input → MentalStateEncoder
              → StructuralCausalModel (BDI causal refinement)
              → K+1 FractalMentalizingColumns (fractal, self-similar)
              → Causal Hierarchy Routing (association/intervention/counterfactual)
              → EpistemicGating per column
              → Cross-Depth Causal Discovery
              → Attention-weighted MentalizingJoin
              → BeliefRevisionModule
              → Task Head

    The architecture is **interpretable by construction**: every internal
    tensor has a clear cognitive-science interpretation (BDI states,
    mentalizing depth weights, epistemic uncertainty, causal graph
    structure, Pearl hierarchy level weights), and these can be
    inspected via the ``InterpretabilityReport``.

    Causal Integration
    ------------------
    The Structural Causal Model (SCM) provides three capabilities:

    1. **Causal Discovery** — the BDI causal graph and cross-depth
       causal structure are *learned* end-to-end via differentiable
       structure learning (NOTEARS; Zheng et al., 2018).

    2. **Pearl's Causal Hierarchy** — mentalizing columns are routed
       through three causal reasoning levels:
       - Depth 0 → Association (seeing): P(Y|X)
       - Depth 1 → Intervention (doing): P(Y|do(X))
       - Depth 2+ → Counterfactual (imagining): P(Y_{x'}|X,Y)

    3. **Counterfactual ToM** — the network can answer questions like
       "If the agent *had* seen X, what would they believe?" via
       abduction-action-prediction (Pearl, 2009).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw observation vector.
    hidden_dim : int
        Internal representation dimensionality.  All columns, attention
        layers, and feed-forward layers use this width.
    mentalizing_depth : int
        Maximum order of mentalizing (number of columns = depth + 1).
        Depth 0 = no mentalizing, 1 = metacognition / basic other-model,
        2 = second-order ToM, etc.
    num_bdi_factors : int
        Currently fixed at 3 (Belief, Desire, Intention).  Reserved for
        future extension to richer mental-state ontologies.
    blocks_per_column : int
        Number of SelfSimilarBlocks stacked within each column.
    num_heads : int
        Attention heads for self- and cross-attention.
    ff_mult : int
        Feed-forward hidden-to-dim ratio.
    dropout : float
        Dropout probability throughout.
    drop_path : float
        Base column drop-path probability (see FractalDropPath).
    num_classes : int | None
        If set, a ClassificationHead is appended.
    causal_model : bool
        If True (default), enable the Structural Causal Model for
        BDI causal reasoning, Pearl hierarchy routing, counterfactual
        generation, and cross-depth causal discovery.
    causal_noise_dim : int
        Dimensionality of exogenous noise in structural equations.
    capacity_schedule : str
        Per-depth capacity scaling strategy (FractalGen-inspired).
        ``"uniform"`` — all columns use the same hidden_dim (default,
        backward-compatible).
        ``"decreasing"`` — deeper columns use progressively smaller
        hidden_dim, reflecting Representational Redescription theory
        (Karmiloff-Smith, 1992): higher-order mentalizing produces
        *compressed* re-descriptions of lower-order representations.
        Concrete schedule: dim_k = hidden_dim × (1 − 0.5 × k/K).
    guiding_belief : bool
        If True, enable the GuidingBeliefModule that injects a coarse
        gist belief into each column via FiLM modulation before
        fine-grained mentalizing (analogous to FractalGen's "guiding
        pixel"; Li et al., 2025).
    gist_dim : int
        Dimensionality of the guiding gist belief vector.
    auxiliary_heads : bool
        If True, attach a lightweight per-column auxiliary classification
        head for deep supervision (FractalGen-inspired: every fractal
        level generates its own loss signal, preventing dead columns).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        mentalizing_depth: int = 3,
        num_bdi_factors: int = 3,
        blocks_per_column: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.15,
        num_classes: Optional[int] = None,
        causal_model: bool = True,
        causal_noise_dim: int = 16,
        capacity_schedule: str = "uniform",
        guiding_belief: bool = True,
        gist_dim: int = 64,
        auxiliary_heads: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mentalizing_depth = mentalizing_depth
        self.factor_dim = hidden_dim // num_bdi_factors
        self.capacity_schedule = capacity_schedule
        self.use_guiding_belief = guiding_belief
        self.use_auxiliary_heads = auxiliary_heads and (num_classes is not None)
        assert hidden_dim % num_bdi_factors == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"num_bdi_factors ({num_bdi_factors})"
        )

        # --- Per-depth capacity schedule (FractalGen-inspired) ---
        # Compute per-column hidden dimensions
        K = mentalizing_depth
        # Quantum: column dims must be divisible by both num_bdi_factors
        # and num_heads to satisfy MultiheadAttention constraints.
        _quantum = num_bdi_factors * num_heads // math.gcd(num_bdi_factors, num_heads)
        if capacity_schedule == "decreasing":
            # dim_k = hidden_dim × (1 − 0.5 × k/K), quantised to _quantum
            self.column_dims = []
            for k in range(K + 1):
                ratio = 1.0 - 0.5 * (k / max(K, 1))
                raw = int(hidden_dim * ratio)
                # round down to nearest multiple of _quantum
                raw = max(raw - raw % _quantum, _quantum)
                self.column_dims.append(raw)
        else:  # "uniform"
            self.column_dims = [hidden_dim] * (K + 1)

        # --- Observation encoder → BDI ---
        self.encoder = MentalStateEncoder(
            input_dim, self.factor_dim, dropout,
        )
        # project packed BDI (3 × factor_dim) → hidden_dim
        self.input_proj = nn.Linear(self.factor_dim * 3, hidden_dim)

        # --- Per-column input projectors (for capacity scheduling) ---
        # When column dims differ from hidden_dim, project h_input → col_dim
        self.col_input_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.column_dims[k])
            if self.column_dims[k] != hidden_dim else nn.Identity()
            for k in range(K + 1)
        ])
        # And project column output → hidden_dim for join/causal
        self.col_output_projs = nn.ModuleList([
            nn.Linear(self.column_dims[k], hidden_dim)
            if self.column_dims[k] != hidden_dim else nn.Identity()
            for k in range(K + 1)
        ])

        # --- Fractal mentalizing columns ---
        # Each column_dim is a multiple of _quantum (= lcm(num_bdi_factors,
        # num_heads)), so column_dim is always divisible by num_heads.
        self.columns = nn.ModuleList([
            FractalMentalizingColumn(
                depth_index=k,
                dim=self.column_dims[k],
                factor_dim=self.column_dims[k] // num_bdi_factors,
                num_blocks=blocks_per_column,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for k in range(K + 1)
        ])

        # --- Epistemic gates (one per column) ---
        self.epistemic_gates = nn.ModuleList([
            EpistemicGate(self.column_dims[k])
            for k in range(K + 1)
        ])

        # --- Guiding Belief Module (FractalGen-inspired) ---
        self.guiding_beliefs: Optional[nn.ModuleList] = None
        if guiding_belief:
            self.guiding_beliefs = nn.ModuleList([
                GuidingBeliefModule(
                    input_dim=hidden_dim,
                    gist_dim=gist_dim,
                    output_dim=self.column_dims[k],
                )
                for k in range(K + 1)
            ])

        # --- Auxiliary classification heads (deep supervision) ---
        self.aux_heads: Optional[nn.ModuleList] = None
        if self.use_auxiliary_heads and num_classes is not None:
            self.aux_heads = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes)
                for _ in range(K + 1)
            ])

        # --- Structural Causal Model (SCM) ---
        self.use_causal = causal_model
        if causal_model:
            self.scm = StructuralCausalModel(
                factor_dim=self.factor_dim,
                noise_dim=causal_noise_dim,
                dropout=dropout,
            )
            self.causal_router = CausalHierarchyRouter(
                dim=hidden_dim,
                max_depth=mentalizing_depth,
            )
            self.causal_discovery = CausalDiscoveryModule(
                factor_dim=self.factor_dim,
                max_depth=mentalizing_depth,
                dropout=dropout,
            )
            # Project hidden_dim → factor_dim for SCM observation input
            self.obs_to_factor = nn.Linear(hidden_dim, self.factor_dim)
            # Project causal BDI back to hidden_dim for column enrichment
            self.causal_to_hidden = nn.Linear(
                self.factor_dim * 3, hidden_dim,
            )
            # Learned counterfactual observation transform
            self.cf_obs_transform = nn.Linear(
                self.factor_dim, self.factor_dim, bias=False,
            )
            # Per-column BDI projectors: project column's native factor_dim
            # → SCM factor_dim (needed when capacity_schedule != "uniform")
            self.bdi_to_scm_projs = nn.ModuleList([
                nn.Linear(self.column_dims[k] // num_bdi_factors, self.factor_dim)
                if self.column_dims[k] // num_bdi_factors != self.factor_dim
                else nn.Identity()
                for k in range(K + 1)
            ])

        # --- Drop-path ---
        self.drop_path = FractalDropPath(
            drop_prob=drop_path,
            max_depth=mentalizing_depth,
            developmental=True,
        )

        # --- Column join ---
        self.join = MentalizingJoin(hidden_dim, mentalizing_depth)

        # --- Belief revision ---
        self.belief_revision = BeliefRevisionModule(hidden_dim, dropout)

        # --- Task head ---
        self.task_head: Optional[nn.Module] = None
        if num_classes is not None:
            self.task_head = ClassificationHead(
                hidden_dim, num_classes, dropout,
            )

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, ones/zeros for LayerNorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        x: Tensor,
        return_interpretability: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, InterpretabilityReport]]:
        """
        Parameters
        ----------
        x : (batch, input_dim)
        return_interpretability : bool
            If True, also return an InterpretabilityReport.

        Returns
        -------
        logits : (batch, num_classes) if task_head is set, else (batch, hidden_dim).
        report : InterpretabilityReport (only if requested).
        """
        B = x.shape[0]

        # 1) Encode observation → initial BDI
        bdi_init, sigma_init = self.encoder(x)
        h_input = self.input_proj(bdi_init.pack())  # (B, D)

        # 2) Run fractal columns (depth 0 … K)
        column_outputs: List[Tensor] = []     # always hidden_dim
        all_bdis: Dict[int, List[BDIState]] = {}
        all_sigmas: Dict[int, Tensor] = {}
        guiding_gists: Dict[int, Tuple[Tensor, Tensor]] = {}
        aux_logits: Dict[int, Tensor] = {}
        projected_bdis: Dict[int, List[BDIState]] = {}

        for k, (col, egate) in enumerate(
            zip(self.columns, self.epistemic_gates)
        ):
            # Project input to column's native dim
            h_col_in = self.col_input_projs[k](h_input)

            # Guiding belief: FiLM conditioning on coarse gist
            if self.guiding_beliefs is not None:
                gamma, beta = self.guiding_beliefs[k](h_input)
                h_col_in = GuidingBeliefModule.modulate(h_col_in, gamma, beta)
                guiding_gists[k] = (gamma.detach(), beta.detach())

            # Build lower-level outputs in column's native dim for cross-attn
            # Strategy: take each prior column_output (hidden_dim) and project
            # to the current column's dim via col_input_projs[k].
            lower: Optional[List[Tensor]] = None
            if k > 0:
                lower = [self.col_input_projs[k](column_outputs[j]) for j in range(k)]

            h_col, bdis_col = col(h_col_in, lower)

            # epistemic gating (in native col dim)
            h_col, sigma_k = egate(h_col)
            all_sigmas[k] = sigma_k

            # drop-path (training only, in native col dim)
            h_col = self.drop_path(h_col, column_index=k)

            # Project to hidden_dim for join & causal modules
            h_col_out = self.col_output_projs[k](h_col)
            column_outputs.append(h_col_out)
            all_bdis[k] = bdis_col

            # Store projected BDI states (common factor_dim) for loss
            _proj_k = self.bdi_to_scm_projs[k]
            projected_bdis[k] = [
                BDIState(
                    _proj_k(b.belief), _proj_k(b.desire), _proj_k(b.intention),
                )
                for b in bdis_col
            ]

            # Auxiliary deep supervision head
            if self.aux_heads is not None:
                aux_logits[k] = self.aux_heads[k](h_col_out)

        # 2.5) Causal processing via Structural Causal Model
        causal_info: Dict = {}
        if self.use_causal:
            obs_factor = self.obs_to_factor(h_input)  # (B, factor_dim)
            causal_hierarchy_weights: Dict[int, Tensor] = {}
            cf_distances: Dict[int, float] = {}
            causal_adj: Optional[Tensor] = None
            dag_pen_total = torch.tensor(0.0, device=x.device)

            for k in range(len(self.columns)):
                # Get this column's last BDI
                bdi_k = all_bdis[k][-1]

                # Project column-native BDI → SCM factor_dim
                _proj = self.bdi_to_scm_projs[k]
                bdi_k_scm = BDIState(
                    _proj(bdi_k.belief),
                    _proj(bdi_k.desire),
                    _proj(bdi_k.intention),
                )

                # Route through Pearl's causal hierarchy
                level_weights = self.causal_router(
                    column_outputs[k], depth_index=k,
                )  # (B, 3)
                causal_hierarchy_weights[k] = level_weights

                # Level 1: Association (standard SCM forward)
                bdi_assoc, scm_info = self.scm(obs_factor, bdi_k_scm)
                causal_adj = scm_info["adjacency"]
                dag_pen_total = dag_pen_total + scm_info["dag_penalty"]

                # Level 2: Intervention — do(Belief = belief_k)
                bdi_interv = self.scm.intervene(
                    obs_factor, bdi_k_scm,
                    target=StructuralCausalModel.VAR_BELIEF,
                    value=bdi_k_scm.belief,
                )

                # Level 3: Counterfactual — "what if obs were different?"
                cf_obs = self.cf_obs_transform(obs_factor)
                bdi_cf = self.scm.counterfactual(
                    obs_factor, bdi_k_scm, cf_obs,
                )

                # Counterfactual distance (interpretability metric)
                with torch.no_grad():
                    cf_dist_val = (
                        bdi_cf.pack() - bdi_k_scm.pack()
                    ).norm(dim=-1).mean().item()
                cf_distances[k] = cf_dist_val

                # Blend all three levels weighted by router
                packed_assoc = bdi_assoc.pack()
                packed_interv = bdi_interv.pack()
                packed_cf = bdi_cf.pack()
                w = level_weights  # (B, 3)
                blended_bdi = (
                    w[:, 0:1] * packed_assoc
                    + w[:, 1:2] * packed_interv
                    + w[:, 2:3] * packed_cf
                )  # (B, 3 * factor_dim)

                # Enrich column output with causal BDI information
                causal_h = self.causal_to_hidden(blended_bdi)
                column_outputs[k] = column_outputs[k] + causal_h

            # Cross-depth causal discovery (project BDIs to SCM factor_dim)
            last_bdis: Dict[int, BDIState] = {}
            for k in all_bdis:
                _b = all_bdis[k][-1]
                _p = self.bdi_to_scm_projs[k]
                last_bdis[k] = BDIState(
                    _p(_b.belief), _p(_b.desire), _p(_b.intention),
                )
            cross_adj, cross_pen = self.causal_discovery.discover(
                last_bdis, obs_factor,
            )
            dag_pen_total = dag_pen_total + cross_pen

            causal_info = {
                "causal_adjacency": causal_adj,
                "causal_hierarchy_weights": causal_hierarchy_weights,
                "cross_depth_adjacency": cross_adj,
                "dag_penalty": dag_pen_total,
                "counterfactual_distances": cf_distances,
            }

        # 3) Attention-weighted join across columns
        joined, alpha = self.join(column_outputs, h_input)

        # 4) Bayesian belief revision
        posterior = self.belief_revision(h_input, joined)

        # 5) Task head
        out = self.task_head(posterior) if self.task_head else posterior

        if return_interpretability:
            report = InterpretabilityReport(
                depth_weights=alpha,
                bdi_states=all_bdis,
                column_uncertainties=all_sigmas,
                causal_adjacency=causal_info.get("causal_adjacency"),
                causal_hierarchy_weights=causal_info.get(
                    "causal_hierarchy_weights"
                ),
                cross_depth_adjacency=causal_info.get(
                    "cross_depth_adjacency"
                ),
                dag_penalty=causal_info.get("dag_penalty"),
                counterfactual_distances=causal_info.get(
                    "counterfactual_distances"
                ),
                auxiliary_logits=aux_logits if aux_logits else None,
                guiding_gists=guiding_gists if guiding_gists else None,
                column_dims=self.column_dims,
                projected_bdi_states=projected_bdis if projected_bdis else None,
            )
            return out, report
        return out

    # ------------------------------------------------------------------ util
    def set_curriculum(self, factor: float) -> None:
        """Anneal developmental drop-path.  factor: 1.0 → 0.0 over training."""
        self.drop_path.set_curriculum(factor)

    def get_tom_head(
        self, factor_dim: Optional[int] = None, dropout: float = 0.1,
    ) -> ToMPredictionHead:
        """Convenience: build a ToMPredictionHead matching this network."""
        fd = factor_dim or self.factor_dim
        return ToMPredictionHead(self.hidden_dim, fd, dropout)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              LOSS FUNCTIONS & TRAINING UTILITIES                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FracToMLoss(nn.Module):
    """Composite loss for the FracToM architecture.

    Combines:
    1. **Task loss** — standard cross-entropy (classification) or MSE.
    2. **BDI consistency loss** — encourages adjacent mentalizing depths
       to produce *related* BDI states (they shouldn't diverge wildly).
    3. **Uncertainty calibration loss** — penalises overconfident
       predictions on incorrect examples (ECE-inspired).
    4. **Depth regularisation** — entropy bonus on α to encourage
       exploring multiple mentalizing depths.
    5. **DAG acyclicity penalty** — NOTEARS constraint ensuring the
       learned BDI causal graph is a valid DAG.
    6. **Causal sparsity** — encourages sparse causal graphs (Occam's
       razor for causal structure).
    7. **Counterfactual ordering** — deeper mentalizing columns should
       exhibit larger counterfactual distances (richer counterfactual
       reasoning at higher ToM depths).
    8. **Auxiliary deep supervision** — per-column classification loss
       inspired by FractalGen (Li et al., 2025): every fractal level
       produces its own task prediction, preventing gradient starvation
       and dead columns.

    λ coefficients control the balance.
    """

    def __init__(
        self,
        task_loss_fn: Optional[nn.Module] = None,
        lambda_bdi: float = 0.01,
        lambda_uncertainty: float = 0.005,
        lambda_depth_entropy: float = 0.01,
        lambda_dag: float = 0.1,
        lambda_causal_sparsity: float = 0.005,
        lambda_counterfactual: float = 0.01,
        lambda_auxiliary: float = 0.1,
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        self.lambda_bdi = lambda_bdi
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_depth_entropy = lambda_depth_entropy
        self.lambda_dag = lambda_dag
        self.lambda_causal_sparsity = lambda_causal_sparsity
        self.lambda_counterfactual = lambda_counterfactual
        self.lambda_auxiliary = lambda_auxiliary

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        report: InterpretabilityReport,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Returns
        -------
        loss : scalar Tensor
        breakdown : dict of individual loss components (for logging).
        """
        # 1) task loss
        task = self.task_loss_fn(logits, targets)

        # 2) BDI consistency: cosine distance between adjacent depths
        bdi_loss = torch.tensor(0.0, device=logits.device)
        # Use projected BDI states (common factor_dim) if available,
        # so cross-column cosine similarity is well-defined.
        bdi_src = report.projected_bdi_states or report.bdi_states
        depths = sorted(bdi_src.keys())
        for i in range(len(depths) - 1):
            bdis_a = bdi_src[depths[i]]
            bdis_b = bdi_src[depths[i + 1]]
            # compare last block's BDI in each column
            a = bdis_a[-1].pack()
            b = bdis_b[-1].pack()
            cos = F.cosine_similarity(a, b, dim=-1).mean()
            bdi_loss = bdi_loss + (1.0 - cos)
        if len(depths) > 1:
            bdi_loss = bdi_loss / (len(depths) - 1)

        # 3) uncertainty calibration: higher uncertainty on wrong predictions
        unc_loss = torch.tensor(0.0, device=logits.device)
        if logits.ndim == 2:  # classification
            preds = logits.argmax(-1)
            correct = (preds == targets).float()
            for sigma in report.column_uncertainties.values():
                s = sigma.squeeze(-1)
                # want: low σ when correct, high σ when wrong
                unc_loss = unc_loss + (correct * s - (1 - correct) * s).mean()
            unc_loss = unc_loss / max(len(report.column_uncertainties), 1)

        # 4) depth entropy bonus (maximise entropy → explore all depths)
        alpha = report.depth_weights  # (B, K+1)
        depth_ent = -(alpha * (alpha + 1e-8).log()).sum(-1).mean()
        max_ent = math.log(alpha.shape[-1])
        depth_reg = max_ent - depth_ent  # minimise this → maximise entropy

        loss = (
            task
            + self.lambda_bdi * bdi_loss
            + self.lambda_uncertainty * unc_loss
            + self.lambda_depth_entropy * depth_reg
        )

        # 5) DAG acyclicity penalty (NOTEARS)
        dag_loss = torch.tensor(0.0, device=logits.device)
        if report.dag_penalty is not None:
            dag_loss = report.dag_penalty
        loss = loss + self.lambda_dag * dag_loss

        # 6) Causal graph sparsity — encourage Occam's razor
        causal_sparse = torch.tensor(0.0, device=logits.device)
        if report.causal_adjacency is not None:
            causal_sparse = report.causal_adjacency.sum()
        if report.cross_depth_adjacency is not None:
            causal_sparse = causal_sparse + report.cross_depth_adjacency.sum()
        loss = loss + self.lambda_causal_sparsity * causal_sparse

        # 7) Counterfactual ordering — deeper columns should have larger
        #    counterfactual distances (they model further-removed scenarios)
        cf_loss = torch.tensor(0.0, device=logits.device)
        if report.counterfactual_distances:
            dists = report.counterfactual_distances
            depths = sorted(dists.keys())
            for i in range(len(depths) - 1):
                # penalise when shallower column has LARGER cf distance
                diff = dists[depths[i]] - dists[depths[i + 1]]
                if diff > 0:
                    cf_loss = cf_loss + diff
            if len(depths) > 1:
                cf_loss = cf_loss / (len(depths) - 1)
        loss = loss + self.lambda_counterfactual * cf_loss

        # 8) Auxiliary deep supervision — per-column classification loss
        #    (FractalGen-inspired: every fractal level generates its own
        #    loss, preventing dead columns & gradient starvation)
        aux_loss = torch.tensor(0.0, device=logits.device)
        if report.auxiliary_logits:
            n_aux = 0
            for k, aux_log in report.auxiliary_logits.items():
                aux_loss = aux_loss + F.cross_entropy(aux_log, targets)
                n_aux += 1
            if n_aux > 0:
                aux_loss = aux_loss / n_aux
        loss = loss + self.lambda_auxiliary * aux_loss

        breakdown = {
            "task": task.item(),
            "bdi_consistency": bdi_loss.item(),
            "uncertainty_cal": unc_loss.item(),
            "depth_entropy_reg": depth_reg.item(),
            "dag_penalty": dag_loss.item() if isinstance(dag_loss, Tensor) else dag_loss,
            "causal_sparsity": causal_sparse.item() if isinstance(causal_sparse, Tensor) else causal_sparse,
            "cf_ordering": cf_loss.item() if isinstance(cf_loss, Tensor) else cf_loss,
            "aux_deepsup": aux_loss.item() if isinstance(aux_loss, Tensor) else aux_loss,
            "total": loss.item(),
        }
        return loss, breakdown


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               INTERPRETABILITY ANALYSIS UTILITIES                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


def analyse_mentalizing_depth(
    report: InterpretabilityReport,
    class_names: Optional[List[str]] = None,
) -> str:
    """Human-readable analysis of which mentalizing depth was used.

    Returns a formatted string describing per-sample depth allocation.
    """
    alpha = report.depth_weights  # (B, K+1)
    B, K1 = alpha.shape
    lines = [f"Mentalizing Depth Analysis ({B} samples, {K1} levels)"]
    lines.append("=" * 60)

    # per-level average weight
    mean_alpha = alpha.mean(0)
    for k in range(K1):
        label = _depth_label(k)
        bar = "█" * int(mean_alpha[k].item() * 40)
        lines.append(f"  Level {k} ({label:20s}): {mean_alpha[k]:.3f}  {bar}")
    lines.append("")

    # dominant level distribution
    dominant = alpha.argmax(-1)  # (B,)
    lines.append("Dominant mentalizing level per sample:")
    for k in range(K1):
        count = (dominant == k).sum().item()
        lines.append(f"  Level {k}: {count}/{B} samples ({100*count/B:.1f}%)")

    # epistemic uncertainty summary
    if report.column_uncertainties:
        lines.append("")
        lines.append("Epistemic uncertainty (σ) per column:")
        for k, sigma in sorted(report.column_uncertainties.items()):
            s = sigma.mean().item()
            lines.append(f"  Column {k}: σ = {s:.4f}")

    # --- Causal analysis ---
    if report.causal_adjacency is not None:
        lines.append("")
        lines.append("Learned BDI Causal Graph (adjacency matrix):")
        var_names = ["Obs", "Belief", "Desire", "Intention"]
        A = report.causal_adjacency.detach()
        header = "         " + "  ".join(f"{n:>9s}" for n in var_names)
        lines.append(header)
        for i, name in enumerate(var_names):
            vals = "  ".join(f"{A[i, j]:.4f}   " for j in range(A.shape[1]))
            lines.append(f"  {name:9s} {vals}")
        lines.append("")
        lines.append("  Key causal edges (strength > 0.5):")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0.5:
                    lines.append(
                        f"    {var_names[i]} → {var_names[j]}:  "
                        f"{A[i, j]:.3f}"
                    )

    if report.dag_penalty is not None:
        dag_val = report.dag_penalty
        if isinstance(dag_val, Tensor):
            dag_val = dag_val.item()
        lines.append(f"  DAG penalty (0 = perfect DAG): {dag_val:.6f}")

    if report.causal_hierarchy_weights:
        lines.append("")
        lines.append("Pearl's Causal Hierarchy — per-column level weights:")
        level_names = ["Association", "Intervention", "Counterfactual"]
        for k in sorted(report.causal_hierarchy_weights.keys()):
            w = report.causal_hierarchy_weights[k].mean(0)  # avg over batch
            parts = "  ".join(
                f"{level_names[i]}: {w[i]:.3f}" for i in range(3)
            )
            lines.append(f"  Column {k}: {parts}")

    if report.cross_depth_adjacency is not None:
        lines.append("")
        lines.append("Cross-Depth Causal Structure:")
        cd = report.cross_depth_adjacency.detach()
        K = cd.shape[0]
        for i in range(K):
            for j in range(K):
                if cd[i, j] > 0.3:
                    lines.append(
                        f"  Column {i} → Column {j}: {cd[i, j]:.3f}"
                    )

    if report.counterfactual_distances:
        lines.append("")
        lines.append("Counterfactual distances per column:")
        for k, d in sorted(report.counterfactual_distances.items()):
            bar = "█" * int(d * 10)
            lines.append(f"  Column {k}: {d:.4f}  {bar}")

    return "\n".join(lines)


def extract_bdi_activations(
    report: InterpretabilityReport,
) -> Dict[int, Dict[str, Tensor]]:
    """Extract BDI activations per column for downstream analysis.

    Returns dict[column_index → {"belief": Tensor, "desire": Tensor,
    "intention": Tensor}], where each tensor has shape (batch, factor_dim).
    """
    result = {}
    for k, bdis in report.bdi_states.items():
        last = bdis[-1]  # last block's BDI
        result[k] = {
            "belief": last.belief.detach(),
            "desire": last.desire.detach(),
            "intention": last.intention.detach(),
        }
    return result


def extract_causal_graph(
    report: InterpretabilityReport,
) -> Dict[str, object]:
    """Extract causal structure information for downstream analysis.

    Returns a dict with:
    - ``bdi_adjacency``: (4, 4) Tensor — BDI causal graph adjacency.
      Rows/cols: [Obs, Belief, Desire, Intention].
    - ``bdi_edges``: list of (source, target, weight) tuples for edges > 0.3.
    - ``cross_depth_adjacency``: (K+1, K+1) Tensor — cross-depth causal graph.
    - ``cross_depth_edges``: list of (src_col, tgt_col, weight) tuples.
    - ``hierarchy_weights``: dict[int → (3,) Tensor] — per-column causal level.
    - ``counterfactual_distances``: dict[int → float].
    - ``dag_penalty``: float — 0 means perfect DAG.
    """
    var_names = ["Obs", "Belief", "Desire", "Intention"]
    result: Dict[str, object] = {}

    if report.causal_adjacency is not None:
        A = report.causal_adjacency.detach()
        result["bdi_adjacency"] = A
        edges = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0.3:
                    edges.append((var_names[i], var_names[j], A[i, j].item()))
        result["bdi_edges"] = edges
    else:
        result["bdi_adjacency"] = None
        result["bdi_edges"] = []

    if report.cross_depth_adjacency is not None:
        cd = report.cross_depth_adjacency.detach()
        result["cross_depth_adjacency"] = cd
        edges = []
        for i in range(cd.shape[0]):
            for j in range(cd.shape[1]):
                if cd[i, j] > 0.3:
                    edges.append((i, j, cd[i, j].item()))
        result["cross_depth_edges"] = edges
    else:
        result["cross_depth_adjacency"] = None
        result["cross_depth_edges"] = []

    if report.causal_hierarchy_weights is not None:
        result["hierarchy_weights"] = {
            k: v.mean(0).detach()
            for k, v in report.causal_hierarchy_weights.items()
        }
    else:
        result["hierarchy_weights"] = {}

    result["counterfactual_distances"] = report.counterfactual_distances or {}

    if report.dag_penalty is not None:
        dag_val = report.dag_penalty
        result["dag_penalty"] = dag_val.item() if isinstance(dag_val, Tensor) else dag_val
    else:
        result["dag_penalty"] = None

    return result


def _depth_label(k: int) -> str:
    labels = {
        0: "Direct (no ToM)",
        1: "Metacognition",
        2: "Basic other-model",
        3: "2nd-order ToM",
        4: "3rd-order ToM",
    }
    return labels.get(k, f"{k}th-order ToM")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                SEQUENCE-INPUT WRAPPER                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SequenceFracToM(nn.Module):
    """Wraps FracToMNet for variable-length sequence inputs.

    Adds a shared Transformer encoder before the FracToM backbone,
    pooling the sequence to a fixed-dimensional vector.  Useful for
    language-based ToM tasks (e.g., reading stories and predicting
    characters' beliefs).

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the embedding layer.
    embed_dim : int
        Token embedding dimensionality.
    seq_encoder_layers : int
        Number of Transformer encoder layers for sequence processing.
    **fractom_kwargs
        Forwarded to ``FracToMNet(input_dim=embed_dim, ...)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        max_seq_len: int = 512,
        seq_encoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        **fractom_kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=seq_encoder_layers,
        )
        self.pool_proj = nn.Linear(embed_dim, embed_dim)

        fractom_kwargs.setdefault("hidden_dim", embed_dim)
        self.fractom = FracToMNet(input_dim=embed_dim, **fractom_kwargs)

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_interpretability: bool = False,
    ):
        """
        Parameters
        ----------
        token_ids : (batch, seq_len)  LongTensor.
        attention_mask : (batch, seq_len)  BoolTensor, True = pad.
        """
        x = self.embedding(token_ids)
        x = self.pos_enc(x)
        x = self.seq_encoder(x, src_key_padding_mask=attention_mask)

        # mean-pool (masking out padding)
        if attention_mask is not None:
            mask_f = (~attention_mask).unsqueeze(-1).float()
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        pooled = self.pool_proj(pooled)
        return self.fractom(pooled, return_interpretability=return_interpretability)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      DEMO / SMOKE TEST                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


def demo_classification() -> None:
    """Demonstrates FracToM on a synthetic classification task.

    Trains on a small synthetic dataset where the correct prediction
    *requires* modelling another agent's perspective (a simplified
    Sally-Anne scenario encoded as vectors).
    """
    import random

    print("=" * 70)
    print("FracToM Demo — Synthetic Theory-of-Mind Classification")
    print("=" * 70)

    torch.manual_seed(42)
    random.seed(42)

    # ---- Synthetic "false-belief" dataset ----
    # Task: predict where Sally will look for her ball.
    # Input: 128-dim vector encoding (ball_true_loc, sally_saw_move,
    #        anne_moved_ball, ...).  Label: 0 = original loc, 1 = new loc.
    # A network *without* ToM would predict the true location (label 1).
    # A network *with* ToM would predict Sally's *believed* location
    # (label 0) when she didn't see the move.
    N_TRAIN, N_TEST = 2000, 400
    INPUT_DIM = 128
    NUM_CLASSES = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def make_data(n: int) -> Tuple[Tensor, Tensor]:
        xs, ys = [], []
        for _ in range(n):
            x = torch.randn(INPUT_DIM)
            sally_saw = x[0].item() > 0        # did Sally see the move?
            ball_moved = x[1].item() > 0        # was ball actually moved?
            if ball_moved and not sally_saw:
                y = 0  # Sally falsely believes ball is in original place
            else:
                y = 1  # Sally knows the true location
            # bias some features to make it learnable
            x[2] = float(sally_saw) + 0.1 * torch.randn(1).item()
            x[3] = float(ball_moved) + 0.1 * torch.randn(1).item()
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    X_train, y_train = make_data(N_TRAIN)
    X_test, y_test = make_data(N_TEST)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    # ---- Build model ----
    model = FracToMNet(
        input_dim=INPUT_DIM,
        hidden_dim=96,         # small for demo speed
        mentalizing_depth=3,   # 4 columns (depths 0–3)
        num_bdi_factors=3,
        blocks_per_column=1,
        num_heads=4,
        ff_mult=2,
        dropout=0.1,
        drop_path=0.1,
        num_classes=NUM_CLASSES,
        causal_model=True,     # enable SCM integration
        causal_noise_dim=16,
        # --- FractalGen-inspired enhancements ---
        capacity_schedule="decreasing",  # per-depth capacity scaling
        guiding_belief=True,             # FiLM gist conditioning
        gist_dim=32,                     # small for demo
        auxiliary_heads=True,            # deep supervision
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Device: {DEVICE}\n")

    criterion = FracToMLoss(
        task_loss_fn=nn.CrossEntropyLoss(),
        lambda_bdi=0.01,
        lambda_uncertainty=0.005,
        lambda_depth_entropy=0.01,
        lambda_dag=0.1,
        lambda_causal_sparsity=0.005,
        lambda_counterfactual=0.01,
        lambda_auxiliary=0.1,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100)

    # ---- Training loop ----
    EPOCHS = 100
    BATCH = 64
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # developmental curriculum: linearly anneal from 1 → 0
        model.set_curriculum(1.0 - epoch / EPOCHS)

        perm = torch.randperm(N_TRAIN, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i : i + BATCH]
            xb, yb = X_train[idx], y_train[idx]

            logits, report = model(xb, return_interpretability=True)
            loss, _ = criterion(logits, yb, report)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()

        # ---- Evaluation ----
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits_test, report_test = model(
                    X_test, return_interpretability=True,
                )
                preds = logits_test.argmax(-1)
                acc = (preds == y_test).float().mean().item()
                best_acc = max(best_acc, acc)

                avg_loss = epoch_loss / (N_TRAIN / BATCH)
                print(
                    f"Epoch {epoch:3d} │ loss {avg_loss:.4f} │ "
                    f"test acc {acc:.3f} │ best {best_acc:.3f}"
                )

    # ---- Interpretability analysis ----
    model.eval()
    with torch.no_grad():
        _, report_final = model(X_test, return_interpretability=True)
    print()
    print(analyse_mentalizing_depth(report_final))
    print()

    # Show BDI activations
    bdi_acts = extract_bdi_activations(report_final)
    print("BDI activation norms per column:")
    for k, acts in bdi_acts.items():
        b_norm = acts["belief"].norm(dim=-1).mean().item()
        d_norm = acts["desire"].norm(dim=-1).mean().item()
        i_norm = acts["intention"].norm(dim=-1).mean().item()
        print(
            f"  Column {k} ({_depth_label(k):20s}): "
            f"‖B‖={b_norm:.3f}  ‖D‖={d_norm:.3f}  ‖I‖={i_norm:.3f}"
        )

    # Show causal graph analysis
    causal_graph = extract_causal_graph(report_final)
    if causal_graph["bdi_edges"]:
        print("\nDiscovered BDI causal edges:")
        for src, tgt, w in causal_graph["bdi_edges"]:
            print(f"  {src} → {tgt}:  {w:.3f}")
    if causal_graph["cross_depth_edges"]:
        print("\nDiscovered cross-depth causal edges:")
        for src, tgt, w in causal_graph["cross_depth_edges"]:
            print(f"  Column {src} → Column {tgt}:  {w:.3f}")
    if causal_graph["hierarchy_weights"]:
        print("\nPearl's Causal Hierarchy weights per column:")
        level_names = ["Assoc", "Interv", "CF"]
        for k, w in sorted(causal_graph["hierarchy_weights"].items()):
            parts = "  ".join(f"{level_names[i]}: {w[i]:.3f}" for i in range(3))
            print(f"  Column {k}: {parts}")

    # Show FractalGen-inspired enhancements
    if report_final.column_dims is not None:
        print("\nPer-column capacity (FractalGen-inspired scheduling):")
        for k, d in enumerate(report_final.column_dims):
            print(f"  Column {k} ({_depth_label(k):20s}): dim={d}")

    if report_final.auxiliary_logits is not None:
        print("\nAuxiliary head accuracy (deep supervision):")
        for k, aux_log in sorted(report_final.auxiliary_logits.items()):
            aux_pred = aux_log.argmax(-1)
            aux_acc = (aux_pred == y_test).float().mean().item()
            print(f"  Column {k}: {aux_acc:.3f}")

    if report_final.guiding_gists is not None:
        print("\nGuiding belief gist stats (γ, β norms):")
        for k, (g, b) in sorted(report_final.guiding_gists.items()):
            print(f"  Column {k}: ‖γ‖={g.norm(dim=-1).mean():.4f}  "
                  f"‖β‖={b.norm(dim=-1).mean():.4f}")

    print(f"\n{'=' * 70}")
    print(f"Final test accuracy: {best_acc:.3f}")
    print(f"{'=' * 70}")


def demo_sequence_tom() -> None:
    """Demonstrates SequenceFracToM on a toy token-sequence task.

    Verifies that the sequence wrapper builds, runs forward pass, and
    backward pass without error.
    """
    print("\n" + "=" * 70)
    print("SequenceFracToM — Forward/Backward Smoke Test")
    print("=" * 70)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = SequenceFracToM(
        vocab_size=1000,
        embed_dim=96,
        max_seq_len=64,
        seq_encoder_layers=2,
        num_heads=4,
        mentalizing_depth=2,
        blocks_per_column=1,
        num_classes=5,
        dropout=0.1,
        drop_path=0.1,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    tokens = torch.randint(0, 1000, (8, 32), device=DEVICE)
    mask = torch.zeros(8, 32, dtype=torch.bool, device=DEVICE)
    mask[:, 25:] = True  # last 7 tokens are padding

    logits, report = model(tokens, mask, return_interpretability=True)
    print(f"Output shape: {logits.shape}")
    print(f"Depth weights shape: {report.depth_weights.shape}")
    print(f"Depth weights (sample 0): {report.depth_weights[0].tolist()}")

    loss = logits.sum()
    loss.backward()
    print("Backward pass: OK")
    print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         ENTRY POINT                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


if __name__ == "__main__":
    demo_classification()
    demo_sequence_tom()
