"""
FracToM — Fractal Theory-of-Mind Neural Network  (MLX Port)
============================================================

Apple-Silicon-native reimplementation of the FracToM architecture using
the `MLX <https://ml-explore.github.io/mlx>`_ framework for accelerated
pretraining, finetuning, and inference on Metal GPUs with unified memory.

Every module, hyperparameter, and architectural decision mirrors the
PyTorch reference ``nn.py``; only the backend differs.  Consult the
original for detailed docstrings and mathematical background.

Key MLX-Specific Adaptations
-----------------------------
* ``torch.matrix_exp`` → Taylor-series approximation (order 8) via
  :func:`_matrix_exp_taylor`.  MLX's ``linalg`` does not expose a
  matrix exponential; the approximation is accurate for the small
  element-wise-squared adjacency matrices used by NOTEARS.
* ``nn.MultiHeadAttention`` — MLX returns output only (no attention
  weights); dropout is applied after the projection instead of on
  attention scores.
* ``register_buffer`` → private attributes (``self._name``) so that
  MLX's parameter auto-discovery skips them.
* ``nn.TransformerEncoder`` → custom stack of pre-norm Transformer
  encoder blocks.
* Training pattern: ``nn.value_and_grad`` + ``optimizer.update`` +
  ``mx.eval`` (lazy evaluation model).

Requirements
------------
    Python ≥ 3.9,  mlx ≥ 0.20

Usage
-----
    import mlx.core as mx
    from mlx_nn import FracToMNet

    model = FracToMNet(input_dim=128, hidden_dim=96, mentalizing_depth=3,
                       num_classes=2, causal_model=True)
    x = mx.random.normal((32, 128))
    out, report = model(x, return_interpretability=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     UTILITY FUNCTIONS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


def _matrix_exp_taylor(M: mx.array, order: int = 8) -> mx.array:
    """Approximate matrix exponential via truncated Taylor series.

    exp(M) ≈ I + M + M²/2! + … + M^n/n!

    Sufficient for the small (4×4) element-wise-squared adjacency
    matrices used in the NOTEARS DAG penalty.
    """
    n = M.shape[0]
    result = mx.eye(n, dtype=M.dtype)
    term = mx.eye(n, dtype=M.dtype)
    for k in range(1, order + 1):
        term = (term @ M) / k
        result = result + term
    return result


def _trace(M: mx.array) -> mx.array:
    """Trace of a square matrix."""
    return mx.sum(mx.diag(M))


def _cosine_similarity(a: mx.array, b: mx.array, axis: int = -1) -> mx.array:
    """Cosine similarity along *axis*."""
    a_n = mx.sqrt(mx.sum(a * a, axis=axis, keepdims=True) + 1e-8)
    b_n = mx.sqrt(mx.sum(b * b, axis=axis, keepdims=True) + 1e-8)
    return mx.sum((a / a_n) * (b / b_n), axis=axis)


def _l2_norm(x: mx.array, axis: int = -1) -> mx.array:
    """L2 norm along *axis*."""
    return mx.sqrt(mx.sum(x * x, axis=axis))


def clip_grad_norm(grads, max_norm: float):
    """Clip gradient tree by global L2 norm (in-place-safe)."""
    from mlx.utils import tree_flatten, tree_map

    flat = tree_flatten(grads)
    total_sq = sum(mx.sum(g * g).item() for _, g in flat)
    total_norm = math.sqrt(total_sq)
    coeff = max_norm / (total_norm + 1e-6)
    if coeff < 1.0:
        grads = tree_map(lambda g: g * coeff, grads)
    return grads


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        UTILITY MODULES                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


class Identity(nn.Module):
    """Pass-through (replaces ``nn.Identity`` which MLX lacks)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class FractalDropPath(nn.Module):
    """Stochastic depth / drop-path for mentalizing columns.

    Supports a *developmental curriculum* where deeper columns start
    with higher drop probability and are gradually unlocked.
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
        # Non-trainable buffer (underscore prefix)
        self._curriculum_factor = mx.array(1.0)

    def set_curriculum(self, factor: float) -> None:
        """Anneal from 1.0 (beginning) → 0.0 (end of training)."""
        self._curriculum_factor = mx.array(max(0.0, min(1.0, factor)))

    def __call__(self, x: mx.array, column_index: int = 0) -> mx.array:
        if not self.training or self.drop_prob == 0.0:
            return x

        p = self.drop_prob
        if self.developmental:
            depth_ratio = column_index / self.max_depth
            p = p + (1.0 - p) * depth_ratio * self._curriculum_factor.item()
        p = min(p, 0.999)

        keep = 1.0 - p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = mx.random.bernoulli(p=keep, shape=shape).astype(x.dtype)
        return x * mask / keep


class GatedResidual(nn.Module):
    """Gated Residual Unit: output = gate ⊙ f(x) + (1 − gate) ⊙ x."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.gate_proj = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.fc1(h)
        g = mx.sigmoid(self.gate_proj(h))
        h = nn.gelu(h)
        h = self.drop(self.fc2(h))
        return g * h + (1.0 - g) * x


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     BDI MENTAL-STATE ENCODER                       ║
# ╚══════════════════════════════════════════════════════════════════════╝


class BDIState:
    """Structured container for a Belief-Desire-Intention triple.

    All three arrays share shape ``(batch, factor_dim)``.
    """

    __slots__ = ("belief", "desire", "intention")

    def __init__(self, belief: mx.array, desire: mx.array, intention: mx.array):
        self.belief = belief
        self.desire = desire
        self.intention = intention

    def pack(self) -> mx.array:
        """→ (batch, 3 × factor_dim)"""
        return mx.concatenate([self.belief, self.desire, self.intention], axis=-1)

    @staticmethod
    def unpack(x: mx.array, factor_dim: int) -> "BDIState":
        b = x[..., :factor_dim]
        d = x[..., factor_dim : 2 * factor_dim]
        i = x[..., 2 * factor_dim :]
        return BDIState(b, d, i)

    def detach(self) -> "BDIState":
        return BDIState(
            mx.stop_gradient(self.belief),
            mx.stop_gradient(self.desire),
            mx.stop_gradient(self.intention),
        )


class MentalStateEncoder(nn.Module):
    """Encodes raw observations into an initial BDI triple + uncertainty."""

    def __init__(self, input_dim: int, factor_dim: int, dropout: float = 0.0):
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
        self.proj_uncertainty = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> Tuple[BDIState, mx.array]:
        h = self.trunk(x)
        bdi = BDIState(
            belief=self.proj_belief(h),
            desire=self.proj_desire(h),
            intention=self.proj_intention(h),
        )
        sigma = nn.softplus(self.proj_uncertainty(h))  # σ > 0
        return bdi, sigma


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               SELF-SIMILAR MENTALIZING BLOCK (ψ)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class GEGLU(nn.Module):
    """Gated GELU activation (Shazeer, 2020)."""

    def __call__(self, x: mx.array) -> mx.array:
        a, gate = mx.split(x, 2, axis=-1)
        return a * nn.gelu(gate)


class SelfSimilarBlock(nn.Module):
    """The *fractal primitive*: one mentalizing transformation.

    LayerNorm → MHA → Residual → [optional CrossAttn] → GEGLU FF
    → Residual → BDI Re-Factoring.
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
        self.use_cross_attn = use_cross_attn

        # --- self-attention ---
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiHeadAttention(dim, num_heads, bias=True)
        self.sa_drop = nn.Dropout(dropout)

        # --- optional cross-attention ---
        if use_cross_attn:
            self.norm_ca = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiHeadAttention(dim, num_heads, bias=True)
            self.ca_drop = nn.Dropout(dropout)

        # --- feed-forward with GEGLU ---
        self.norm_ff = nn.LayerNorm(dim)
        inner = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, inner * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

        # --- BDI re-factoring ---
        self.bdi_proj = nn.Linear(dim, factor_dim * 3)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
    ) -> Tuple[mx.array, BDIState]:
        squeeze = x.ndim == 2
        if squeeze:
            x = mx.expand_dims(x, axis=1)  # (B, 1, D)

        # self-attention
        h = self.norm_sa(x)
        h_sa = self.self_attn(h, h, h)
        x = x + self.sa_drop(h_sa)

        # cross-attention (perspective shifting)
        if self.use_cross_attn and context is not None:
            h = self.norm_ca(x)
            ctx = mx.expand_dims(context, axis=1) if context.ndim == 2 else context
            h_ca = self.cross_attn(h, ctx, ctx)
            x = x + self.ca_drop(h_ca)

        # feed-forward
        x = x + self.ff(self.norm_ff(x))

        # BDI re-factoring: pool over seq → structured BDI
        bdi_raw = self.bdi_proj(mx.mean(x, axis=1))
        bdi = BDIState.unpack(bdi_raw, self.factor_dim)

        if squeeze:
            x = mx.squeeze(x, axis=1)
        return x, bdi


# ╔══════════════════════════════════════════════════════════════════════╗
# ║           PERSPECTIVE-SHIFTING CROSS-DEPTH ATTENTION               ║
# ╚══════════════════════════════════════════════════════════════════════╝


class PerspectiveShiftAttention(nn.Module):
    """Learned perspective transform + cross-depth attention.

    Models the cognitive operation of "putting yourself in another's
    shoes" before reading off their beliefs/desires/intentions.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.perspective_transform = nn.Linear(dim, dim, bias=False)
        # Orthogonal init for the perspective rotation
        self.perspective_transform.weight = nn.init.orthogonal()(
            self.perspective_transform.weight
        )
        self.attn = nn.MultiHeadAttention(dim, num_heads, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)

    def __call__(
        self,
        query: mx.array,
        lower_level_states: List[mx.array],
    ) -> mx.array:
        if not lower_level_states:
            return query

        kv = mx.stack(lower_level_states, axis=1)  # (B, num_lower, D)
        kv = self.perspective_transform(kv)
        kv = self.norm_kv(kv)

        q = mx.expand_dims(self.norm_q(query), axis=1)  # (B, 1, D)
        out = self.attn(q, kv, kv)
        out = mx.squeeze(out, axis=1)  # (B, D)

        g = mx.sigmoid(self.gate_proj(out))
        return query + g * self.attn_drop(out)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    FRACTAL MENTALIZING COLUMN                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalMentalizingColumn(nn.Module):
    """Column *k* in the fractal architecture (k-th order mentalizing).

    Stacks ``num_blocks`` SelfSimilarBlocks; columns with k > 0 receive
    cross-level context from all columns 0..k−1.
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

        self.blocks = [
            SelfSimilarBlock(
                dim=dim,
                factor_dim=factor_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                use_cross_attn=(depth_index > 0),
            )
            for _ in range(num_blocks)
        ]

        self.perspective_attn: Optional[PerspectiveShiftAttention] = None
        if depth_index > 0:
            self.perspective_attn = PerspectiveShiftAttention(
                dim, num_heads, dropout,
            )

        self.output_norm = nn.LayerNorm(dim)

    def __call__(
        self,
        x: mx.array,
        lower_level_outputs: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, List[BDIState]]:
        bdis: List[BDIState] = []

        # Perspective shift: enrich input with lower-level context
        if self.perspective_attn is not None and lower_level_outputs:
            x = self.perspective_attn(x, lower_level_outputs)

        h = x
        for block in self.blocks:
            ctx = None
            if lower_level_outputs and block.use_cross_attn:
                ctx = lower_level_outputs[-1]
            h, bdi = block(h, context=ctx)
            bdis.append(bdi)

        h = self.output_norm(h)
        return h, bdis


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   EPISTEMIC GATING MODULE                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


class EpistemicGate(nn.Module):
    """Scales column output by confidence = 1 / (1 + σ_k)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj1 = nn.Linear(dim, dim // 4)
        self.proj2 = nn.Linear(dim // 4, 1)

    def __call__(self, h: mx.array) -> Tuple[mx.array, mx.array]:
        sigma = nn.softplus(self.proj2(nn.gelu(self.proj1(h))))  # (B, 1)
        confidence = 1.0 / (1.0 + sigma)
        return h * confidence, sigma


# ╔══════════════════════════════════════════════════════════════════════╗
# ║       GUIDING BELIEF MODULE  (FractalGen-inspired)                 ║
# ╚══════════════════════════════════════════════════════════════════════╝


class GuidingBeliefModule(nn.Module):
    """Coarse-to-fine FiLM conditioning (inspired by FractalGen's
    "guiding pixel"; Li et al., 2025).

    Predicts a gist belief from the input and injects it into each
    SelfSimilarBlock via Feature-wise Linear Modulation:

        h' = γ(gist) ⊙ h + β(gist)
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
        self.film_gamma = nn.Linear(gist_dim, output_dim)
        self.film_beta = nn.Linear(gist_dim, output_dim)

        # Identity init:  γ → 1, β → 0  so the module is a no-op at start
        self.film_gamma.weight = mx.zeros_like(self.film_gamma.weight)
        self.film_gamma.bias = mx.ones((output_dim,))
        self.film_beta.weight = mx.zeros_like(self.film_beta.weight)
        self.film_beta.bias = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        gist = self.gist_encoder(x)
        gamma = self.film_gamma(gist)
        beta = self.film_beta(gist)
        return gamma, beta

    @staticmethod
    def modulate(h: mx.array, gamma: mx.array, beta: mx.array) -> mx.array:
        """Apply FiLM modulation."""
        if h.ndim == 3:
            gamma = mx.expand_dims(gamma, axis=1)
            beta = mx.expand_dims(beta, axis=1)
        return gamma * h + beta


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    BAYESIAN BELIEF REVISION                        ║
# ╚══════════════════════════════════════════════════════════════════════╝


class BeliefRevisionModule(nn.Module):
    """Gated Bayesian belief update:

        posterior = gate × evidence + (1 − gate) × prior
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

    def __call__(self, prior: mx.array, evidence: mx.array) -> mx.array:
        prior_h = self.prior_proj(prior)
        evidence_h = self.evidence_proj(evidence)
        gate = self.gate_net(mx.concatenate([prior_h, evidence_h], axis=-1))
        posterior = gate * evidence_h + (1.0 - gate) * prior_h
        return self.norm(posterior)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          STRUCTURAL CAUSAL MODEL (SCM) MODULES                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class LearnableCausalGraph(nn.Module):
    """Differentiable DAG with NOTEARS acyclicity penalty.

    Parameterises a continuous adjacency matrix A ∈ ℝ^{d×d};
    h(A) = tr(e^{A⊙A}) − d = 0  iff DAG.
    """

    def __init__(self, num_variables: int, init_sparsity: float = 0.3):
        super().__init__()
        self.num_variables = num_variables
        # Trainable (public attribute → discovered by Module)
        self.raw_adjacency = (
            mx.random.normal((num_variables, num_variables)) * init_sparsity
        )
        # Non-trainable mask (private attribute)
        self._diag_mask = 1.0 - mx.eye(num_variables)

    @property
    def adjacency(self) -> mx.array:
        """Weighted adjacency ∈ [0, 1]^{d×d}, no self-loops."""
        return mx.sigmoid(self.raw_adjacency) * self._diag_mask

    def dag_penalty(self) -> mx.array:
        """NOTEARS: 0 iff DAG."""
        A = self.adjacency
        M = A * A
        expm = _matrix_exp_taylor(M)
        return _trace(expm) - self.num_variables

    def __call__(self) -> mx.array:
        return self.adjacency


class StructuralEquationNetwork(nn.Module):
    """Neural structural equation  X_j = f_j(Pa(X_j), ε_j)."""

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
        self.gate_net = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def __call__(
        self,
        parent_repr: mx.array,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        h_parent = self.parent_proj(parent_repr)
        if noise is None:
            noise = mx.random.normal((parent_repr.shape[0], self.noise_dim))
        h_noise = self.noise_proj(noise)
        combined = self.combine(mx.concatenate([h_parent, h_noise], axis=-1))
        g = self.gate_net(combined)
        return g * combined + (1.0 - g) * h_parent


class StructuralCausalModel(nn.Module):
    """Differentiable SCM over BDI variables — supports all three
    levels of Pearl's causal hierarchy (association, intervention,
    counterfactual).
    """

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

        self.causal_graph = LearnableCausalGraph(
            num_variables=self.NUM_VARS, init_sparsity=0.3,
        )
        # BDI prior (Bratman, 1987)
        self.causal_graph.raw_adjacency = mx.array([
            #  O     B     D     I
            [0.0, 2.0, 0.5, 0.0],   # from Obs
            [0.0, 0.0, 1.5, 2.0],   # from Belief
            [0.0, 0.0, 0.0, 1.5],   # from Desire
            [0.0, 0.0, 0.0, 0.0],   # from Intention
        ])

        self.eq_belief = StructuralEquationNetwork(factor_dim, noise_dim, dropout)
        self.eq_desire = StructuralEquationNetwork(factor_dim, noise_dim, dropout)
        self.eq_intention = StructuralEquationNetwork(factor_dim, noise_dim, dropout)

        self.noise_encoder = nn.Sequential(
            nn.Linear(factor_dim * 3, factor_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim * 2, noise_dim * 3),
        )

        self.parent_agg = [
            nn.Linear(factor_dim * self.NUM_VARS, factor_dim)
            for _ in range(self.NUM_VARS)
        ]

    # -- helpers ----------------------------------------------------------

    def _aggregate_parents(
        self,
        var_idx: int,
        variables: List[mx.array],
        adjacency: mx.array,
    ) -> mx.array:
        parent_weights = adjacency[:, var_idx]  # (NUM_VARS,)
        weighted = [v * parent_weights[i] for i, v in enumerate(variables)]
        stacked = mx.concatenate(weighted, axis=-1)  # (B, factor_dim*NUM_VARS)
        return self.parent_agg[var_idx](stacked)

    # -- Level 1: Association (seeing) ------------------------------------

    def __call__(
        self,
        observation: mx.array,
        bdi_init: BDIState,
    ) -> Tuple[BDIState, Dict[str, mx.array]]:
        """Forward (associational) causal pass — P(BDI | Obs)."""
        A = self.causal_graph()
        variables = [
            observation,
            bdi_init.belief,
            bdi_init.desire,
            bdi_init.intention,
        ]

        parent_b = self._aggregate_parents(self.VAR_BELIEF, variables, A)
        belief = self.eq_belief(parent_b)
        variables[self.VAR_BELIEF] = belief

        parent_d = self._aggregate_parents(self.VAR_DESIRE, variables, A)
        desire = self.eq_desire(parent_d)
        variables[self.VAR_DESIRE] = desire

        parent_i = self._aggregate_parents(self.VAR_INTENTION, variables, A)
        intention = self.eq_intention(parent_i)

        return BDIState(belief, desire, intention), {
            "adjacency": A,
            "dag_penalty": self.causal_graph.dag_penalty(),
        }

    # -- Level 2: Intervention (doing) ------------------------------------

    def intervene(
        self,
        observation: mx.array,
        bdi_init: BDIState,
        target: int,
        value: mx.array,
    ) -> BDIState:
        """Pearl's do(X_target = value) — sever incoming edges."""
        A = self.causal_graph()
        # Zero out column ``target`` (sever incoming edges)
        col_mask = (mx.arange(A.shape[1]) != target).astype(A.dtype)
        A = A * col_mask

        variables = [
            observation,
            bdi_init.belief,
            bdi_init.desire,
            bdi_init.intention,
        ]
        variables[target] = value

        eqs = [None, self.eq_belief, self.eq_desire, self.eq_intention]
        for var_idx in range(self.VAR_BELIEF, self.NUM_VARS):
            if var_idx == target:
                continue
            parent = self._aggregate_parents(var_idx, variables, A)
            variables[var_idx] = eqs[var_idx](parent)

        return BDIState(
            variables[self.VAR_BELIEF],
            variables[self.VAR_DESIRE],
            variables[self.VAR_INTENTION],
        )

    # -- Level 3: Counterfactual (imagining) ------------------------------

    def counterfactual(
        self,
        observation: mx.array,
        bdi_observed: BDIState,
        cf_observation: mx.array,
    ) -> BDIState:
        """Abduction → Action → Prediction (Pearl, 2009)."""
        bdi_packed = bdi_observed.pack()
        noise_all = self.noise_encoder(bdi_packed)
        noise_b = noise_all[..., : self.noise_dim]
        noise_d = noise_all[..., self.noise_dim : 2 * self.noise_dim]
        noise_i = noise_all[..., 2 * self.noise_dim :]

        A = self.causal_graph()
        variables = [
            cf_observation,
            mx.zeros_like(bdi_observed.belief),
            mx.zeros_like(bdi_observed.desire),
            mx.zeros_like(bdi_observed.intention),
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
    """Soft-routes mentalizing columns through Pearl's 3-level hierarchy:
    association / intervention / counterfactual.
    """

    def __init__(self, dim: int, max_depth: int):
        super().__init__()
        self.max_depth = max(max_depth, 1)
        self.num_levels = 3
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, self.num_levels),
        )

    def _depth_prior(self, depth_index: int) -> mx.array:
        """Soft prior over causal levels for depth *k*."""
        if depth_index == 0:
            return mx.array([2.0, 0.5, 0.0])
        elif depth_index == 1:
            return mx.array([0.5, 2.0, 0.5])
        else:
            ratio = min(depth_index / self.max_depth, 1.0)
            return mx.array([0.2, 1.0 - 0.5 * ratio, 1.0 + ratio])

    def __call__(self, h: mx.array, depth_index: int) -> mx.array:
        """Returns (batch, 3) soft weights."""
        logits = self.router(h) + self._depth_prior(depth_index)
        return mx.softmax(logits, axis=-1)


class CausalDiscoveryModule(nn.Module):
    """Discovers cross-depth causal structure via pairwise scoring
    with depth-ordering prior and NOTEARS penalty.
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

        # Build depth-ordering prior: shallower → deeper preferred
        size = max_depth + 1
        bias_vals = []
        for i in range(size):
            row = []
            for j in range(size):
                if i < j:
                    row.append(1.0)    # forward causation
                elif i > j:
                    row.append(-2.0)   # discourage backward
                else:
                    row.append(0.0)
            bias_vals.append(row)
        self.depth_bias = mx.array(bias_vals)

    def discover(
        self,
        bdi_per_depth: Dict[int, BDIState],
        observation: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Returns (cross_adj, penalty)."""
        depth_reps = []
        for k in sorted(bdi_per_depth.keys()):
            bdi = bdi_per_depth[k]
            rep = (bdi.belief + bdi.desire + bdi.intention) / 3.0
            depth_reps.append(rep)

        K = len(depth_reps)

        # Build adjacency by scoring all pairs
        adj_rows = []
        for i in range(K):
            row_vals = []
            for j in range(K):
                if i == j:
                    row_vals.append(mx.array(0.0))
                    continue
                pair = mx.concatenate(
                    [
                        mx.mean(depth_reps[i], axis=0, keepdims=True),
                        mx.mean(depth_reps[j], axis=0, keepdims=True),
                    ],
                    axis=-1,
                )
                score = mx.squeeze(self.edge_scorer(pair))
                bias = (
                    self.depth_bias[i, j]
                    if (i < self.depth_bias.shape[0] and j < self.depth_bias.shape[1])
                    else mx.array(0.0)
                )
                row_vals.append(mx.sigmoid(score + bias))
            adj_rows.append(mx.stack(row_vals))
        adj = mx.stack(adj_rows)

        # DAG penalty (NOTEARS)
        M = adj * adj
        expm = _matrix_exp_taylor(M)
        dag_pen = _trace(expm) - K
        sparsity = mx.sum(adj)
        penalty = dag_pen + 0.01 * sparsity

        return adj, penalty


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 ATTENTION-WEIGHTED COLUMN JOIN                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MentalizingJoin(nn.Module):
    """Input-dependent soft selection:  joined = Σ_k α_k h_k."""

    def __init__(self, dim: int, max_depth: int):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self._scale = math.sqrt(dim)

    def __call__(
        self,
        column_outputs: List[mx.array],
        input_rep: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        q = mx.expand_dims(self.query_proj(input_rep), axis=1)  # (B, 1, D)
        keys = mx.stack(column_outputs, axis=1)                 # (B, K+1, D)
        k = self.key_proj(keys)
        scores = mx.sum(q * k, axis=-1) / self._scale           # (B, K+1)
        alpha = mx.softmax(scores, axis=-1)                      # (B, K+1)
        joined = mx.sum(mx.expand_dims(alpha, axis=-1) * keys, axis=1)
        return joined, alpha


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  INTERPRETABILITY REPORT                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


@dataclass
class InterpretabilityReport:
    """Collects all interpretable signals from a forward pass."""

    depth_weights: mx.array
    bdi_states: Dict[int, List[BDIState]] = field(default_factory=dict)
    column_uncertainties: Dict[int, mx.array] = field(default_factory=dict)
    belief_revision_gate: Optional[mx.array] = None
    causal_adjacency: Optional[mx.array] = None
    causal_hierarchy_weights: Optional[Dict[int, mx.array]] = None
    cross_depth_adjacency: Optional[mx.array] = None
    dag_penalty: Optional[mx.array] = None
    counterfactual_distances: Optional[Dict[int, float]] = None
    auxiliary_logits: Optional[Dict[int, mx.array]] = None
    guiding_gists: Optional[Dict[int, Tuple[mx.array, mx.array]]] = None
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

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(x)


class ToMPredictionHead(nn.Module):
    """Outputs a BDI prediction for Theory-of-Mind tasks."""

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

    def __call__(self, x: mx.array) -> BDIState:
        raw = self.proj(x)
        return BDIState.unpack(raw, self.factor_dim)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              COMPLETE FracToM ARCHITECTURE                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FracToMNet(nn.Module):
    """Fractal Theory-of-Mind Network — complete architecture (MLX).

    Input → MentalStateEncoder → Fractal Columns → SCM → Causal Discovery
    → Epistemic Gating → Attention Join → Belief Revision → Task Head.
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

        # --- Per-depth capacity schedule ---
        K = mentalizing_depth
        _quantum = (
            num_bdi_factors
            * num_heads
            // math.gcd(num_bdi_factors, num_heads)
        )
        if capacity_schedule == "decreasing":
            self.column_dims = []
            for k in range(K + 1):
                ratio = 1.0 - 0.5 * (k / max(K, 1))
                raw = int(hidden_dim * ratio)
                raw = max(raw - raw % _quantum, _quantum)
                self.column_dims.append(raw)
        else:
            self.column_dims = [hidden_dim] * (K + 1)

        # --- Observation encoder → BDI ---
        self.encoder = MentalStateEncoder(
            input_dim, self.factor_dim, dropout,
        )
        self.input_proj = nn.Linear(self.factor_dim * 3, hidden_dim)

        # --- Per-column input / output projectors ---
        self.col_input_projs = [
            nn.Linear(hidden_dim, self.column_dims[k])
            if self.column_dims[k] != hidden_dim
            else Identity()
            for k in range(K + 1)
        ]
        self.col_output_projs = [
            nn.Linear(self.column_dims[k], hidden_dim)
            if self.column_dims[k] != hidden_dim
            else Identity()
            for k in range(K + 1)
        ]

        # --- Fractal mentalizing columns ---
        self.columns = [
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
        ]

        # --- Epistemic gates ---
        self.epistemic_gates = [
            EpistemicGate(self.column_dims[k]) for k in range(K + 1)
        ]

        # --- Guiding Belief Module ---
        self.guiding_beliefs: Optional[list] = None
        if guiding_belief:
            self.guiding_beliefs = [
                GuidingBeliefModule(
                    input_dim=hidden_dim,
                    gist_dim=gist_dim,
                    output_dim=self.column_dims[k],
                )
                for k in range(K + 1)
            ]

        # --- Auxiliary heads (deep supervision) ---
        self.aux_heads: Optional[list] = None
        if self.use_auxiliary_heads and num_classes is not None:
            self.aux_heads = [
                nn.Linear(hidden_dim, num_classes) for _ in range(K + 1)
            ]

        # --- Structural Causal Model ---
        self.use_causal = causal_model
        if causal_model:
            self.scm = StructuralCausalModel(
                factor_dim=self.factor_dim,
                noise_dim=causal_noise_dim,
                dropout=dropout,
            )
            self.causal_router = CausalHierarchyRouter(
                dim=hidden_dim, max_depth=mentalizing_depth,
            )
            self.causal_discovery = CausalDiscoveryModule(
                factor_dim=self.factor_dim,
                max_depth=mentalizing_depth,
                dropout=dropout,
            )
            self.obs_to_factor = nn.Linear(hidden_dim, self.factor_dim)
            self.causal_to_hidden = nn.Linear(self.factor_dim * 3, hidden_dim)
            self.cf_obs_transform = nn.Linear(
                self.factor_dim, self.factor_dim, bias=False,
            )
            self.bdi_to_scm_projs = [
                nn.Linear(
                    self.column_dims[k] // num_bdi_factors, self.factor_dim
                )
                if self.column_dims[k] // num_bdi_factors != self.factor_dim
                else Identity()
                for k in range(K + 1)
            ]

        # --- Drop-path ---
        self.drop_path_mod = FractalDropPath(
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
        """Glorot-uniform for Linear layers; restore FiLM identity init."""

        def _init(key: str, module: nn.Module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.glorot_uniform()(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight = mx.ones_like(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)

        self.apply_to_modules(_init)

        # Restore causal graph BDI prior (overwritten by _init)
        if self.use_causal:
            self.scm.causal_graph.raw_adjacency = mx.array([
                [0.0, 2.0, 0.5, 0.0],
                [0.0, 0.0, 1.5, 2.0],
                [0.0, 0.0, 0.0, 1.5],
                [0.0, 0.0, 0.0, 0.0],
            ])

        # Restore FiLM identity init (γ=1, β=0 so module is no-op)
        if self.guiding_beliefs is not None:
            for gb in self.guiding_beliefs:
                gb.film_gamma.weight = mx.zeros_like(gb.film_gamma.weight)
                gb.film_gamma.bias = mx.ones_like(gb.film_gamma.bias)
                gb.film_beta.weight = mx.zeros_like(gb.film_beta.weight)
                gb.film_beta.bias = mx.zeros_like(gb.film_beta.bias)

        # Restore orthogonal init for perspective transforms
        for col in self.columns:
            if col.perspective_attn is not None:
                w = col.perspective_attn.perspective_transform.weight
                col.perspective_attn.perspective_transform.weight = (
                    nn.init.orthogonal()(w)
                )

    # ------------------------------------------------------------------ fwd

    def __call__(
        self,
        x: mx.array,
        return_interpretability: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, InterpretabilityReport]]:
        B = x.shape[0]

        # 1) Encode observation → initial BDI
        bdi_init, sigma_init = self.encoder(x)
        h_input = self.input_proj(bdi_init.pack())  # (B, D)

        # 2) Run fractal columns (depth 0 … K)
        column_outputs: List[mx.array] = []
        all_bdis: Dict[int, List[BDIState]] = {}
        all_sigmas: Dict[int, mx.array] = {}
        guiding_gists: Dict[int, Tuple[mx.array, mx.array]] = {}
        aux_logits: Dict[int, mx.array] = {}
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
                guiding_gists[k] = (
                    mx.stop_gradient(gamma),
                    mx.stop_gradient(beta),
                )

            # Build lower-level outputs in column's native dim
            lower: Optional[List[mx.array]] = None
            if k > 0:
                lower = [
                    self.col_input_projs[k](column_outputs[j])
                    for j in range(k)
                ]

            h_col, bdis_col = col(h_col_in, lower)

            # Epistemic gating (in native col dim)
            h_col, sigma_k = egate(h_col)
            all_sigmas[k] = sigma_k

            # Drop-path (training only)
            h_col = self.drop_path_mod(h_col, column_index=k)

            # Project to hidden_dim for join & causal modules
            h_col_out = self.col_output_projs[k](h_col)
            column_outputs.append(h_col_out)
            all_bdis[k] = bdis_col

            # Store projected BDI states (common factor_dim) for loss
            if self.use_causal:
                _proj_k = self.bdi_to_scm_projs[k]
                projected_bdis[k] = [
                    BDIState(
                        _proj_k(b.belief),
                        _proj_k(b.desire),
                        _proj_k(b.intention),
                    )
                    for b in bdis_col
                ]

            # Auxiliary deep supervision head
            if self.aux_heads is not None:
                aux_logits[k] = self.aux_heads[k](h_col_out)

        # 2.5) Causal processing via SCM
        causal_info: Dict = {}
        if self.use_causal:
            obs_factor = self.obs_to_factor(h_input)  # (B, factor_dim)
            causal_hierarchy_weights: Dict[int, mx.array] = {}
            cf_distances: Dict[int, float] = {}
            causal_adj: Optional[mx.array] = None
            dag_pen_total = mx.array(0.0)

            for k in range(len(self.columns)):
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
                )
                causal_hierarchy_weights[k] = level_weights

                # Level 1: Association
                bdi_assoc, scm_info = self.scm(obs_factor, bdi_k_scm)
                causal_adj = scm_info["adjacency"]
                dag_pen_total = dag_pen_total + scm_info["dag_penalty"]

                # Level 2: Intervention — do(Belief = belief_k)
                bdi_interv = self.scm.intervene(
                    obs_factor,
                    bdi_k_scm,
                    target=StructuralCausalModel.VAR_BELIEF,
                    value=bdi_k_scm.belief,
                )

                # Level 3: Counterfactual
                cf_obs = self.cf_obs_transform(obs_factor)
                bdi_cf = self.scm.counterfactual(
                    obs_factor, bdi_k_scm, cf_obs,
                )

                # Counterfactual distance (interpretability metric)
                cf_dist_val = mx.mean(
                    _l2_norm(bdi_cf.pack() - bdi_k_scm.pack(), axis=-1)
                ).item()
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
                )

                # Enrich column output with causal BDI information
                causal_h = self.causal_to_hidden(blended_bdi)
                column_outputs[k] = column_outputs[k] + causal_h

            # Cross-depth causal discovery
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

        # 2.6) Cross-depth causal gating
        if self.use_causal and causal_info.get("cross_depth_adjacency") is not None:
            cross_adj = causal_info["cross_depth_adjacency"]
            K_cols = len(column_outputs)
            cols_stacked = mx.stack(column_outputs, axis=0)  # (K+1, B, D)
            for k in range(K_cols):
                weights = cross_adj[:, k]  # (K+1,)
                # Zero out self-loop
                mask = (mx.arange(K_cols) != k).astype(weights.dtype)
                weights = weights * mask
                if mx.sum(weights).item() > 1e-8:
                    w = weights.reshape(-1, 1, 1)  # (K+1, 1, 1)
                    causal_contrib = mx.sum(w * cols_stacked, axis=0)
                    column_outputs[k] = column_outputs[k] + causal_contrib

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
                cross_depth_adjacency=causal_info.get("cross_depth_adjacency"),
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
        """Anneal developmental drop-path: 1.0 → 0.0 over training."""
        self.drop_path_mod.set_curriculum(factor)

    def get_tom_head(
        self, factor_dim: Optional[int] = None, dropout: float = 0.1,
    ) -> ToMPredictionHead:
        fd = factor_dim or self.factor_dim
        return ToMPredictionHead(self.hidden_dim, fd, dropout)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              LOSS FUNCTIONS & TRAINING UTILITIES                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FracToMLoss:
    """Composite loss for the FracToM architecture.

    8-component loss: task + BDI consistency + uncertainty calibration +
    depth entropy + DAG penalty + causal sparsity + counterfactual
    ordering + auxiliary deep supervision.

    Use ``compute_loss()`` inside ``nn.value_and_grad`` (no ``.item()``
    calls) and ``__call__`` for logging (with ``.item()`` for readability).
    """

    def __init__(
        self,
        task_loss_fn: Optional[Callable] = None,
        lambda_bdi: float = 0.01,
        lambda_uncertainty: float = 0.005,
        lambda_depth_entropy: float = 0.01,
        lambda_dag: float = 0.1,
        lambda_causal_sparsity: float = 0.005,
        lambda_counterfactual: float = 0.01,
        lambda_auxiliary: float = 0.1,
    ):
        self.task_loss_fn = task_loss_fn or (
            lambda logits, targets: mx.mean(
                nn.losses.cross_entropy(logits, targets)
            )
        )
        self.lambda_bdi = lambda_bdi
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_depth_entropy = lambda_depth_entropy
        self.lambda_dag = lambda_dag
        self.lambda_causal_sparsity = lambda_causal_sparsity
        self.lambda_counterfactual = lambda_counterfactual
        self.lambda_auxiliary = lambda_auxiliary

    def compute_loss(
        self,
        logits: mx.array,
        targets: mx.array,
        report: InterpretabilityReport,
    ) -> mx.array:
        """Differentiable scalar loss (no ``.item()`` calls).

        Safe for use inside ``nn.value_and_grad``.
        """
        # 1) Task loss
        task = self.task_loss_fn(logits, targets)

        # 2) BDI consistency
        bdi_loss = mx.array(0.0)
        bdi_src = report.projected_bdi_states or report.bdi_states
        depths = sorted(bdi_src.keys())
        for idx in range(len(depths) - 1):
            a = bdi_src[depths[idx]][-1].pack()
            b = bdi_src[depths[idx + 1]][-1].pack()
            cos = mx.mean(_cosine_similarity(a, b, axis=-1))
            bdi_loss = bdi_loss + (1.0 - cos)
        if len(depths) > 1:
            bdi_loss = bdi_loss / (len(depths) - 1)

        # 3) Uncertainty calibration
        unc_loss = mx.array(0.0)
        if logits.ndim == 2:
            preds = mx.argmax(logits, axis=-1)
            correct = (preds == targets).astype(mx.float32)
            for sigma in report.column_uncertainties.values():
                s = mx.squeeze(sigma, axis=-1)
                unc_loss = unc_loss + mx.mean(correct * s - (1 - correct) * s)
            unc_loss = unc_loss / max(len(report.column_uncertainties), 1)

        # 4) Depth entropy bonus
        alpha = report.depth_weights
        depth_ent = -mx.sum(alpha * mx.log(alpha + 1e-8), axis=-1)
        depth_ent = mx.mean(depth_ent)
        max_ent = math.log(alpha.shape[-1])
        depth_reg = max_ent - depth_ent

        loss = (
            task
            + self.lambda_bdi * bdi_loss
            + self.lambda_uncertainty * unc_loss
            + self.lambda_depth_entropy * depth_reg
        )

        # 5) DAG acyclicity penalty
        if report.dag_penalty is not None:
            loss = loss + self.lambda_dag * report.dag_penalty

        # 6) Causal sparsity
        causal_sparse = mx.array(0.0)
        if report.causal_adjacency is not None:
            causal_sparse = causal_sparse + mx.sum(report.causal_adjacency)
        if report.cross_depth_adjacency is not None:
            causal_sparse = causal_sparse + 0.1 * mx.sum(
                report.cross_depth_adjacency
            )
        loss = loss + self.lambda_causal_sparsity * causal_sparse

        # 7) Counterfactual ordering
        cf_loss = mx.array(0.0)
        if report.counterfactual_distances:
            dists = report.counterfactual_distances
            ds = sorted(dists.keys())
            for idx in range(len(ds) - 1):
                diff = dists[ds[idx]] - dists[ds[idx + 1]]
                if diff > 0:
                    cf_loss = cf_loss + diff
            if len(ds) > 1:
                cf_loss = cf_loss / (len(ds) - 1)
        loss = loss + self.lambda_counterfactual * cf_loss

        # 8) Auxiliary deep supervision
        aux_loss = mx.array(0.0)
        if report.auxiliary_logits:
            n_aux = 0
            for _k, aux_log in report.auxiliary_logits.items():
                aux_loss = aux_loss + mx.mean(
                    nn.losses.cross_entropy(aux_log, targets)
                )
                n_aux += 1
            if n_aux > 0:
                aux_loss = aux_loss / n_aux
        loss = loss + self.lambda_auxiliary * aux_loss

        return loss

    def __call__(
        self,
        logits: mx.array,
        targets: mx.array,
        report: InterpretabilityReport,
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Returns (loss, breakdown_dict) — ``breakdown`` values are
        Python floats for logging convenience."""
        loss = self.compute_loss(logits, targets, report)
        # We cannot cheaply recompute individual components for breakdown
        # without duplicating work, so just return the total.
        breakdown = {"total": loss.item()}
        return loss, breakdown


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               INTERPRETABILITY ANALYSIS UTILITIES                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


def analyse_mentalizing_depth(
    report: InterpretabilityReport,
    class_names: Optional[List[str]] = None,
) -> str:
    """Human-readable analysis string."""
    alpha = report.depth_weights
    B, K1 = alpha.shape
    lines = [f"Mentalizing Depth Analysis ({B} samples, {K1} levels)"]
    lines.append("=" * 60)

    mean_alpha = mx.mean(alpha, axis=0)
    for k in range(K1):
        label = _depth_label(k)
        bar = "█" * int(mean_alpha[k].item() * 40)
        lines.append(
            f"  Level {k} ({label:20s}): {mean_alpha[k].item():.3f}  {bar}"
        )
    lines.append("")

    dominant = mx.argmax(alpha, axis=-1)
    lines.append("Dominant mentalizing level per sample:")
    for k in range(K1):
        count = int(mx.sum(dominant == k).item())
        lines.append(f"  Level {k}: {count}/{B} samples ({100*count/B:.1f}%)")

    if report.column_uncertainties:
        lines.append("")
        lines.append("Epistemic uncertainty (σ) per column:")
        for k, sigma in sorted(report.column_uncertainties.items()):
            s = mx.mean(sigma).item()
            lines.append(f"  Column {k}: σ = {s:.4f}")

    if report.causal_adjacency is not None:
        lines.append("")
        lines.append("Learned BDI Causal Graph:")
        var_names = ["Obs", "Belief", "Desire", "Intention"]
        A = mx.stop_gradient(report.causal_adjacency)
        header = "         " + "  ".join(f"{n:>9s}" for n in var_names)
        lines.append(header)
        for i, name in enumerate(var_names):
            vals = "  ".join(f"{A[i, j].item():.4f}   " for j in range(4))
            lines.append(f"  {name:9s} {vals}")
        lines.append("")
        lines.append("  Key edges (> 0.5):")
        for i in range(4):
            for j in range(4):
                if A[i, j].item() > 0.5:
                    lines.append(
                        f"    {var_names[i]} → {var_names[j]}: "
                        f"{A[i, j].item():.3f}"
                    )

    if report.dag_penalty is not None:
        lines.append(
            f"  DAG penalty: {report.dag_penalty.item():.6f}"
        )

    if report.causal_hierarchy_weights:
        lines.append("")
        lines.append("Pearl's Causal Hierarchy per column:")
        level_names = ["Association", "Intervention", "Counterfactual"]
        for k in sorted(report.causal_hierarchy_weights.keys()):
            w = mx.mean(report.causal_hierarchy_weights[k], axis=0)
            parts = "  ".join(
                f"{level_names[i]}: {w[i].item():.3f}" for i in range(3)
            )
            lines.append(f"  Column {k}: {parts}")

    if report.cross_depth_adjacency is not None:
        lines.append("")
        lines.append("Cross-Depth Causal Structure:")
        cd = mx.stop_gradient(report.cross_depth_adjacency)
        K = cd.shape[0]
        col_labels = [f"Col {i}" for i in range(K)]
        header = "          " + "  ".join(f"{c:>7s}" for c in col_labels)
        lines.append(header)
        for i in range(K):
            row_vals = "  ".join(
                f"{cd[i, j].item():7.3f}" if i != j else "      ·"
                for j in range(K)
            )
            lines.append(f"  Col {i}  {row_vals}")

    if report.counterfactual_distances:
        lines.append("")
        lines.append("Counterfactual distances per column:")
        for k, d in sorted(report.counterfactual_distances.items()):
            bar = "█" * int(d * 10)
            lines.append(f"  Column {k}: {d:.4f}  {bar}")

    return "\n".join(lines)


def extract_bdi_activations(
    report: InterpretabilityReport,
) -> Dict[int, Dict[str, mx.array]]:
    """Extract BDI activations per column."""
    result = {}
    for k, bdis in report.bdi_states.items():
        last = bdis[-1]
        result[k] = {
            "belief": mx.stop_gradient(last.belief),
            "desire": mx.stop_gradient(last.desire),
            "intention": mx.stop_gradient(last.intention),
        }
    return result


def extract_causal_graph(
    report: InterpretabilityReport,
) -> Dict[str, object]:
    """Extract causal structure for downstream analysis."""
    var_names = ["Obs", "Belief", "Desire", "Intention"]
    result: Dict[str, object] = {}

    if report.causal_adjacency is not None:
        A = mx.stop_gradient(report.causal_adjacency)
        result["bdi_adjacency"] = A
        edges = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j].item() > 0.3:
                    edges.append((var_names[i], var_names[j], A[i, j].item()))
        result["bdi_edges"] = edges
    else:
        result["bdi_adjacency"] = None
        result["bdi_edges"] = []

    if report.cross_depth_adjacency is not None:
        cd = mx.stop_gradient(report.cross_depth_adjacency)
        result["cross_depth_adjacency"] = cd
        edges = []
        for i in range(cd.shape[0]):
            for j in range(cd.shape[1]):
                if cd[i, j].item() > 0.15:
                    edges.append((i, j, cd[i, j].item()))
        result["cross_depth_edges"] = edges
    else:
        result["cross_depth_adjacency"] = None
        result["cross_depth_edges"] = []

    if report.causal_hierarchy_weights is not None:
        result["hierarchy_weights"] = {
            k: mx.stop_gradient(mx.mean(v, axis=0))
            for k, v in report.causal_hierarchy_weights.items()
        }
    else:
        result["hierarchy_weights"] = {}

    result["counterfactual_distances"] = report.counterfactual_distances or {}

    if report.dag_penalty is not None:
        result["dag_penalty"] = report.dag_penalty.item()
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


class TransformerEncoderBlock(nn.Module):
    """Single pre-norm Transformer encoder block."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dims)
        self.ff = nn.Sequential(
            nn.Linear(dims, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dims),
            nn.Dropout(dropout),
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + self.attn_drop(h)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm Transformer encoder blocks."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderBlock(dims, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pos = mx.arange(max_len).reshape(-1, 1).astype(mx.float32)
        div = mx.exp(
            mx.arange(0, dim, 2).astype(mx.float32)
            * (-math.log(10000.0) / dim)
        )
        sin_part = mx.sin(pos * div)  # (max_len, dim//2)
        cos_part = mx.cos(pos * div)  # (max_len, dim//2)
        # Interleave sin/cos: stack → (max_len, dim//2, 2) → reshape (max_len, dim)
        pe = mx.reshape(
            mx.stack([sin_part, cos_part], axis=-1), (max_len, dim),
        )
        self._pe = mx.expand_dims(pe, axis=0)  # (1, max_len, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self._pe[:, : x.shape[1]]


class SequenceFracToM(nn.Module):
    """Wraps FracToMNet for variable-length sequence (e.g., text) inputs.

    Adds Embedding → PosEnc → TransformerEncoder → Pool → FracToMNet.
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
        self.seq_encoder = TransformerEncoder(
            dims=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim * 4,
            num_layers=seq_encoder_layers,
            dropout=dropout,
        )
        self.pool_proj = nn.Linear(embed_dim, embed_dim)

        fractom_kwargs.setdefault("hidden_dim", embed_dim)
        self.fractom = FracToMNet(input_dim=embed_dim, **fractom_kwargs)

    def __call__(
        self,
        token_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_interpretability: bool = False,
    ):
        """
        Parameters
        ----------
        token_ids : (batch, seq_len) int array.
        attention_mask : (batch, seq_len) bool array, True = pad.
        """
        x = self.embedding(token_ids)
        x = self.pos_enc(x)

        # Convert padding mask to additive attention mask
        attn_mask = None
        if attention_mask is not None:
            # (B, S) → (B, 1, 1, S): True → -1e9, False → 0
            attn_mask = mx.where(
                mx.expand_dims(mx.expand_dims(attention_mask, axis=1), axis=1),
                mx.array(-1e9),
                mx.array(0.0),
            )

        x = self.seq_encoder(x, mask=attn_mask)

        # Mean-pool (masking out padding)
        if attention_mask is not None:
            mask_f = mx.expand_dims(
                (~attention_mask).astype(mx.float32), axis=-1
            )
            pooled = mx.sum(x * mask_f, axis=1) / mx.clip(
                mx.sum(mask_f, axis=1), a_min=1.0, a_max=None
            )
        else:
            pooled = mx.mean(x, axis=1)

        pooled = self.pool_proj(pooled)
        return self.fractom(
            pooled, return_interpretability=return_interpretability,
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      DEMO / SMOKE TEST                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


def demo_classification() -> None:
    """Demonstrates FracToM on a synthetic false-belief classification task
    using MLX's training loop (``nn.value_and_grad``).
    """
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    print("=" * 70)
    print("FracToM (MLX) Demo — Synthetic Theory-of-Mind Classification")
    print("=" * 70)

    mx.random.seed(42)

    # ---- Synthetic "false-belief" dataset ----
    N_TRAIN, N_TEST = 2000, 400
    INPUT_DIM = 128
    NUM_CLASSES = 2

    def make_data(n: int) -> Tuple[mx.array, mx.array]:
        xs, ys = [], []
        import random

        random.seed(42)
        for _ in range(n):
            x = mx.random.normal((INPUT_DIM,))
            mx.eval(x)  # force eval so .item() works
            sally_saw = x[0].item() > 0
            ball_moved = x[1].item() > 0
            if ball_moved and not sally_saw:
                y = 0
            else:
                y = 1
            x_list = x.tolist()
            x_list[2] = float(sally_saw) + 0.1 * mx.random.normal(()).item()
            x_list[3] = float(ball_moved) + 0.1 * mx.random.normal(()).item()
            xs.append(mx.array(x_list))
            ys.append(y)
        return mx.stack(xs), mx.array(ys, dtype=mx.int32)

    X_train, y_train = make_data(N_TRAIN)
    X_test, y_test = make_data(N_TEST)

    # ---- Build model ----
    model = FracToMNet(
        input_dim=INPUT_DIM,
        hidden_dim=96,
        mentalizing_depth=3,
        num_bdi_factors=3,
        blocks_per_column=1,
        num_heads=4,
        ff_mult=2,
        dropout=0.1,
        drop_path=0.1,
        num_classes=NUM_CLASSES,
        causal_model=True,
        causal_noise_dim=16,
        capacity_schedule="decreasing",
        guiding_belief=True,
        gist_dim=32,
        auxiliary_heads=True,
    )
    mx.eval(model.parameters())

    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {total_params:,}")

    criterion = FracToMLoss(
        lambda_bdi=0.01,
        lambda_uncertainty=0.005,
        lambda_depth_entropy=0.01,
        lambda_dag=0.1,
        lambda_causal_sparsity=0.005,
        lambda_counterfactual=0.01,
        lambda_auxiliary=0.1,
    )

    EPOCHS = 100
    BATCH = 64
    total_steps = EPOCHS * ((N_TRAIN + BATCH - 1) // BATCH)
    schedule = optim.cosine_decay(init=3e-4, decay_steps=total_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    # ---- Train step (closed over model & criterion) ----
    def train_step(xb, yb):
        logits, report = model(xb, return_interpretability=True)
        return criterion.compute_loss(logits, yb, report)

    loss_and_grad_fn = nn.value_and_grad(model, train_step)

    # ---- Training loop ----
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.set_curriculum(1.0 - epoch / EPOCHS)

        # Permutation via argsort of random
        perm = mx.argsort(mx.random.uniform(shape=(N_TRAIN,)))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i : i + BATCH]
            xb = X_train[idx]
            yb = y_train[idx]

            loss, grads = loss_and_grad_fn(xb, yb)
            grads = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        # ---- Evaluation ----
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            logits_test, report_test = model(
                X_test, return_interpretability=True,
            )
            preds = mx.argmax(logits_test, axis=-1)
            acc = mx.mean((preds == y_test).astype(mx.float32)).item()
            best_acc = max(best_acc, acc)

            avg_loss = epoch_loss / max(n_batches, 1)
            print(
                f"Epoch {epoch:3d} │ loss {avg_loss:.4f} │ "
                f"test acc {acc:.3f} │ best {best_acc:.3f}"
            )

    # ---- Final interpretability analysis ----
    model.eval()
    _, report_final = model(X_test, return_interpretability=True)
    print()
    print(analyse_mentalizing_depth(report_final))

    bdi_acts = extract_bdi_activations(report_final)
    print("\nBDI activation norms per column:")
    for k, acts in bdi_acts.items():
        b_n = mx.mean(_l2_norm(acts["belief"], axis=-1)).item()
        d_n = mx.mean(_l2_norm(acts["desire"], axis=-1)).item()
        i_n = mx.mean(_l2_norm(acts["intention"], axis=-1)).item()
        print(
            f"  Column {k} ({_depth_label(k):20s}): "
            f"‖B‖={b_n:.3f}  ‖D‖={d_n:.3f}  ‖I‖={i_n:.3f}"
        )

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
            parts = "  ".join(
                f"{level_names[i]}: {w[i].item():.3f}" for i in range(3)
            )
            print(f"  Column {k}: {parts}")

    if report_final.column_dims is not None:
        print("\nPer-column capacity (FractalGen-inspired scheduling):")
        for k, d in enumerate(report_final.column_dims):
            print(f"  Column {k} ({_depth_label(k):20s}): dim={d}")

    if report_final.auxiliary_logits is not None:
        print("\nAuxiliary head accuracy (deep supervision):")
        for k, aux_log in sorted(report_final.auxiliary_logits.items()):
            aux_pred = mx.argmax(aux_log, axis=-1)
            aux_acc = mx.mean(
                (aux_pred == y_test).astype(mx.float32)
            ).item()
            print(f"  Column {k}: {aux_acc:.3f}")

    if report_final.guiding_gists is not None:
        print("\nGuiding belief gist stats (γ, β norms):")
        for k, (g, b) in sorted(report_final.guiding_gists.items()):
            g_n = mx.mean(_l2_norm(g, axis=-1)).item()
            b_n = mx.mean(_l2_norm(b, axis=-1)).item()
            print(f"  Column {k}: ‖γ‖={g_n:.4f}  ‖β‖={b_n:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Final test accuracy: {best_acc:.3f}")
    print(f"{'=' * 70}")


def demo_sequence_tom() -> None:
    """Forward/backward smoke test for SequenceFracToM."""
    from mlx.utils import tree_flatten

    print("\n" + "=" * 70)
    print("SequenceFracToM (MLX) — Forward/Backward Smoke Test")
    print("=" * 70)

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
    )
    mx.eval(model.parameters())

    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Parameters: {total_params:,}")

    tokens = mx.random.randint(0, 1000, shape=(8, 32))
    mask = mx.zeros((8, 32), dtype=mx.bool_)
    # Last 7 tokens are padding
    mask_list = mask.tolist()
    for row in mask_list:
        for j in range(25, 32):
            row[j] = True
    mask = mx.array(mask_list)

    logits, report = model(tokens, mask, return_interpretability=True)
    print(f"Output shape: {logits.shape}")
    print(f"Depth weights shape: {report.depth_weights.shape}")
    dw = report.depth_weights[0]
    print(f"Depth weights (sample 0): {[round(dw[i].item(), 4) for i in range(dw.shape[0])]}")

    # Test backward via value_and_grad
    def loss_fn(tokens, mask):
        out, _ = model(tokens, mask, return_interpretability=True)
        return mx.sum(out)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_grad_fn(tokens, mask)
    mx.eval(loss)
    print(f"Loss: {loss.item():.4f}")
    print("Backward pass: OK")
    print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         ENTRY POINT                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


if __name__ == "__main__":
    demo_classification()
    demo_sequence_tom()
