"""
baseline.py — FractalGen-Inspired Baseline Network
====================================================

Reproduces the core architecture from:

    "Fractal Generative Models"
    Tianhong Li, Qinyi Sun, Lijie Fan, Kaiming He
    arXiv:2502.17437 (2025)

Adapted from pixel-by-pixel image generation to tabular classification so
that it can be directly compared against FracToM on the same dual-agent
collaboration / competition tasks.

Key Architectural Ideas Retained from the Paper
------------------------------------------------
1. **Recursive Fractal Structure**: The network is built by recursively
   composing self-similar transformer modules at multiple fractal levels,
   mirroring how FractalGen recursively invokes autoregressive models.

2. **Self-Similar Modules**: Each fractal level uses the same template
   (attention blocks + feed-forward), differing only in learned parameters
   and capacity (hidden_dim, num_blocks), exactly like the paper's fractal
   generator design (Table 1).

3. **Condition Passing Between Levels**: The output of each parent level
   serves as the *condition* for the next child level, analogous to how
   ``cond_list`` propagates through fractal levels in fractalgen.py.

4. **Progressive Capacity Reduction**: Deeper fractal levels use smaller
   hidden dimensions and fewer transformer blocks, matching the paper's
   design where finer-grained levels are modelled with lighter networks.

5. **Drop-Path Regularization**: Stochastic depth at the block level,
   consistent with the DropPath used in the paper's AR and MAR modules.

What This Baseline Does **NOT** Have (FracToM-unique features)
--------------------------------------------------------------
- No BDI-factored latent space (Belief-Desire-Intention decomposition)
- No mentalizing columns / Theory-of-Mind hierarchy
- No perspective-shifting cross-depth attention
- No epistemic gating
- No developmental drop-path curriculum
- No Bayesian belief revision module
- No interpretability report (depth weights, BDI states, uncertainty)

Usage
-----
    from baseline import FractalGenNet

    model = FractalGenNet(
        input_dim=36,
        num_classes=4,
        num_levels=3,
        hidden_dims=[256, 128, 64],
        num_blocks_list=[4, 2, 1],
        num_heads_list=[8, 4, 2],
        dropout=0.1,
    )
    logits = model(x)  # x: (batch, input_dim)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        DROP PATH                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample.

    Reproduced from the paper's AR module (ar.py: class DropPath), which
    applies structured drop per sample during training.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        if self.scale_by_keep:
            mask = mask / keep_prob
        return x * mask


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    SELF-SIMILAR TRANSFORMER BLOCK                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalTransformerBlock(nn.Module):
    """A single transformer block used at every fractal level.

    This is the atomic building block analogous to TransformerBlock in the
    paper's AR model and Block in the MAR model.  It follows the standard
    pre-norm architecture:

        LayerNorm → Multi-Head Self-Attention → Residual + DropPath
        LayerNorm → Feed-Forward (SwiGLU) → Residual + DropPath

    The block is **self-similar**: the same template is reused at every
    fractal level, with only the hidden dimension differing — this is the
    core fractal property from the paper.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # --- self-attention ---
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True,
        )

        # --- feed-forward (SwiGLU, following the paper's AR module) ---
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        # SwiGLU: two projections for gating
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)

        # --- drop path ---
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, dim) or (batch, dim)

        Returns
        -------
        (batch, seq_len, dim) or (batch, dim)
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)

        # self-attention with pre-norm
        h = self.norm1(x)
        h_attn, _ = self.attn(h, h, h)
        x = x + self.drop_path(h_attn)

        # feed-forward with SwiGLU
        h = self.norm2(x)
        ff_out = self.ffn_dropout(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        x = x + self.drop_path(ff_out)

        if squeeze:
            x = x.squeeze(1)
        return x


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     FRACTAL LEVEL (GENERATOR)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalLevel(nn.Module):
    """A single fractal level, analogous to one AR/MAR generator in the paper.

    Each level:
    1. Receives a *condition* vector from the parent level (or from the
       input embedding for level 0).
    2. Processes the condition through a stack of self-similar transformer
       blocks.
    3. Produces an output representation that serves as the condition for
       the next (child) level.

    This mirrors fractalgen.py's ``FractalGen`` class where each level
    contains a generator that takes ``cond_list`` from the parent and
    produces conditions for ``next_fractal``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        cond_dim: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # project input features into this level's hidden space
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # condition embedding (from parent fractal level)
        self.has_cond = cond_dim > 0
        if self.has_cond:
            self.cond_proj = nn.Linear(cond_dim, hidden_dim, bias=True)

        # learned positional embedding (analogous to pos_embed_learned in
        # the paper's MAR/AR models)
        # seq_len = 1 for condition token + 1 for input = 2
        max_seq = 2 if self.has_cond else 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # stack of self-similar transformer blocks
        # linearly increasing drop-path rate (following ViT convention)
        dpr = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            FractalTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)  — input features.
        cond : (batch, cond_dim) — condition from parent level (if any).

        Returns
        -------
        (batch, hidden_dim) — output for next level or classifier.
        """
        # project input
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = h.unsqueeze(1)  # (B, 1, D)

        if self.has_cond and cond is not None:
            c = self.cond_proj(cond).unsqueeze(1)  # (B, 1, D)
            h = torch.cat([c, h], dim=1)  # (B, 2, D)

        # add positional embedding
        h = h + self.pos_embed[:, :h.size(1)]

        # pass through transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # mean-pool over sequence dimension
        h = h.mean(dim=1)  # (B, D)
        return h


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 COMPLETE FractalGen CLASSIFIER                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class FractalGenNet(nn.Module):
    """Fractal Generative Model adapted for classification.

    Reproduces the recursive, self-similar architecture from
    "Fractal Generative Models" (arXiv:2502.17437) for tabular
    classification tasks.

    Architecture Overview
    ---------------------
        Input
          │
          ▼
      ┌──────────────────┐
      │  FractalLevel 0  │  (largest capacity)
      │  [N₀ blocks]     │
      └────────┬─────────┘
               │ condition
               ▼
      ┌──────────────────┐
      │  FractalLevel 1  │  (smaller capacity)
      │  [N₁ blocks]     │  ← receives cond from Level 0
      └────────┬─────────┘
               │ condition
               ▼
      ┌──────────────────┐
      │  FractalLevel 2  │  (smallest capacity)
      │  [N₂ blocks]     │  ← receives cond from Level 1
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   Aggregation    │  concat all level outputs
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │  Classifier Head │
      └──────────────────┘

    This mirrors the paper's fractal structure where:
    - Level 0 is the top-level generator (32 blocks, 1024 dim)
    - Level 1 is the mid-level generator (8 blocks, 512 dim)
    - Level 2 is the finest-level generator (3 blocks, 128 dim)
    And conditions flow recursively from parent to child levels.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    num_classes : int
        Number of output classes.
    num_levels : int
        Number of fractal levels (default 3, matching the paper).
    hidden_dims : list of int
        Hidden dimension for each level (decreasing, matching Table 1).
    num_blocks_list : list of int
        Number of transformer blocks per level (decreasing).
    num_heads_list : list of int
        Number of attention heads per level.
    mlp_ratio : float
        Feed-forward expansion ratio.
    dropout : float
        Dropout probability.
    attn_dropout : float
        Attention dropout probability.
    drop_path_rate : float
        Maximum drop-path rate across levels.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_levels: int = 3,
        hidden_dims: Optional[List[int]] = None,
        num_blocks_list: Optional[List[int]] = None,
        num_heads_list: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels

        # Default configurations matching the paper's decreasing capacity
        if hidden_dims is None:
            hidden_dims = self._default_dims(num_levels)
        if num_blocks_list is None:
            num_blocks_list = self._default_blocks(num_levels)
        if num_heads_list is None:
            num_heads_list = self._default_heads(num_levels)

        assert len(hidden_dims) == num_levels
        assert len(num_blocks_list) == num_levels
        assert len(num_heads_list) == num_levels

        self.hidden_dims = hidden_dims

        # Build fractal levels recursively
        self.levels = nn.ModuleList()
        for lvl in range(num_levels):
            cond_dim = hidden_dims[lvl - 1] if lvl > 0 else 0
            self.levels.append(
                FractalLevel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dims[lvl],
                    num_blocks=num_blocks_list[lvl],
                    num_heads=num_heads_list[lvl],
                    cond_dim=cond_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path_rate=drop_path_rate * (lvl + 1) / num_levels,
                )
            )

        # Aggregation: concatenate all level outputs, then project
        total_dim = sum(hidden_dims)
        self.aggregate_proj = nn.Sequential(
            nn.LayerNorm(total_dim, eps=1e-6),
            nn.Linear(total_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dims[0], eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], num_classes),
        )

        self._init_weights()

    @staticmethod
    def _default_dims(num_levels: int) -> List[int]:
        """Decreasing hidden dims (inspired by Table 1 ratios)."""
        base = 128
        return [base // (2 ** i) for i in range(num_levels)]

    @staticmethod
    def _default_blocks(num_levels: int) -> List[int]:
        """Decreasing block counts."""
        blocks = [4, 2, 1]
        while len(blocks) < num_levels:
            blocks.append(1)
        return blocks[:num_levels]

    @staticmethod
    def _default_heads(num_levels: int) -> List[int]:
        """Decreasing head counts."""
        heads = [8, 4, 2]
        while len(heads) < num_levels:
            heads.append(1)
        return heads[:num_levels]

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim) — input features.

        Returns
        -------
        logits : (batch, num_classes)
        """
        level_outputs = []
        cond = None

        for lvl, level in enumerate(self.levels):
            h = level(x, cond=cond)
            level_outputs.append(h)
            cond = h  # condition for the next level

        # aggregate all levels (fractal multi-scale aggregation)
        concat = torch.cat(level_outputs, dim=-1)
        agg = self.aggregate_proj(concat)

        # classify
        logits = self.classifier(agg)
        return logits

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    ADDITIONAL BASELINES                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class VanillaMLP(nn.Module):
    """Standard multi-layer perceptron baseline.

    No fractal structure, no self-similarity, no ToM components.
    Serves as a lower bound for comparison.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaTransformer(nn.Module):
    """Standard (non-fractal) transformer encoder for classification.

    Uses the same total depth and hidden dim as FractalGenNet but without
    the recursive multi-level structure.  This isolates the contribution
    of the fractal design.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            FractalTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = h.unsqueeze(1)  # (B, 1, D)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, 2, D)
        h = h + self.pos_embed

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        cls_out = h[:, 0]  # CLS token

        return self.classifier(cls_out)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           DEMO                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


if __name__ == "__main__":
    print("=" * 70)
    print("FractalGen Baseline — Smoke Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, D_in, C = 32, 36, 4

    # FractalGenNet
    model_fg = FractalGenNet(
        input_dim=D_in,
        num_classes=C,
        num_levels=3,
        hidden_dims=[120, 60, 30],
        num_blocks_list=[3, 2, 1],
        num_heads_list=[4, 2, 2],
        dropout=0.1,
    ).to(device)
    x = torch.randn(B, D_in, device=device)
    out = model_fg(x)
    print(f"FractalGenNet:        output={out.shape}, params={model_fg.param_count():,}")

    # VanillaMLP
    model_mlp = VanillaMLP(D_in, C, hidden_dim=128, num_layers=4).to(device)
    out_mlp = model_mlp(x)
    print(f"VanillaMLP:           output={out_mlp.shape}, params={model_mlp.param_count():,}")

    # VanillaTransformer
    model_vt = VanillaTransformer(D_in, C, hidden_dim=120, num_blocks=6).to(device)
    out_vt = model_vt(x)
    print(f"VanillaTransformer:   output={out_vt.shape}, params={model_vt.param_count():,}")

    # Backward pass
    loss = out.sum() + out_mlp.sum() + out_vt.sum()
    loss.backward()
    print("\nBackward pass: OK")
    print("=" * 70)
