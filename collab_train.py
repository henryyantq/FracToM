"""
collab_train.py

Recursive Social Reasoning Task for FracToM
============================================

A sophisticated multi-round social dilemma with hidden types, cheap-talk
signaling, nested beliefs, and deception — designed to fully exercise
FracToM's hierarchical mentalizing, BDI decomposition, epistemic gating,
and Bayesian belief revision.

Why the old task was too easy
-----------------------------
The previous 4-class task (SYNC_DIRECT / NEGOTIATE / LEADER_FOLLOW /
EXPLORE_SAFE) was solvable by checking whether cross-beliefs were
correct.  A simple MLP could learn the label from the one-hot goal
vectors and confidence scalars, making FracToM's recursive mentalizing
columns, BDI factoring, and epistemic gating unnecessary overhead.

Task definition (6-way classification)
--------------------------------------
Two agents (A and B) interact over 8 rounds in a resource-sharing game.
Each agent has a *hidden type* (cooperative, competitive, conditional, or
deceptive) that governs behaviour.  Agents send *signals* (cheap talk)
and then take *actions* (actual cooperation decisions) each round.

The observer sees everything *except* the true types, and must classify
the **ground-truth interaction state**:

0: TRANSPARENT_COOPERATE
   Both genuinely cooperative, beliefs accurate.
   → Surface pattern matching (ToM depth 0).

1: INFORMED_COOPERATE
   At least one conditional agent, but beliefs support cooperation.
   → Requires assessing belief-action alignment (depth 1).

2: DETECTED_DECEPTION
   A deceptive agent's inconsistency is visible (early-onset deception).
   → Requires other-modelling to spot signal-action gaps (depth 2).

3: HIDDEN_DECEPTION
   A deceptive agent with late-onset deception maintains high
   consistency.  Surface features mimic class 0.
   → Requires subtle trait analysis + late-round patterns (depth 2–3).

4: MUTUAL_DEFECTION
   Both competitive or cascading distrust → mutual defection.
   → Requires recognising antagonistic dynamics (depth 1–2).

5: RECURSIVE_BELIEF_ERROR
   Neither agent is deceptive, but wrong *second-order* beliefs cause
   defensive behaviour.  "I think you think I'm hostile → I hedge →
   you see me hedge → you also hedge → spiral."
   → Requires full recursive mentalizing (depth 3).

Why FracToM's inductive biases help
------------------------------------
* **BDI decomposition**: belief features, desire/trait features, and
  action/intention features naturally map to the three BDI subspaces.
* **Hierarchical mentalizing**: classes 0–5 require progressively deeper
  ToM reasoning — columns 0–3 each add value for different classes.
* **Epistemic gating**: the model should be uncertain about class 3
  (hidden deception looks like genuine cooperation on the surface).
* **Belief revision**: updating initial impressions based on multi-round
  action history mirrors Bayesian belief updating.
* **Perspective-shifting attention**: detecting deception (class 2/3)
  requires "putting yourself in the deceptive agent's shoes".
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nn import (
    FracToMLoss, FracToMNet, analyse_mentalizing_depth,
    extract_causal_graph, extract_bdi_activations,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CONSTANTS                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Agent hidden types
COOPERATIVE = 0
COMPETITIVE = 1
CONDITIONAL = 2
DECEPTIVE   = 3
NUM_TYPES   = 4
TYPE_NAMES  = ["COOPERATIVE", "COMPETITIVE", "CONDITIONAL", "DECEPTIVE"]

# Interaction rounds
NUM_ROUNDS = 8

# Classification labels (ordered by mentalizing depth required)
CLASS_NAMES = [
    "TRANSPARENT_COOPERATE",    # 0 — depth 0
    "INFORMED_COOPERATE",       # 1 — depth 1
    "DETECTED_DECEPTION",       # 2 — depth 2
    "HIDDEN_DECEPTION",         # 3 — depth 2–3
    "MUTUAL_DEFECTION",         # 4 — depth 1–2
    "RECURSIVE_BELIEF_ERROR",   # 5 — depth 3
]
NUM_CLASSES = len(CLASS_NAMES)

# Personality trait centroids per type (6-dim).
# DECEPTIVE is deliberately close to COOPERATIVE — the model must learn
# to distinguish them through *behavioural* evidence, not trait vectors.
_TRAIT_CENTROIDS = torch.tensor([
    [ 0.70,  0.80,  0.20,  0.30,  0.60,  0.10],  # COOPERATIVE
    [ 0.20,  0.10,  0.80,  0.70,  0.30,  0.60],  # COMPETITIVE
    [ 0.50,  0.60,  0.40,  0.40,  0.50,  0.30],  # CONDITIONAL
    [ 0.65,  0.72,  0.28,  0.38,  0.55,  0.18],  # DECEPTIVE (close to COOP!)
], dtype=torch.float32)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     DATASET CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

@dataclass
class DatasetConfig:
    """Configuration for the recursive social reasoning dataset."""

    n_samples: int = 12_000
    trait_noise_std: float = 0.18
    action_noise_std: float = 0.12
    signal_noise_std: float = 0.08
    belief_noise_std: float = 0.12
    feature_noise_std: float = 0.05
    seed: int = 7


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  BEHAVIOUR SIMULATION ENGINE                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _generate_traits(true_type: int, cfg: DatasetConfig,
                     gen: torch.Generator) -> Tensor:
    """Noisy personality trait vector from the type centroid."""
    centroid = _TRAIT_CENTROIDS[true_type]
    noise = cfg.trait_noise_std * torch.randn(6, generator=gen)
    return (centroid + noise).clamp(0.0, 1.0)


def _cooperative_action(rng: random.Random) -> float:
    return _clamp01(0.85 + 0.12 * rng.gauss(0, 1))


def _competitive_action(rng: random.Random) -> float:
    return _clamp01(0.15 + 0.12 * rng.gauss(0, 1))


def _conditional_action(round_idx: int, partner_prev: List[float],
                        rng: random.Random) -> float:
    if round_idx == 0:
        return _clamp01(0.78 + 0.10 * rng.gauss(0, 1))
    recent = sum(partner_prev[-2:]) / len(partner_prev[-2:])
    return _clamp01(recent + 0.08 * rng.gauss(0, 1))


def _deceptive_action(round_idx: int, onset: int,
                      rng: random.Random) -> float:
    if round_idx < onset:
        return _clamp01(0.83 + 0.09 * rng.gauss(0, 1))
    progress = (round_idx - onset) / max(1, NUM_ROUNDS - onset)
    base = 0.83 - 0.65 * progress
    return _clamp01(base + 0.10 * rng.gauss(0, 1))


def _generate_signal(true_type: int, action: float,
                     rng: random.Random) -> float:
    """Generate cheap-talk signal.  DECEPTIVE agents always signal high."""
    if true_type == COOPERATIVE:
        return _clamp01(action + 0.05 * rng.gauss(0, 1))
    elif true_type == COMPETITIVE:
        if rng.random() < 0.4:
            return _clamp01(0.55 + 0.15 * rng.gauss(0, 1))  # fake coop
        return _clamp01(action + 0.08 * rng.gauss(0, 1))
    elif true_type == CONDITIONAL:
        return _clamp01(action + 0.06 * rng.gauss(0, 1))
    else:  # DECEPTIVE — always signals cooperation
        return _clamp01(0.88 + 0.07 * rng.gauss(0, 1))


def simulate_episode(
    type_a: int,
    type_b: int,
    onset_a: int,
    onset_b: int,
    a_defensive: bool,
    b_defensive: bool,
    rng: random.Random,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Simulate NUM_ROUNDS of interaction.

    Returns (actions_a, actions_b, signals_a, signals_b).

    The ``defensive`` flag models the effect of wrong second-order beliefs:
    the agent gradually reduces cooperation mid-episode because it fears
    the partner views it as hostile.
    """
    actions_a: List[float] = []
    actions_b: List[float] = []
    signals_a: List[float] = []
    signals_b: List[float] = []

    for r in range(NUM_ROUNDS):
        # --- base actions from type-specific policies ---
        if type_a == COOPERATIVE:
            a_act = _cooperative_action(rng)
        elif type_a == COMPETITIVE:
            a_act = _competitive_action(rng)
        elif type_a == CONDITIONAL:
            a_act = _conditional_action(r, actions_b, rng)
        else:
            a_act = _deceptive_action(r, onset_a, rng)

        if type_b == COOPERATIVE:
            b_act = _cooperative_action(rng)
        elif type_b == COMPETITIVE:
            b_act = _competitive_action(rng)
        elif type_b == CONDITIONAL:
            b_act = _conditional_action(r, actions_a, rng)
        else:
            b_act = _deceptive_action(r, onset_b, rng)

        # --- defensive adjustment (wrong 2nd-order belief) ---
        # Kicks in gradually after round 3, creating a distinctive
        # declining-cooperation pattern that differs from both genuine
        # competition (low from start) and deception (signal stays high).
        if a_defensive and r >= 3:
            strength = 0.18 * (r - 2) / (NUM_ROUNDS - 3)
            a_act = a_act * (1.0 - strength) + 0.30 * strength
        if b_defensive and r >= 3:
            strength = 0.18 * (r - 2) / (NUM_ROUNDS - 3)
            b_act = b_act * (1.0 - strength) + 0.30 * strength

        a_act = _clamp01(a_act)
        b_act = _clamp01(b_act)

        # --- signals ---
        a_sig = _generate_signal(type_a, a_act, rng)
        b_sig = _generate_signal(type_b, b_act, rng)

        actions_a.append(a_act)
        actions_b.append(b_act)
        signals_a.append(a_sig)
        signals_b.append(b_sig)

    return actions_a, actions_b, signals_a, signals_b


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    BELIEF GENERATION                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _make_belief_vector(true_type: int, is_correct: bool,
                        cfg: DatasetConfig,
                        gen: torch.Generator) -> Tensor:
    """Soft probability vector over NUM_TYPES.

    If ``is_correct``, most mass concentrates on ``true_type``.
    Otherwise, mass concentrates on a random wrong type.
    """
    probs = torch.zeros(NUM_TYPES)

    if is_correct:
        probs[true_type] = 0.55 + 0.35 * torch.rand(1, generator=gen).item()
    else:
        wrong = [t for t in range(NUM_TYPES) if t != true_type]
        choice = wrong[int(torch.randint(0, len(wrong), (1,), generator=gen).item())]
        probs[choice] = 0.50 + 0.35 * torch.rand(1, generator=gen).item()

    # distribute remaining mass
    remaining = 1.0 - probs.sum()
    uniform_noise = torch.rand(NUM_TYPES, generator=gen)
    mask = probs == 0.0
    uniform_noise[~mask] = 0.0
    if uniform_noise.sum() > 0:
        probs = probs + remaining * uniform_noise / uniform_noise.sum()

    # add noise and re-normalise
    noise = cfg.belief_noise_std * torch.randn(NUM_TYPES, generator=gen)
    probs = (probs + noise).clamp(0.02, 1.0)
    probs = probs / probs.sum()
    return probs


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   FEATURE CONSTRUCTION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _pearson(x: List[float], y: List[float]) -> float:
    """Simple Pearson correlation (returns 0 for degenerate cases)."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / (sx * sy) ** 0.5


def _consistency(actions: List[float], signals: List[float]) -> float:
    return 1.0 - sum(abs(a - s) for a, s in zip(actions, signals)) / len(actions)


def build_feature_vector(
    declared_a: int,
    declared_b: int,
    traits_a: Tensor,
    traits_b: Tensor,
    actions_a: List[float],
    actions_b: List[float],
    signals_a: List[float],
    signals_b: List[float],
    belief_a: Tensor,
    belief_b: Tensor,
    sob_a: Tensor,
    sob_b: Tensor,
    conf_a_1st: float,
    conf_b_1st: float,
    conf_a_2nd: float,
    conf_b_2nd: float,
    env: Tensor,
    cfg: DatasetConfig,
    gen: torch.Generator,
) -> Tensor:
    """Build the ~91-dim feature vector from agent states.

    Feature groups
    --------------
    Declared types          : 2 × 4 = 8
    Personality traits      : 2 × 6 = 12
    Actions (per round)     : 2 × 8 = 16
    Signals (per round)     : 2 × 8 = 16
    1st-order beliefs       : 2 × 4 = 8
    2nd-order beliefs       : 2 × 4 = 8
    Consistency scores      : 2
    Cooperation rates       : 2
    Signal means            : 2
    Late−early gap change   : 2
    Cooperation trend       : 2
    Action reciprocity      : 1
    Action volatility       : 2
    Confidence scores       : 4
    Environmental context   : 6
    ─────────────────────────────
    Total                   : 91
    """
    # one-hot declared types
    decl_a = torch.zeros(NUM_TYPES); decl_a[declared_a] = 1.0
    decl_b = torch.zeros(NUM_TYPES); decl_b[declared_b] = 1.0

    act_a = torch.tensor(actions_a, dtype=torch.float32)
    act_b = torch.tensor(actions_b, dtype=torch.float32)
    sig_a = torch.tensor(signals_a, dtype=torch.float32)
    sig_b = torch.tensor(signals_b, dtype=torch.float32)

    # derived scalar features
    cons_a = torch.tensor([_consistency(actions_a, signals_a)])
    cons_b = torch.tensor([_consistency(actions_b, signals_b)])

    coop_a = torch.tensor([sum(actions_a) / NUM_ROUNDS])
    coop_b = torch.tensor([sum(actions_b) / NUM_ROUNDS])
    sig_mean_a = torch.tensor([sum(signals_a) / NUM_ROUNDS])
    sig_mean_b = torch.tensor([sum(signals_b) / NUM_ROUNDS])

    # late-vs-early gap change (key for detecting late-onset deception)
    mid = NUM_ROUNDS // 2
    early_gap_a = sum(abs(a - s) for a, s in zip(actions_a[:mid], signals_a[:mid])) / mid
    late_gap_a  = sum(abs(a - s) for a, s in zip(actions_a[mid:], signals_a[mid:])) / (NUM_ROUNDS - mid)
    early_gap_b = sum(abs(a - s) for a, s in zip(actions_b[:mid], signals_b[:mid])) / mid
    late_gap_b  = sum(abs(a - s) for a, s in zip(actions_b[mid:], signals_b[mid:])) / (NUM_ROUNDS - mid)
    gap_change_a = torch.tensor([late_gap_a - early_gap_a])
    gap_change_b = torch.tensor([late_gap_b - early_gap_b])

    # cooperation trend (correlation with round index)
    rounds = list(range(NUM_ROUNDS))
    trend_a = torch.tensor([_pearson(rounds, actions_a)])
    trend_b = torch.tensor([_pearson(rounds, actions_b)])

    reciprocity = torch.tensor([_pearson(actions_a, actions_b)])

    vol_a = torch.tensor([sum(abs(actions_a[i+1] - actions_a[i])
                              for i in range(NUM_ROUNDS - 1)) / (NUM_ROUNDS - 1)])
    vol_b = torch.tensor([sum(abs(actions_b[i+1] - actions_b[i])
                              for i in range(NUM_ROUNDS - 1)) / (NUM_ROUNDS - 1)])

    confs = torch.tensor([conf_a_1st, conf_b_1st, conf_a_2nd, conf_b_2nd])

    feature = torch.cat([
        decl_a, decl_b,            # 8
        traits_a, traits_b,        # 12
        act_a, act_b,              # 16
        sig_a, sig_b,              # 16
        belief_a, belief_b,        # 8
        sob_a, sob_b,              # 8
        cons_a, cons_b,            # 2
        coop_a, coop_b,            # 2
        sig_mean_a, sig_mean_b,    # 2
        gap_change_a, gap_change_b,# 2
        trend_a, trend_b,          # 2
        reciprocity,               # 1
        vol_a, vol_b,              # 2
        confs,                     # 4
        env,                       # 6
    ])                             # = 91

    noise = cfg.feature_noise_std * torch.randn(feature.shape[0], generator=gen)
    return feature + noise


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     DATASET CLASS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

class RecursiveSocialDataset(Dataset):
    """Multi-round social dilemma with hidden types, deception, and nested
    beliefs.  Each sample is a complete 8-round episode between two agents.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.features, self.labels = self._generate_all()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        return self.features.shape[-1]

    # ------------------------------------------------------------------ gen
    def _generate_all(self) -> Tuple[Tensor, Tensor]:
        rng = random.Random(self.cfg.seed)
        gen = torch.Generator().manual_seed(self.cfg.seed)

        per_class = self.cfg.n_samples // NUM_CLASSES
        remainder = self.cfg.n_samples % NUM_CLASSES

        all_feats: List[Tensor] = []
        all_labels: List[int] = []

        for cls in range(NUM_CLASSES):
            n = per_class + (1 if cls < remainder else 0)
            for _ in range(n):
                feat, label = self._generate_one(cls, rng, gen)
                all_feats.append(feat)
                all_labels.append(label)

        feats = torch.stack(all_feats)
        labels = torch.tensor(all_labels, dtype=torch.long)
        perm = torch.randperm(len(labels), generator=gen)
        return feats[perm], labels[perm]

    def _generate_one(self, cls: int, rng: random.Random,
                      gen: torch.Generator) -> Tuple[Tensor, int]:
        # 1) agent types consistent with class
        type_a, type_b = self._sample_types(cls, rng)

        # 2) deception onset
        onset_a = self._deception_onset(type_a, cls, rng)
        onset_b = self._deception_onset(type_b, cls, rng)

        # 3) defensive flags (only class 5)
        a_def = cls == 5
        b_def = cls == 5

        # 4) personality traits
        traits_a = _generate_traits(type_a, self.cfg, gen)
        traits_b = _generate_traits(type_b, self.cfg, gen)

        # 5) declared types (DECEPTIVE lies)
        declared_a = COOPERATIVE if type_a == DECEPTIVE else type_a
        declared_b = COOPERATIVE if type_b == DECEPTIVE else type_b

        # 6) simulate episode
        acts_a, acts_b, sigs_a, sigs_b = simulate_episode(
            type_a, type_b, onset_a, onset_b, a_def, b_def, rng,
        )

        # 7) first-order beliefs
        b1a_correct, b1b_correct = self._belief_accuracy_1st(cls, rng)
        belief_a = _make_belief_vector(type_b, b1a_correct, self.cfg, gen)
        belief_b = _make_belief_vector(type_a, b1b_correct, self.cfg, gen)

        # 8) second-order beliefs
        b2a_correct, b2b_correct = self._belief_accuracy_2nd(cls, rng)
        sob_a = _make_belief_vector(type_a, b2a_correct, self.cfg, gen)
        sob_b = _make_belief_vector(type_b, b2b_correct, self.cfg, gen)

        # 9) confidence scores
        conf_a_1st = 0.70 + 0.25 * rng.random() if b1a_correct else 0.25 + 0.30 * rng.random()
        conf_b_1st = 0.70 + 0.25 * rng.random() if b1b_correct else 0.25 + 0.30 * rng.random()
        conf_a_2nd = 0.50 + 0.30 * rng.random() if b2a_correct else 0.20 + 0.25 * rng.random()
        conf_b_2nd = 0.50 + 0.30 * rng.random() if b2b_correct else 0.20 + 0.25 * rng.random()

        # 10) environment
        stakes   = torch.rand(1, generator=gen)
        pressure = torch.rand(1, generator=gen)
        progress = torch.tensor([1.0])
        payoff_a = torch.tensor([sum(acts_a) * 0.5 + sum(acts_b) * 0.3])
        payoff_b = torch.tensor([sum(acts_b) * 0.5 + sum(acts_a) * 0.3])
        mutual   = torch.tensor([
            sum(a * b for a, b in zip(acts_a, acts_b)) / NUM_ROUNDS
        ])
        env = torch.cat([stakes, pressure, progress, payoff_a, payoff_b, mutual])

        feat = build_feature_vector(
            declared_a, declared_b, traits_a, traits_b,
            acts_a, acts_b, sigs_a, sigs_b,
            belief_a, belief_b, sob_a, sob_b,
            conf_a_1st, conf_b_1st, conf_a_2nd, conf_b_2nd,
            env, self.cfg, gen,
        )
        return feat, cls

    # --------------------------------------------------------- type sampling
    def _sample_types(self, cls: int, rng: random.Random) -> Tuple[int, int]:
        if cls == 0:  # TRANSPARENT_COOPERATE
            return COOPERATIVE, COOPERATIVE
        elif cls == 1:  # INFORMED_COOPERATE
            return rng.choice([
                (COOPERATIVE, CONDITIONAL),
                (CONDITIONAL, COOPERATIVE),
                (CONDITIONAL, CONDITIONAL),
            ])
        elif cls == 2:  # DETECTED_DECEPTION
            other = rng.choice([COOPERATIVE, CONDITIONAL])
            return (other, DECEPTIVE) if rng.random() < 0.5 else (DECEPTIVE, other)
        elif cls == 3:  # HIDDEN_DECEPTION
            other = rng.choice([COOPERATIVE, CONDITIONAL])
            return (other, DECEPTIVE) if rng.random() < 0.5 else (DECEPTIVE, other)
        elif cls == 4:  # MUTUAL_DEFECTION
            return rng.choice([
                (COMPETITIVE, COMPETITIVE),
                (COMPETITIVE, rng.choice([COOPERATIVE, CONDITIONAL, DECEPTIVE])),
                (rng.choice([COOPERATIVE, CONDITIONAL, DECEPTIVE]), COMPETITIVE),
            ])
        else:  # cls == 5: RECURSIVE_BELIEF_ERROR
            return rng.choice([
                (COOPERATIVE, COOPERATIVE),
                (COOPERATIVE, CONDITIONAL),
                (CONDITIONAL, COOPERATIVE),
                (CONDITIONAL, CONDITIONAL),
            ])

    # ----------------------------------------------- deception onset timing
    def _deception_onset(self, agent_type: int, cls: int,
                         rng: random.Random) -> int:
        if agent_type != DECEPTIVE:
            return NUM_ROUNDS  # no deception
        if cls == 2:  # DETECTED: early onset → visible inconsistency
            return rng.randint(2, 4)
        if cls == 3:  # HIDDEN: late onset → hard to spot
            return rng.randint(6, 7)
        return rng.randint(3, 6)

    # ----------------------------------------------- belief accuracy
    def _belief_accuracy_1st(self, cls: int,
                             rng: random.Random) -> Tuple[bool, bool]:
        if cls == 0:
            return True, True
        elif cls == 1:
            return True, rng.random() < 0.80
        elif cls == 2:  # A detects B's deception
            return True, rng.random() < 0.50
        elif cls == 3:  # A does NOT detect
            return False, rng.random() < 0.50
        elif cls == 4:
            return rng.random() < 0.55, rng.random() < 0.55
        else:  # cls == 5 — first-order beliefs are CORRECT
            return True, True

    def _belief_accuracy_2nd(self, cls: int,
                             rng: random.Random) -> Tuple[bool, bool]:
        if cls == 0:
            return True, True
        elif cls == 1:
            return rng.random() < 0.65, rng.random() < 0.65
        elif cls in (2, 3):
            return rng.random() < 0.45, rng.random() < 0.45
        elif cls == 4:
            return rng.random() < 0.40, rng.random() < 0.40
        else:  # cls == 5 — second-order beliefs WRONG (key signal)
            return False, False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  EVALUATION UTILITIES                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def class_distribution(labels: Tensor) -> Dict[int, int]:
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    return {i: int(counts[i].item()) for i in range(NUM_CLASSES)}


def evaluate(
    model: FracToMNet,
    criterion: FracToMLoss,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float], Tensor]:
    """Returns (loss, accuracy, loss_breakdown, confusion_matrix)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    breakdown_acc: Dict[str, float] = {
        "task": 0.0, "bdi_consistency": 0.0,
        "uncertainty_cal": 0.0, "depth_entropy_reg": 0.0,
        "dag_penalty": 0.0, "causal_sparsity": 0.0,
        "cf_ordering": 0.0, "aux_deepsup": 0.0, "total": 0.0,
    }
    all_preds: List[Tensor] = []
    all_tgts:  List[Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, report = model(xb, return_interpretability=True)
            loss, bd = criterion(logits, yb, report)

            bs = yb.shape[0]
            total_loss += loss.item() * bs
            preds = logits.argmax(-1)
            correct += int((preds == yb).sum().item())
            count += bs
            for k in breakdown_acc:
                breakdown_acc[k] += bd[k] * bs
            all_preds.append(preds.cpu())
            all_tgts.append(yb.cpu())

    avg_loss = total_loss / max(1, count)
    acc = correct / max(1, count)
    avg_bd = {k: v / max(1, count) for k, v in breakdown_acc.items()}

    preds_cat = torch.cat(all_preds)
    tgts_cat  = torch.cat(all_tgts)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(tgts_cat, preds_cat):
        cm[t.item(), p.item()] += 1
    return avg_loss, acc, avg_bd, cm


def print_confusion_matrix(cm: Tensor) -> None:
    max_name = max(len(n) for n in CLASS_NAMES)
    header = " " * (max_name + 4) + " ".join([f"P{i:>4d}" for i in range(NUM_CLASSES)])
    print(header)
    for i in range(NUM_CLASSES):
        row = " ".join([f"{cm[i, j].item():>5d}" for j in range(NUM_CLASSES)])
        print(f"T{i:>1d} {CLASS_NAMES[i]:>{max_name}s} {row}")


def print_per_class_accuracy(cm: Tensor) -> None:
    print("\nPer-class accuracy:")
    for i in range(NUM_CLASSES):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc = correct / max(1, total)
        bar = "█" * int(acc * 30)
        print(f"  {CLASS_NAMES[i]:>25s}: {acc:.3f}  {bar}  ({correct}/{total})")

    # highlight hard pairs
    print("\nKey confusion pairs (hardest distinctions):")
    for (a, b), desc in [
        ((0, 3), "TRANSPARENT_COOPERATE ↔ HIDDEN_DECEPTION  (false-belief test)"),
        ((0, 5), "TRANSPARENT_COOPERATE ↔ RECURSIVE_BELIEF   (recursive ToM)"),
        ((2, 3), "DETECTED ↔ HIDDEN DECEPTION               (deception depth)"),
    ]:
        c_ab = cm[a, b].item()
        c_ba = cm[b, a].item()
        t_a = cm[a].sum().item()
        t_b = cm[b].sum().item()
        print(f"  {desc}")
        print(f"    class {a}→{b}: {c_ab}/{t_a} ({c_ab/max(1,t_a):.3f})   "
              f"class {b}→{a}: {c_ba}/{t_b} ({c_ba/max(1,t_b):.3f})")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     VISUALIZATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def save_tom_hierarchy_visualization(
    depth_weights: Tensor,
    sigma_map: Dict[int, Tensor],
    output_dir: str,
    max_samples_heatmap: int = 120,
) -> None:
    """Save ToM depth-usage distribution and epistemic uncertainty plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Visualization skipped: pip install matplotlib")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alpha = depth_weights.cpu()
    n, K = alpha.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        alpha.argmax(-1).numpy(),
        bins=[i - 0.5 for i in range(K + 1)],
        edgecolor="black", rwidth=0.9,
    )
    axes[0].set_xticks(list(range(K)))
    axes[0].set_xlabel("Dominant ToM depth")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dominant Theory-of-Mind Depth Distribution")

    hc = min(max_samples_heatmap, n)
    axes[1].imshow(alpha[:hc].numpy(), aspect="auto", cmap="viridis")
    axes[1].set_xlabel("ToM depth")
    axes[1].set_ylabel("Sample index")
    axes[1].set_title(f"Per-sample ToM depth weights (first {hc})")
    plt.tight_layout()
    fig.savefig(out / "tom_depth_weights.png", dpi=160)
    plt.close(fig)

    if sigma_map:
        levels = sorted(sigma_map.keys())
        means = [float(sigma_map[k].mean()) for k in levels]
        fig2, ax2 = plt.subplots(figsize=(8, 4.8))
        ax2.bar(levels, means)
        ax2.set_xlabel("ToM depth")
        ax2.set_ylabel("Mean epistemic uncertainty (σ)")
        ax2.set_title("Epistemic Uncertainty by ToM Depth")
        ax2.set_xticks(levels)
        plt.tight_layout()
        fig2.savefig(out / "tom_uncertainty.png", dpi=160)
        plt.close(fig2)

    print(f"Saved ToM hierarchy visualizations to: {out.resolve()}")


def save_causal_graph_visualization(
    report,
    output_dir: str,
) -> None:
    """Save causal graph visualizations: adjacency heatmap + Pearl hierarchy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Causal visualization skipped: pip install matplotlib")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cg = extract_causal_graph(report)

    # --- 1. BDI adjacency heatmap ---
    if cg["bdi_adjacency"] is not None:
        adj = cg["bdi_adjacency"].cpu().numpy()
        bdi_labels = ["Observation", "Belief", "Desire", "Intention"]
        n_bdi = adj.shape[0]
        labels = bdi_labels[:n_bdi] if n_bdi <= len(bdi_labels) else [
            f"V{i}" for i in range(n_bdi)
        ]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(adj, cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n_bdi))
        ax.set_yticks(range(n_bdi))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n_bdi):
            for j in range(n_bdi):
                ax.text(j, i, f"{adj[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if adj[i, j] > 0.5 else "black")
        ax.set_title("BDI Causal Adjacency (SCM)")
        ax.set_xlabel("Effect")
        ax.set_ylabel("Cause")
        fig.colorbar(im, ax=ax, shrink=0.75)
        plt.tight_layout()
        fig.savefig(out / "causal_bdi_adjacency.png", dpi=160)
        plt.close(fig)

    # --- 2. Pearl's Causal Hierarchy bar chart ---
    if cg["hierarchy_weights"]:
        cols = sorted(cg["hierarchy_weights"].keys())
        levels = ["Association", "Intervention", "Counterfactual"]
        weights_arr = torch.stack([cg["hierarchy_weights"][c] for c in cols]).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(cols))
        width = 0.25
        for i, lvl in enumerate(levels):
            ax.bar([xi + i * width for xi in x], weights_arr[:, i],
                   width=width, label=lvl)
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels([f"Col {c}" for c in cols])
        ax.set_ylabel("Weight")
        ax.set_title("Pearl's Causal Hierarchy Routing per ToM Depth")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out / "causal_pearl_hierarchy.png", dpi=160)
        plt.close(fig)

    # --- 3. Cross-depth adjacency ---
    if cg["cross_depth_adjacency"] is not None:
        cross = cg["cross_depth_adjacency"].cpu().numpy()
        n_cols = cross.shape[0]
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(cross, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_cols))
        ax.set_xticklabels([f"Col {i}" for i in range(n_cols)])
        ax.set_yticklabels([f"Col {i}" for i in range(n_cols)])
        for i in range(n_cols):
            for j in range(n_cols):
                ax.text(j, i, f"{cross[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if cross[i, j] > 0.5 else "black")
        ax.set_title("Cross-Depth Causal Structure")
        ax.set_xlabel("Effect depth")
        ax.set_ylabel("Cause depth")
        fig.colorbar(im, ax=ax, shrink=0.75)
        plt.tight_layout()
        fig.savefig(out / "causal_cross_depth.png", dpi=160)
        plt.close(fig)

    print(f"Saved causal graph visualizations to: {out.resolve()}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      TRAINING LOOP                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def train(args: argparse.Namespace) -> None:
    # Select device: prefer MPS (macOS Metal) > CUDA > CPU
    if not args.cpu:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = "cpu"

    # ─── dataset ──────────────────────────────────────────────────────
    cfg = DatasetConfig(
        n_samples=args.samples,
        trait_noise_std=args.trait_noise,
        action_noise_std=args.action_noise,
        signal_noise_std=args.signal_noise,
        belief_noise_std=args.belief_noise,
        feature_noise_std=args.feature_noise,
        seed=args.seed,
    )
    dataset = RecursiveSocialDataset(cfg)

    n_train = int(len(dataset) * args.train_ratio)
    n_test  = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    train_labels = torch.tensor(
        [dataset.labels[i] for i in train_set.indices], dtype=torch.long,
    )
    test_labels = torch.tensor(
        [dataset.labels[i] for i in test_set.indices], dtype=torch.long,
    )

    print("=" * 90)
    print("FracToM — Recursive Social Reasoning Task")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Samples: total={len(dataset)} train={n_train} test={n_test}")
    print(f"Input dim: {dataset.input_dim}")
    print(f"Classes ({NUM_CLASSES}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print(f"\nTrain distribution: {class_distribution(train_labels)}")
    print(f"Test  distribution: {class_distribution(test_labels)}")
    print()

    # ─── model ────────────────────────────────────────────────────────
    model = FracToMNet(
        input_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        mentalizing_depth=args.depth,
        num_bdi_factors=3,
        blocks_per_column=args.blocks,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        drop_path=args.drop_path,
        num_classes=NUM_CLASSES,
        causal_model=args.causal_model,
        causal_noise_dim=args.causal_noise_dim,
        capacity_schedule=args.capacity_schedule,
        guiding_belief=args.guiding_belief,
        gist_dim=args.gist_dim,
        auxiliary_heads=args.auxiliary_heads,
    ).to(device)

    criterion = FracToMLoss(
        task_loss_fn=nn.CrossEntropyLoss(),
        lambda_bdi=args.lambda_bdi,
        lambda_uncertainty=args.lambda_uncertainty,
        lambda_depth_entropy=args.lambda_depth_entropy,
        lambda_dag=args.lambda_dag,
        lambda_causal_sparsity=args.lambda_causal_sparsity,
        lambda_counterfactual=args.lambda_counterfactual,
        lambda_auxiliary=args.lambda_auxiliary,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Column dims: {model.column_dims}")
    print(f"Capacity schedule: {args.capacity_schedule}")
    print(f"Guiding belief: {args.guiding_belief}  gist_dim={args.gist_dim}")
    print(f"Auxiliary heads: {args.auxiliary_heads}  λ_aux={args.lambda_auxiliary}")
    print(f"Causal model: {args.causal_model}")
    if args.causal_model:
        causal_params = (
            sum(p.numel() for p in model.scm.parameters())
            + sum(p.numel() for p in model.causal_router.parameters())
            + sum(p.numel() for p in model.causal_discovery.parameters())
        )
        print(f"  SCM params: {causal_params:,}")
        print(f"  Causal noise dim: {args.causal_noise_dim}")
        print(f"  λ_dag={args.lambda_dag}  λ_sparse={args.lambda_causal_sparsity}  λ_cf={args.lambda_counterfactual}")
    print("Starting training...\n")

    # ─── train loop ───────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        curriculum = max(0.0, 1.0 - epoch / args.epochs)
        model.set_curriculum(curriculum)

        run_loss = 0.0
        run_correct = 0
        run_count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, report = model(xb, return_interpretability=True)
            loss, _ = criterion(logits, yb, report)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = yb.shape[0]
            run_loss += loss.item() * bs
            run_correct += int((logits.argmax(-1) == yb).sum().item())
            run_count += bs

        scheduler.step()
        tr_loss = run_loss / max(1, run_count)
        tr_acc  = run_correct / max(1, run_count)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            te_loss, te_acc, bd, _ = evaluate(
                model, criterion, test_loader, device,
            )
            best_acc = max(best_acc, te_acc)
            print(
                f"Epoch {epoch:03d} | cur={curriculum:.3f} | "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.3f} best={best_acc:.3f}"
            )
            causal_parts = ""
            if args.causal_model:
                causal_parts = (
                    f" dag={bd.get('dag_penalty', 0):.4f}"
                    f" sparse={bd.get('causal_sparsity', 0):.4f}"
                    f" cf={bd.get('cf_ordering', 0):.4f}"
                )
            aux_part = ""
            if args.auxiliary_heads:
                aux_part = f" aux={bd.get('aux_deepsup', 0):.4f}"
            print(
                f"          loss → task={bd['task']:.4f} "
                f"bdi={bd['bdi_consistency']:.4f} "
                f"unc={bd['uncertainty_cal']:.4f} "
                f"depth={bd['depth_entropy_reg']:.4f}"
                f"{causal_parts}{aux_part}"
            )

    print("\nTraining complete.")

    # ─── final evaluation ─────────────────────────────────────────────
    final_loss, final_acc, final_bd, cm = evaluate(
        model, criterion, test_loader, device,
    )

    print("\n" + "=" * 90)
    print("Final Evaluation")
    print("=" * 90)
    print(f"Test loss : {final_loss:.4f}")
    print(f"Test acc  : {final_acc:.4f}")
    print(f"Best acc  : {best_acc:.4f}")
    print(f"Loss parts: {{{', '.join(f'{k}: {v:.5f}' for k, v in final_bd.items())}}}")

    print("\nConfusion Matrix (rows = true, cols = pred):")
    print_confusion_matrix(cm)
    print_per_class_accuracy(cm)

    # ─── interpretability ─────────────────────────────────────────────
    model.eval()
    all_dw: List[Tensor] = []
    sigma_acc: Dict[int, List[Tensor]] = {}
    last_report = None

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            _, report = model(xb, return_interpretability=True)
            all_dw.append(report.depth_weights.cpu())
            for k, s in report.column_uncertainties.items():
                sigma_acc.setdefault(k, []).append(s.cpu())
            last_report = report

    dw = torch.cat(all_dw, dim=0)

    # Build a report-like object for analyse_mentalizing_depth that
    # contains the causal fields from the last batch for display.
    from nn import InterpretabilityReport
    summary_report = InterpretabilityReport(
        depth_weights=dw,
        bdi_states={},
        column_uncertainties={k: torch.cat(v) for k, v in sigma_acc.items()},
        causal_adjacency=(
            last_report.causal_adjacency if last_report else None
        ),
        causal_hierarchy_weights=(
            last_report.causal_hierarchy_weights if last_report else None
        ),
        cross_depth_adjacency=(
            last_report.cross_depth_adjacency if last_report else None
        ),
        dag_penalty=(
            last_report.dag_penalty if last_report else None
        ),
        counterfactual_distances=(
            last_report.counterfactual_distances if last_report else None
        ),
        auxiliary_logits=(
            last_report.auxiliary_logits if last_report else None
        ),
        guiding_gists=(
            last_report.guiding_gists if last_report else None
        ),
        column_dims=(
            last_report.column_dims if last_report else None
        ),
    )

    print("\n" + "=" * 90)
    print("Mentalizing & Causal Interpretability Summary")
    print("=" * 90)
    print(analyse_mentalizing_depth(summary_report))

    # Causal graph analysis
    if args.causal_model and last_report is not None:
        cg = extract_causal_graph(last_report)
        print("\n" + "-" * 60)
        print("Causal Graph Summary (last test batch)")
        print("-" * 60)
        if cg["bdi_edges"]:
            print("BDI causal edges (strength > 0.3):")
            for src, tgt, w in cg["bdi_edges"]:
                arrow = "━━▶" if w > 0.7 else "──▶" if w > 0.5 else "╌╌▶"
                print(f"  {src:>9s} {arrow} {tgt:<9s}  ({w:.3f})")
        if cg["cross_depth_edges"]:
            print("\nCross-depth causal edges:")
            for src, tgt, w in cg["cross_depth_edges"]:
                print(f"  Column {src} → Column {tgt}:  {w:.3f}")
        if cg["hierarchy_weights"]:
            print("\nPearl's Causal Hierarchy per column:")
            level_names = ["Association", "Intervention", "Counterfactual"]
            for k, w in sorted(cg["hierarchy_weights"].items()):
                dominant = level_names[w.argmax().item()]
                parts = "  ".join(
                    f"{level_names[i][:5]}: {w[i]:.3f}" for i in range(3)
                )
                print(f"  Column {k}: {parts}  [{dominant}]")
        if cg["counterfactual_distances"]:
            print("\nCounterfactual distances (larger = richer CF reasoning):")
            for k, d in sorted(cg["counterfactual_distances"].items()):
                bar = "█" * int(d * 5)
                print(f"  Column {k}: {d:.4f}  {bar}")
        dag = cg.get("dag_penalty")
        if dag is not None:
            print(f"\nDAG penalty (0 = valid DAG): {dag:.6f}")

    save_tom_hierarchy_visualization(
        dw, summary_report.column_uncertainties,
        args.viz_dir, args.viz_heatmap_samples,
    )

    # FractalGen-inspired enhancements diagnostics
    if last_report is not None:
        print("\n" + "-" * 60)
        print("FractalGen-Inspired Enhancements")
        print("-" * 60)
        if last_report.column_dims is not None:
            print("Per-column capacity (capacity_schedule="
                  f"{args.capacity_schedule}):")
            for k, d in enumerate(last_report.column_dims):
                print(f"  Column {k}: dim={d}")
        if last_report.auxiliary_logits is not None:
            print("\nAuxiliary head accuracy (deep supervision):")
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    _, rep = model(xb, return_interpretability=True)
                    if rep.auxiliary_logits:
                        for k, al in sorted(rep.auxiliary_logits.items()):
                            acc = (al.argmax(-1) == yb).float().mean().item()
                            print(f"  Column {k}: {acc:.3f}")
                    break  # one batch suffices
        if last_report.guiding_gists is not None:
            print("\nGuiding belief gist stats (γ, β norms):")
            for k, (gamma, beta) in sorted(
                last_report.guiding_gists.items()
            ):
                print(
                    f"  Column {k}: ‖γ‖={gamma.norm():.4f}  "
                    f"‖β‖={beta.norm():.4f}"
                )

    # Save causal graph visualization
    if args.causal_model and last_report is not None:
        save_causal_graph_visualization(
            last_report, args.viz_dir,
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CLI                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train FracToM on a recursive social reasoning task"
    )
    # data
    p.add_argument("--samples",        type=int,   default=12000)
    p.add_argument("--trait-noise",    type=float, default=0.18)
    p.add_argument("--action-noise",   type=float, default=0.12)
    p.add_argument("--signal-noise",   type=float, default=0.08)
    p.add_argument("--belief-noise",   type=float, default=0.12)
    p.add_argument("--feature-noise",  type=float, default=0.05)
    p.add_argument("--train-ratio",    type=float, default=0.8)

    # model
    p.add_argument("--hidden-dim",     type=int,   default=120)
    p.add_argument("--depth",          type=int,   default=3)
    p.add_argument("--blocks",         type=int,   default=1)
    p.add_argument("--heads",          type=int,   default=4)
    p.add_argument("--ff-mult",        type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--drop-path",      type=float, default=0.1)

    # optimisation
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-2)
    p.add_argument("--eval-every",     type=int,   default=5)

    # loss weights
    p.add_argument("--lambda-bdi",            type=float, default=0.01)
    p.add_argument("--lambda-uncertainty",    type=float, default=0.005)
    p.add_argument("--lambda-depth-entropy",  type=float, default=0.01)
    p.add_argument("--lambda-dag",            type=float, default=0.1)
    p.add_argument("--lambda-causal-sparsity",type=float, default=0.005)
    p.add_argument("--lambda-counterfactual", type=float, default=0.01)

    # FractalGen-inspired enhancements
    p.add_argument("--capacity-schedule", type=str, default="decreasing",
                   choices=["uniform", "decreasing"],
                   help="Per-column dim schedule (default: decreasing)")
    p.add_argument("--guiding-belief",   action="store_true", default=True,
                   help="Enable guiding belief FiLM module (default: on)")
    p.add_argument("--no-guiding-belief", dest="guiding_belief",
                   action="store_false",
                   help="Disable guiding belief FiLM module")
    p.add_argument("--gist-dim",         type=int, default=32,
                   help="Gist dim for guiding belief module")
    p.add_argument("--auxiliary-heads",  action="store_true", default=True,
                   help="Enable auxiliary deep supervision heads (default: on)")
    p.add_argument("--no-auxiliary-heads", dest="auxiliary_heads",
                   action="store_false",
                   help="Disable auxiliary deep supervision heads")
    p.add_argument("--lambda-auxiliary",  type=float, default=0.1,
                   help="Weight for auxiliary deep supervision loss")

    # causal model
    p.add_argument("--causal-model",   action="store_true", default=True,
                   help="Enable Structural Causal Model (default: on)")
    p.add_argument("--no-causal-model", dest="causal_model", action="store_false",
                   help="Disable Structural Causal Model")
    p.add_argument("--causal-noise-dim", type=int, default=16,
                   help="Exogenous noise dim for SCM structural equations")

    # misc
    p.add_argument("--seed",           type=int,   default=7)
    p.add_argument("--cpu",            action="store_true")
    p.add_argument("--viz-dir",        type=str,   default="visualizations")
    p.add_argument("--viz-heatmap-samples", type=int, default=120)

    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    from math import gcd
    _quantum = (3 * args.heads) // gcd(3, args.heads)
    if args.hidden_dim % _quantum != 0:
        raise ValueError(
            f"--hidden-dim must be divisible by lcm(num_bdi_factors=3, "
            f"num_heads={args.heads}) = {_quantum}"
        )
    if args.samples < 600:
        raise ValueError("--samples should be ≥ 600 for 6-class balance")

    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
