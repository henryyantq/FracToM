"""
collab_train.py

Dual-agent collaboration training script for FracToM.

What this script provides
-------------------------
1) A random generative dataset that simulates *dual-agent collaboration*.
2) A complete FracToM training/evaluation pipeline.
3) Interpretable outputs (mentalizing-depth usage + uncertainty stats).

Task definition (4-way classification)
--------------------------------------
Given two agents A/B with:
- private goals (true intent),
- noisy beliefs about each other's goals,
- confidence in those beliefs,
- shared context and resource pressure,
predict the best collaboration policy class:

0: SYNC_DIRECT      (both share same goal)
1: NEGOTIATE        (both have correct cross-beliefs but goals differ)
2: LEADER_FOLLOW    (exactly one side has correct cross-belief)
3: EXPLORE_SAFE     (both cross-beliefs are incorrect)

This is synthetic but structured to require combining self + other-model cues,
which matches FracToM's design objective.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nn import FracToMLoss, FracToMNet, analyse_mentalizing_depth


CLASS_NAMES = [
    "SYNC_DIRECT",
    "NEGOTIATE",
    "LEADER_FOLLOW",
    "EXPLORE_SAFE",
]


@dataclass
class DatasetConfig:
    """Configuration for synthetic dual-agent collaboration data."""

    n_samples: int = 6000
    num_goals: int = 6
    noise_prob: float = 0.33
    feature_noise_std: float = 0.08
    seed: int = 7


class DualAgentCollabDataset(Dataset):
    """Random generative dataset for dual-agent collaboration.

    Each sample contains two agents with private goals and noisy beliefs
    about each other. The target class depends on *joint* reasoning quality.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.features, self.labels = self._generate_dataset()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        return self.features.shape[-1]

    def _one_hot(self, idx: int, size: int) -> Tensor:
        v = torch.zeros(size)
        v[idx] = 1.0
        return v

    def _noisy_observation(self, true_goal: int, num_goals: int, noise_prob: float, rng: random.Random) -> Tuple[int, int]:
        """Return (observed_goal, is_correct)."""
        if rng.random() < noise_prob:
            wrong_candidates = [g for g in range(num_goals) if g != true_goal]
            observed = rng.choice(wrong_candidates)
            return observed, 0
        return true_goal, 1

    def _target_class(
        self,
        goal_a: int,
        goal_b: int,
        a_correct_about_b: int,
        b_correct_about_a: int,
    ) -> int:
        if goal_a == goal_b:
            return 0  # SYNC_DIRECT
        if a_correct_about_b == 1 and b_correct_about_a == 1:
            return 1  # NEGOTIATE
        if a_correct_about_b + b_correct_about_a == 1:
            return 2  # LEADER_FOLLOW
        return 3      # EXPLORE_SAFE

    def _generate_sample(self, rng: random.Random, torch_gen: torch.Generator) -> Tuple[Tensor, int]:
        num_goals = self.cfg.num_goals

        # True private goals of each agent
        goal_a = rng.randrange(num_goals)
        goal_b = rng.randrange(num_goals)

        # Noisy cross-observations
        obs_a_about_b, a_correct_about_b = self._noisy_observation(
            goal_b, num_goals, self.cfg.noise_prob, rng
        )
        obs_b_about_a, b_correct_about_a = self._noisy_observation(
            goal_a, num_goals, self.cfg.noise_prob, rng
        )

        # Shared environment context
        shared_context = torch.randn(6, generator=torch_gen)
        resource_pressure = torch.rand(1, generator=torch_gen)
        time_pressure = torch.rand(1, generator=torch_gen)

        # Agent-specific confidence signals (calibrated with correctness but noisy)
        conf_a = torch.tensor([
            0.75 + 0.2 * rng.random() if a_correct_about_b else 0.15 + 0.35 * rng.random()
        ])
        conf_b = torch.tensor([
            0.75 + 0.2 * rng.random() if b_correct_about_a else 0.15 + 0.35 * rng.random()
        ])

        # One-hot goal and observed-goal encodings
        a_goal_oh = self._one_hot(goal_a, num_goals)
        b_goal_oh = self._one_hot(goal_b, num_goals)
        a_obs_oh = self._one_hot(obs_a_about_b, num_goals)
        b_obs_oh = self._one_hot(obs_b_about_a, num_goals)

        # Additional latent collaboration descriptors
        # (induce richer signal geometry while preserving label semantics)
        goal_gap = torch.tensor([abs(goal_a - goal_b) / max(1, num_goals - 1)], dtype=torch.float32)
        mutual_certainty = torch.tensor([(conf_a.item() + conf_b.item()) / 2.0], dtype=torch.float32)

        feature = torch.cat(
            [
                a_goal_oh,
                b_goal_oh,
                a_obs_oh,
                b_obs_oh,
                conf_a,
                conf_b,
                resource_pressure,
                time_pressure,
                goal_gap,
                mutual_certainty,
                shared_context,
            ],
            dim=0,
        )

        # Add small continuous perturbation
        noise = self.cfg.feature_noise_std * torch.randn(feature.shape[0], generator=torch_gen)
        feature = feature + noise

        label = self._target_class(goal_a, goal_b, a_correct_about_b, b_correct_about_a)
        return feature, label

    def _generate_dataset(self) -> Tuple[Tensor, Tensor]:
        rng = random.Random(self.cfg.seed)
        torch_gen = torch.Generator().manual_seed(self.cfg.seed)

        xs: List[Tensor] = []
        ys: List[int] = []

        for _ in range(self.cfg.n_samples):
            x, y = self._generate_sample(rng, torch_gen)
            xs.append(x)
            ys.append(y)

        features = torch.stack(xs).float()
        labels = torch.tensor(ys, dtype=torch.long)
        return features, labels


def class_distribution(labels: Tensor, num_classes: int) -> Dict[int, int]:
    counts = torch.bincount(labels, minlength=num_classes)
    return {idx: int(counts[idx].item()) for idx in range(num_classes)}


def evaluate(
    model: FracToMNet,
    criterion: FracToMLoss,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float], Tensor]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_breakdown = {
        "task": 0.0,
        "bdi_consistency": 0.0,
        "uncertainty_cal": 0.0,
        "depth_entropy_reg": 0.0,
        "total": 0.0,
    }

    all_preds: List[Tensor] = []
    all_targets: List[Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, report = model(xb, return_interpretability=True)
            loss, breakdown = criterion(logits, yb, report)

            bs = yb.shape[0]
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == yb).sum().item())
            total_count += bs

            for k in total_breakdown:
                total_breakdown[k] += breakdown[k] * bs

            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    avg_breakdown = {k: v / max(1, total_count) for k, v in total_breakdown.items()}

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)

    num_classes = len(CLASS_NAMES)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets_cat, preds_cat):
        cm[t.item(), p.item()] += 1

    return avg_loss, acc, avg_breakdown, cm


def print_confusion_matrix(cm: Tensor) -> None:
    header = " " * 16 + " ".join([f"P{i:>4d}" for i in range(cm.shape[1])])
    print(header)
    for i in range(cm.shape[0]):
        row = " ".join([f"{cm[i, j].item():>5d}" for j in range(cm.shape[1])])
        print(f"T{i:>2d} {CLASS_NAMES[i]:>11s} {row}")


def save_tom_hierarchy_visualization(
    depth_weights: Tensor,
    sigma_map: Dict[int, Tensor],
    output_dir: str,
    max_samples_heatmap: int = 120,
) -> None:
    """Save ToM hierarchy visualizations (depth usage + uncertainty).

    This creates two files under ``output_dir``:
    - ``tom_depth_weights.png``: dominant-depth distribution + per-sample heatmap
    - ``tom_uncertainty.png``: mean epistemic uncertainty per ToM level
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Visualization skipped: matplotlib is not installed.")
        print("Install with: pip install matplotlib")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    alpha = depth_weights.cpu()
    num_samples, num_levels = alpha.shape

    dominant = alpha.argmax(dim=-1).numpy()
    heat_count = min(max_samples_heatmap, num_samples)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        dominant,
        bins=[i - 0.5 for i in range(num_levels + 1)],
        edgecolor="black",
        rwidth=0.9,
    )
    axes[0].set_xticks(list(range(num_levels)))
    axes[0].set_xlabel("Dominant ToM depth")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dominant Theory-of-Mind Depth Distribution")

    axes[1].imshow(alpha[:heat_count].numpy(), aspect="auto", cmap="viridis")
    axes[1].set_xlabel("ToM depth")
    axes[1].set_ylabel("Sample index")
    axes[1].set_title(f"Per-sample ToM depth weights (first {heat_count})")

    plt.tight_layout()
    depth_plot_path = out / "tom_depth_weights.png"
    fig.savefig(depth_plot_path, dpi=160)
    plt.close(fig)

    if sigma_map:
        levels = sorted(sigma_map.keys())
        sigma_means = [float(sigma_map[k].mean().item()) for k in levels]

        fig2, ax2 = plt.subplots(figsize=(8, 4.8))
        ax2.bar(levels, sigma_means)
        ax2.set_xlabel("ToM depth")
        ax2.set_ylabel("Mean epistemic uncertainty (sigma)")
        ax2.set_title("Epistemic Uncertainty by ToM Depth")
        ax2.set_xticks(levels)
        plt.tight_layout()
        sigma_plot_path = out / "tom_uncertainty.png"
        fig2.savefig(sigma_plot_path, dpi=160)
        plt.close(fig2)

    print(f"Saved ToM hierarchy visualizations to: {out.resolve()}")


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # ------------------------ dataset ------------------------
    full_cfg = DatasetConfig(
        n_samples=args.samples,
        num_goals=args.num_goals,
        noise_prob=args.noise_prob,
        feature_noise_std=args.feature_noise,
        seed=args.seed,
    )
    full_dataset = DualAgentCollabDataset(full_cfg)

    # split train/test
    n_train = int(len(full_dataset) * args.train_ratio)
    n_test = len(full_dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # class balance report
    train_labels = torch.tensor([full_dataset.labels[i] for i in train_set.indices], dtype=torch.long)
    test_labels = torch.tensor([full_dataset.labels[i] for i in test_set.indices], dtype=torch.long)

    print("=" * 86)
    print("FracToM Dual-Agent Collaboration Training")
    print("=" * 86)
    print(f"Device: {device}")
    print(f"Samples: total={len(full_dataset)} train={n_train} test={n_test}")
    print(f"Input dim: {full_dataset.input_dim}")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Train class distribution: {class_distribution(train_labels, len(CLASS_NAMES))}")
    print(f"Test  class distribution: {class_distribution(test_labels, len(CLASS_NAMES))}")
    print()

    # ------------------------ model ------------------------
    model = FracToMNet(
        input_dim=full_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        mentalizing_depth=args.depth,
        num_bdi_factors=3,
        blocks_per_column=args.blocks,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        drop_path=args.drop_path,
        num_classes=len(CLASS_NAMES),
    ).to(device)

    criterion = FracToMLoss(
        task_loss_fn=nn.CrossEntropyLoss(),
        lambda_bdi=args.lambda_bdi,
        lambda_uncertainty=args.lambda_uncertainty,
        lambda_depth_entropy=args.lambda_depth_entropy,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print("Starting training...\n")

    # ------------------------ train loop ------------------------
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()

        # developmental curriculum: 1 -> 0
        curriculum_factor = max(0.0, 1.0 - epoch / args.epochs)
        model.set_curriculum(curriculum_factor)

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, report = model(xb, return_interpretability=True)
            loss, _ = criterion(logits, yb, report)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = yb.shape[0]
            running_loss += loss.item() * bs
            running_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            running_count += bs

        scheduler.step()

        train_loss = running_loss / max(1, running_count)
        train_acc = running_correct / max(1, running_count)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            test_loss, test_acc, breakdown, _ = evaluate(model, criterion, test_loader, device)
            best_acc = max(best_acc, test_acc)
            print(
                f"Epoch {epoch:03d} | curriculum={curriculum_factor:.3f} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} best={best_acc:.3f}"
            )
            print(
                f"          loss parts -> task={breakdown['task']:.4f} "
                f"bdi={breakdown['bdi_consistency']:.4f} "
                f"unc={breakdown['uncertainty_cal']:.4f} "
                f"depth_reg={breakdown['depth_entropy_reg']:.4f}"
            )

    print("\nTraining complete.")

    # ------------------------ final evaluation + interpretability ------------------------
    final_loss, final_acc, final_breakdown, cm = evaluate(model, criterion, test_loader, device)
    print("\n" + "=" * 86)
    print("Final Evaluation")
    print("=" * 86)
    print(f"Final test loss: {final_loss:.4f}")
    print(f"Final test acc : {final_acc:.4f}")
    print(f"Best test acc  : {best_acc:.4f}")
    print("Loss breakdown :", {k: round(v, 5) for k, v in final_breakdown.items()})

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print_confusion_matrix(cm)

    # collect interpretability report on the whole test set
    model.eval()
    all_depth_weights: List[Tensor] = []
    sigma_accum: Dict[int, List[Tensor]] = {}

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            _, report = model(xb, return_interpretability=True)
            all_depth_weights.append(report.depth_weights.cpu())
            for k, sigma in report.column_uncertainties.items():
                sigma_accum.setdefault(k, []).append(sigma.cpu())

    depth_weights = torch.cat(all_depth_weights, dim=0)
    fake_report = type("TmpReport", (), {})()
    fake_report.depth_weights = depth_weights
    fake_report.column_uncertainties = {
        k: torch.cat(v, dim=0) for k, v in sigma_accum.items()
    }
    fake_report.bdi_states = {}

    print("\n" + "=" * 86)
    print("Mentalizing Interpretability Summary")
    print("=" * 86)
    print(analyse_mentalizing_depth(fake_report))

    save_tom_hierarchy_visualization(
        depth_weights=depth_weights,
        sigma_map=fake_report.column_uncertainties,
        output_dir=args.viz_dir,
        max_samples_heatmap=args.viz_heatmap_samples,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FracToM on a synthetic dual-agent collaboration dataset"
    )

    # data
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--num-goals", type=int, default=6)
    parser.add_argument("--noise-prob", type=float, default=0.33)
    parser.add_argument("--feature-noise", type=float, default=0.08)
    parser.add_argument("--train-ratio", type=float, default=0.8)

    # model
    parser.add_argument("--hidden-dim", type=int, default=120)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)

    # optimisation
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=5)

    # FracToM loss weights
    parser.add_argument("--lambda-bdi", type=float, default=0.01)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.005)
    parser.add_argument("--lambda-depth-entropy", type=float, default=0.01)

    # misc
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--viz-dir", type=str, default="visualizations")
    parser.add_argument("--viz-heatmap-samples", type=int, default=120)

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # sanity checks
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if args.hidden_dim % 3 != 0:
        raise ValueError("--hidden-dim must be divisible by 3 (BDI factors)")
    if args.num_goals < 2:
        raise ValueError("--num-goals must be >= 2")
    if args.samples < 100:
        raise ValueError("--samples should be at least 100 for meaningful train/test split")
    if args.viz_heatmap_samples < 10:
        raise ValueError("--viz-heatmap-samples must be >= 10")

    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
