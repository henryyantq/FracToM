"""
compete_train.py

Dual-agent competition simulation + FracToM training.

This script builds a synthetic random generative dataset for *competitive*
dual-agent settings, trains FracToM from nn.py, evaluates classification
quality, and runs a post-training competition simulation against baselines.

Competition task (policy prediction for Agent A)
------------------------------------------------
Predict one of 5 strategic policies:

0: HOLD_POSITION    (stabilize, avoid over-commit)
1: AGGRESSIVE_PUSH  (press advantage)
2: BAIT_SWITCH      (deceptive redirection)
3: DENY_AND_BLOCK   (contest and deny shared objective)
4: ADAPTIVE_RETREAT (disengage and recover tempo)

The label is generated from latent variables:
- true objectives of A/B,
- noisy observations about each other,
- relative strength and stamina,
- risk/map/scarcity context,
- confidence and correctness of opponent-model beliefs.

Why FracToM is appropriate:
- This setting requires reasoning about self and opponent states,
- multi-level mentalizing is useful for deceptive and denial strategies,
- interpretability report can expose depth usage under competitive pressure.
"""

from __future__ import annotations

import argparse
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
    "HOLD_POSITION",
    "AGGRESSIVE_PUSH",
    "BAIT_SWITCH",
    "DENY_AND_BLOCK",
    "ADAPTIVE_RETREAT",
]


@dataclass
class DatasetConfig:
    n_samples: int = 9000
    num_objectives: int = 6
    obs_noise_prob: float = 0.30
    feature_noise_std: float = 0.06
    seed: int = 23


class DualAgentCompeteDataset(Dataset):
    """Random generative competitive dataset for policy prediction.

    Labels are generated via a structured rule set over latent game factors.
    The resulting task is non-trivial and typically benefits from opponent
    modelling (a good fit for FracToM mentalizing depths).
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.features, self.labels = self._build()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        return self.features.shape[-1]

    @staticmethod
    def _one_hot(index: int, size: int) -> Tensor:
        vec = torch.zeros(size)
        vec[index] = 1.0
        return vec

    def _noisy_view(
        self,
        true_obj: int,
        rng: random.Random,
    ) -> Tuple[int, int]:
        """Return (observed_objective, is_correct)."""
        if rng.random() < self.cfg.obs_noise_prob:
            candidates = [j for j in range(self.cfg.num_objectives) if j != true_obj]
            return rng.choice(candidates), 0
        return true_obj, 1

    def _label_rule(
        self,
        objective_a: int,
        objective_b: int,
        a_correct_about_b: int,
        b_correct_about_a: int,
        strength_a: float,
        strength_b: float,
        stamina_a: float,
        stamina_b: float,
        risk_level: float,
        scarcity: float,
        conf_a: float,
        conf_b: float,
    ) -> int:
        """Policy label for Agent A in competitive setting."""
        advantage = (strength_a - strength_b) + 0.35 * (stamina_a - stamina_b)
        same_objective = int(objective_a == objective_b)

        # Shared objective + high scarcity -> denial contest
        if same_objective and scarcity > 0.62 and conf_a > 0.55 and conf_b > 0.55:
            return 3  # DENY_AND_BLOCK

        # Strong advantage and good opponent model -> push
        if advantage > 0.22 and a_correct_about_b == 1 and conf_a > 0.58 and risk_level < 0.75:
            return 1  # AGGRESSIVE_PUSH

        # Disadvantage + weak confidence/correctness -> retreat
        if advantage < -0.24 and (a_correct_about_b == 0 or conf_a < 0.45):
            return 4  # ADAPTIVE_RETREAT

        # Deception-friendly setting
        # A is uncertain about B while B seems certain about A, and objectives differ.
        if (
            objective_a != objective_b
            and a_correct_about_b == 0
            and b_correct_about_a == 1
            and abs(advantage) < 0.28
            and risk_level > 0.30
        ):
            return 2  # BAIT_SWITCH

        # Default stable policy
        return 0  # HOLD_POSITION

    def _make_sample(self, rng: random.Random, tgen: torch.Generator) -> Tuple[Tensor, int]:
        num_obj = self.cfg.num_objectives

        # Latent true objectives
        objective_a = rng.randrange(num_obj)
        objective_b = rng.randrange(num_obj)

        # Noisy cross-observations
        obs_a_about_b, a_correct_about_b = self._noisy_view(objective_b, rng)
        obs_b_about_a, b_correct_about_a = self._noisy_view(objective_a, rng)

        # Capabilities
        strength_a = rng.random()
        strength_b = rng.random()
        stamina_a = rng.random()
        stamina_b = rng.random()

        # Environment/game context
        risk_level = rng.random()
        map_openness = rng.random()
        scarcity = rng.random()
        tempo = rng.random()

        # Confidence aligned (imperfectly) with correctness
        conf_a = (0.62 + 0.35 * rng.random()) if a_correct_about_b else (0.10 + 0.45 * rng.random())
        conf_b = (0.62 + 0.35 * rng.random()) if b_correct_about_a else (0.10 + 0.45 * rng.random())

        # Derived relational descriptors
        advantage = (strength_a - strength_b) + 0.35 * (stamina_a - stamina_b)
        power_ratio = (strength_a + 1e-4) / (strength_b + 1e-4)
        objective_overlap = float(objective_a == objective_b)
        confidence_gap = conf_a - conf_b
        mutual_uncertainty = 1.0 - ((conf_a + conf_b) / 2.0)

        # Encodings
        objective_a_oh = self._one_hot(objective_a, num_obj)
        objective_b_oh = self._one_hot(objective_b, num_obj)
        obs_a_oh = self._one_hot(obs_a_about_b, num_obj)
        obs_b_oh = self._one_hot(obs_b_about_a, num_obj)

        scalars = torch.tensor(
            [
                strength_a,
                strength_b,
                stamina_a,
                stamina_b,
                conf_a,
                conf_b,
                risk_level,
                map_openness,
                scarcity,
                tempo,
                advantage,
                power_ratio,
                objective_overlap,
                float(a_correct_about_b),
                float(b_correct_about_a),
                confidence_gap,
                mutual_uncertainty,
            ],
            dtype=torch.float32,
        )

        latent_context = torch.randn(7, generator=tgen)

        x = torch.cat(
            [objective_a_oh, objective_b_oh, obs_a_oh, obs_b_oh, scalars, latent_context],
            dim=0,
        )

        # Add small perturbation for robustness
        x = x + self.cfg.feature_noise_std * torch.randn(x.shape[0], generator=tgen)

        y = self._label_rule(
            objective_a,
            objective_b,
            a_correct_about_b,
            b_correct_about_a,
            strength_a,
            strength_b,
            stamina_a,
            stamina_b,
            risk_level,
            scarcity,
            conf_a,
            conf_b,
        )
        return x, y

    def _build(self) -> Tuple[Tensor, Tensor]:
        rng = random.Random(self.cfg.seed)
        tgen = torch.Generator().manual_seed(self.cfg.seed)

        xs: List[Tensor] = []
        ys: List[int] = []

        for _ in range(self.cfg.n_samples):
            x, y = self._make_sample(rng, tgen)
            xs.append(x)
            ys.append(y)

        return torch.stack(xs).float(), torch.tensor(ys, dtype=torch.long)


def class_distribution(labels: Tensor, num_classes: int) -> Dict[int, int]:
    counts = torch.bincount(labels, minlength=num_classes)
    return {i: int(counts[i].item()) for i in range(num_classes)}


def evaluate(
    model: FracToMNet,
    criterion: FracToMLoss,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float], Tensor]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    running = {
        "task": 0.0,
        "bdi_consistency": 0.0,
        "uncertainty_cal": 0.0,
        "depth_entropy_reg": 0.0,
        "total": 0.0,
    }

    all_pred: List[Tensor] = []
    all_true: List[Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, report = model(xb, return_interpretability=True)
            loss, breakdown = criterion(logits, yb, report)

            bs = yb.shape[0]
            total_samples += bs
            total_loss += loss.item() * bs
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == yb).sum().item())

            for k in running:
                running[k] += breakdown[k] * bs

            all_pred.append(pred.cpu())
            all_true.append(yb.cpu())

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    avg_breakdown = {k: v / max(1, total_samples) for k, v in running.items()}

    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)

    num_classes = len(CLASS_NAMES)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t.item(), p.item()] += 1

    return avg_loss, acc, avg_breakdown, cm


def print_confusion_matrix(cm: Tensor) -> None:
    print(" " * 18 + " ".join([f"P{i:>4d}" for i in range(cm.shape[1])]))
    for i in range(cm.shape[0]):
        row = " ".join([f"{cm[i, j].item():>5d}" for j in range(cm.shape[1])])
        print(f"T{i:>2d} {CLASS_NAMES[i]:>14s} {row}")


def save_tom_hierarchy_visualization(
    depth_weights: Tensor,
    sigma_map: Dict[int, Tensor],
    output_dir: str,
    max_samples_heatmap: int = 120,
) -> None:
    """Save ToM hierarchy visualizations (depth usage + uncertainty)."""
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


def outcome_win_prob(pred_class: int, true_class: int) -> float:
    """Competition simulator payoff model.

    If predicted strategy matches oracle label, win probability is high.
    Near-miss strategies are moderate; mismatches are low.
    """
    if pred_class == true_class:
        return 0.72

    near_pairs = {
        (1, 0), (0, 1),
        (3, 0), (0, 3),
        (2, 0), (0, 2),
        (4, 0), (0, 4),
        (2, 4), (4, 2),
    }
    if (pred_class, true_class) in near_pairs:
        return 0.50

    return 0.28


def run_competition_simulation(
    model: FracToMNet,
    dataset: DualAgentCompeteDataset,
    episodes: int,
    seed: int,
    device: str,
) -> None:
    """Simulate competition rounds with trained model vs baselines."""
    rng = random.Random(seed)

    model.eval()

    wins_model = 0
    wins_random = 0
    wins_majority = 0

    reward_model = 0.0
    reward_random = 0.0
    reward_majority = 0.0

    majority_class = int(torch.bincount(dataset.labels).argmax().item())

    with torch.no_grad():
        for _ in range(episodes):
            idx = rng.randrange(len(dataset))
            x, y = dataset[idx]
            true_class = int(y.item())

            xb = x.unsqueeze(0).to(device)
            logits = model(xb)
            pred_class = int(logits.argmax(dim=-1).item())

            random_class = rng.randrange(len(CLASS_NAMES))
            majority_pred = majority_class

            p_model = outcome_win_prob(pred_class, true_class)
            p_random = outcome_win_prob(random_class, true_class)
            p_majority = outcome_win_prob(majority_pred, true_class)

            # Bernoulli outcomes
            w_model = 1 if rng.random() < p_model else 0
            w_random = 1 if rng.random() < p_random else 0
            w_majority = 1 if rng.random() < p_majority else 0

            wins_model += w_model
            wins_random += w_random
            wins_majority += w_majority

            # Utility: +1 for win, -0.7 for loss
            reward_model += (1.0 if w_model else -0.7)
            reward_random += (1.0 if w_random else -0.7)
            reward_majority += (1.0 if w_majority else -0.7)

    print("\n" + "=" * 90)
    print("Dual-Agent Competition Simulation")
    print("=" * 90)
    print(f"Episodes: {episodes}")
    print(f"Majority baseline class: {majority_class} ({CLASS_NAMES[majority_class]})")
    print()

    print("Win rate comparison:")
    print(f"  FracToM policy     : {wins_model / episodes:.3f}")
    print(f"  Random policy      : {wins_random / episodes:.3f}")
    print(f"  Majority policy    : {wins_majority / episodes:.3f}")
    print()

    print("Average utility per episode:")
    print(f"  FracToM policy     : {reward_model / episodes:.3f}")
    print(f"  Random policy      : {reward_random / episodes:.3f}")
    print(f"  Majority policy    : {reward_majority / episodes:.3f}")


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # ------------------------ data ------------------------
    ds_cfg = DatasetConfig(
        n_samples=args.samples,
        num_objectives=args.num_objectives,
        obs_noise_prob=args.obs_noise_prob,
        feature_noise_std=args.feature_noise,
        seed=args.seed,
    )
    full_ds = DualAgentCompeteDataset(ds_cfg)

    n_train = int(len(full_ds) * args.train_ratio)
    n_test = len(full_ds) - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    train_labels = torch.tensor([full_ds.labels[i] for i in train_ds.indices], dtype=torch.long)
    test_labels = torch.tensor([full_ds.labels[i] for i in test_ds.indices], dtype=torch.long)

    print("=" * 90)
    print("FracToM Dual-Agent Competition Training")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Samples: total={len(full_ds)} train={n_train} test={n_test}")
    print(f"Input dim: {full_ds.input_dim}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train class distribution: {class_distribution(train_labels, len(CLASS_NAMES))}")
    print(f"Test  class distribution: {class_distribution(test_labels, len(CLASS_NAMES))}")
    print()

    # ------------------------ model ------------------------
    model = FracToMNet(
        input_dim=full_ds.input_dim,
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
    print(f"Model parameters: {total_params:,}")
    print("Start training...\n")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()

        curriculum = max(0.0, 1.0 - epoch / args.epochs)
        model.set_curriculum(curriculum)

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
                f"Epoch {epoch:03d} | curr={curriculum:.3f} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} best={best_acc:.3f}"
            )
            print(
                f"            loss parts -> task={breakdown['task']:.4f} "
                f"bdi={breakdown['bdi_consistency']:.4f} "
                f"unc={breakdown['uncertainty_cal']:.4f} "
                f"depth_reg={breakdown['depth_entropy_reg']:.4f}"
            )

    print("\nTraining complete.")

    final_loss, final_acc, final_breakdown, cm = evaluate(model, criterion, test_loader, device)
    print("\n" + "=" * 90)
    print("Final Evaluation")
    print("=" * 90)
    print(f"Final test loss: {final_loss:.4f}")
    print(f"Final test acc : {final_acc:.4f}")
    print(f"Best test acc  : {best_acc:.4f}")
    print("Loss breakdown :", {k: round(v, 5) for k, v in final_breakdown.items()})

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print_confusion_matrix(cm)

    # interpretability summary over test set
    model.eval()
    depth_weights: List[Tensor] = []
    sigma_map: Dict[int, List[Tensor]] = {}

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            _, report = model(xb, return_interpretability=True)
            depth_weights.append(report.depth_weights.cpu())
            for k, sigma in report.column_uncertainties.items():
                sigma_map.setdefault(k, []).append(sigma.cpu())

    merged_depth = torch.cat(depth_weights, dim=0)
    fake_report = type("TmpReport", (), {})()
    fake_report.depth_weights = merged_depth
    fake_report.column_uncertainties = {k: torch.cat(v, dim=0) for k, v in sigma_map.items()}
    fake_report.bdi_states = {}

    print("\n" + "=" * 90)
    print("Mentalizing Interpretability Summary")
    print("=" * 90)
    print(analyse_mentalizing_depth(fake_report))

    save_tom_hierarchy_visualization(
        depth_weights=merged_depth,
        sigma_map=fake_report.column_uncertainties,
        output_dir=args.viz_dir,
        max_samples_heatmap=args.viz_heatmap_samples,
    )

    # post-training competition simulator
    run_competition_simulation(
        model=model,
        dataset=full_ds,
        episodes=args.sim_episodes,
        seed=args.seed + 17,
        device=device,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FracToM on synthetic dual-agent competition data and run simulation"
    )

    # data
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--num-objectives", type=int, default=6)
    parser.add_argument("--obs-noise-prob", type=float, default=0.30)
    parser.add_argument("--feature-noise", type=float, default=0.06)
    parser.add_argument("--train-ratio", type=float, default=0.8)

    # model
    parser.add_argument("--hidden-dim", type=int, default=120)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)

    # optimization
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=5)

    # FracToM loss weights
    parser.add_argument("--lambda-bdi", type=float, default=0.01)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.005)
    parser.add_argument("--lambda-depth-entropy", type=float, default=0.01)

    # simulation
    parser.add_argument("--sim-episodes", type=int, default=2000)

    # misc
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--viz-dir", type=str, default="visualizations_compete")
    parser.add_argument("--viz-heatmap-samples", type=int, default=120)

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_arg_parser().parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if args.hidden_dim % 3 != 0:
        raise ValueError("--hidden-dim must be divisible by 3 for BDI factors")
    if args.num_objectives < 2:
        raise ValueError("--num-objectives must be >= 2")
    if args.samples < 200:
        raise ValueError("--samples should be >= 200")
    if args.sim_episodes < 100:
        raise ValueError("--sim-episodes should be >= 100")
    if args.viz_heatmap_samples < 10:
        raise ValueError("--viz-heatmap-samples must be >= 10")

    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
