"""
eval.py — Comparative Evaluation: FracToM vs. FractalGen Baselines
===================================================================

This script trains and evaluates **four** models on both the collaboration
and competition dual-agent tasks, producing a structured comparison:

    1. **FracToM**          — Fractal Theory-of-Mind (nn.py)
    2. **FractalGenNet**    — Fractal Generative Models baseline (baseline.py)
    3. **VanillaTransformer** — Non-fractal transformer (baseline.py)
    4. **VanillaMLP**       — Standard MLP lower bound (baseline.py)

Usage
-----
    python eval.py                       # default settings
    python eval.py --epochs 60 --seed 42
    python eval.py --tasks collab        # only collaboration task
    python eval.py --tasks compete       # only competition task
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# --- FracToM imports ---
from nn import FracToMLoss, FracToMNet, InterpretabilityReport, analyse_mentalizing_depth

# --- Baseline imports ---
from baseline import FractalGenNet, VanillaMLP, VanillaTransformer

# --- Task-specific dataset imports ---
from collab_train import (
    CLASS_NAMES as COLLAB_CLASSES,
    DatasetConfig as CollabDatasetConfig,
    DualAgentCollabDataset,
)
from compete_train import (
    CLASS_NAMES as COMPETE_CLASSES,
    DatasetConfig as CompeteDatasetConfig,
    DualAgentCompeteDataset,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          HELPERS                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


@dataclass
class RunResult:
    """Stores the results of a single training run."""

    model_name: str
    task_name: str
    num_params: int
    best_test_acc: float
    final_test_acc: float
    final_test_loss: float
    train_time_sec: float
    per_class_acc: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[Tensor] = None
    # FracToM-specific
    depth_weights_mean: Optional[List[float]] = None
    uncertainty_mean: Optional[Dict[int, float]] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def per_class_accuracy(cm: Tensor, class_names: List[str]) -> Dict[str, float]:
    """Compute per-class accuracy from confusion matrix."""
    result = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        result[name] = correct / max(total, 1)
    return result


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    GENERIC TRAINING LOOP                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    task_name: str,
    class_names: List[str],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    is_fractom: bool = False,
    fractom_loss_kwargs: Optional[Dict[str, float]] = None,
    eval_every: int = 5,
    verbose: bool = True,
) -> RunResult:
    """Train a model and return structured results.

    Handles both FracToM (with interpretability report) and baseline
    models (standard cross-entropy only).
    """
    model = model.to(device)
    num_params = count_params(model)
    num_classes = len(class_names)

    # loss
    if is_fractom:
        kw = fractom_loss_kwargs or {}
        criterion = FracToMLoss(
            task_loss_fn=nn.CrossEntropyLoss(),
            lambda_bdi=kw.get("lambda_bdi", 0.01),
            lambda_uncertainty=kw.get("lambda_uncertainty", 0.005),
            lambda_depth_entropy=kw.get("lambda_depth_entropy", 0.01),
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        # FracToM developmental curriculum
        if is_fractom and hasattr(model, "set_curriculum"):
            curriculum = max(0.0, 1.0 - epoch / epochs)
            model.set_curriculum(curriculum)

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if is_fractom:
                logits, report = model(xb, return_interpretability=True)
                loss, _ = criterion(logits, yb, report)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = yb.shape[0]
            running_loss += loss.item() * bs
            running_correct += int((logits.argmax(-1) == yb).sum().item())
            running_count += bs

        scheduler.step()

        # --- evaluate ---
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            test_loss, test_acc, cm = _evaluate_model(
                model, is_fractom, criterion, test_loader, device, num_classes,
            )
            best_acc = max(best_acc, test_acc)

            if verbose:
                train_loss = running_loss / max(1, running_count)
                train_acc = running_correct / max(1, running_count)
                print(
                    f"  [{model_name}] Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                    f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} best={best_acc:.3f}"
                )

    train_time = time.time() - t_start

    # --- final evaluation ---
    final_loss, final_acc, cm = _evaluate_model(
        model, is_fractom, criterion, test_loader, device, num_classes,
    )
    pca = per_class_accuracy(cm, class_names)

    # --- FracToM interpretability ---
    depth_means = None
    unc_means = None
    if is_fractom:
        depth_means, unc_means = _collect_fractom_interpretability(
            model, test_loader, device,
        )

    return RunResult(
        model_name=model_name,
        task_name=task_name,
        num_params=num_params,
        best_test_acc=best_acc,
        final_test_acc=final_acc,
        final_test_loss=final_loss,
        train_time_sec=train_time,
        per_class_acc=pca,
        confusion_matrix=cm,
        depth_weights_mean=depth_means,
        uncertainty_mean=unc_means,
    )


def _evaluate_model(
    model: nn.Module,
    is_fractom: bool,
    criterion: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Tuple[float, float, Tensor]:
    """Evaluate a model and return (loss, accuracy, confusion_matrix)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if is_fractom:
                logits, report = model(xb, return_interpretability=True)
                loss, _ = criterion(logits, yb, report)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            bs = yb.shape[0]
            total_loss += loss.item() * bs
            preds = logits.argmax(-1)
            total_correct += int((preds == yb).sum().item())
            total_count += bs

            for t, p in zip(yb.cpu(), preds.cpu()):
                cm[t.item(), p.item()] += 1

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc, cm


def _collect_fractom_interpretability(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[List[float], Dict[int, float]]:
    """Collect FracToM mentalizing depth weights and uncertainty."""
    model.eval()
    all_alpha: List[Tensor] = []
    sigma_accum: Dict[int, List[Tensor]] = {}

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, report = model(xb, return_interpretability=True)
            all_alpha.append(report.depth_weights.cpu())
            for k, sigma in report.column_uncertainties.items():
                sigma_accum.setdefault(k, []).append(sigma.cpu())

    alpha = torch.cat(all_alpha, dim=0)
    depth_means = alpha.mean(0).tolist()

    unc_means = {}
    for k, v in sigma_accum.items():
        unc_means[k] = float(torch.cat(v, dim=0).mean().item())

    return depth_means, unc_means


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       BUILD MODELS                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝


def build_models(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 120,
    depth: int = 3,
    dropout: float = 0.1,
) -> Dict[str, Tuple[nn.Module, bool]]:
    """Build all models for comparison.

    Returns dict of {name: (model, is_fractom)}.
    """
    models = {}

    # 1. FracToM
    models["FracToM"] = (
        FracToMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            mentalizing_depth=depth,
            num_bdi_factors=3,
            blocks_per_column=1,
            num_heads=4,
            ff_mult=2,
            dropout=dropout,
            drop_path=0.1,
            num_classes=num_classes,
        ),
        True,
    )

    # 2. FractalGenNet (from the paper)
    # Match approximately similar parameter count
    # 3 levels with decreasing capacity, matching paper's design
    fg_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
    # Ensure all dims are at least 2 and divisible for heads
    fg_dims = [max(d, 4) for d in fg_dims]
    fg_heads = [max(1, d // 30) for d in fg_dims]
    # Ensure dim is divisible by num_heads
    for i in range(len(fg_dims)):
        while fg_dims[i] % fg_heads[i] != 0:
            fg_heads[i] = max(1, fg_heads[i] - 1)

    models["FractalGen"] = (
        FractalGenNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_levels=3,
            hidden_dims=fg_dims,
            num_blocks_list=[3, 2, 1],
            num_heads_list=fg_heads,
            mlp_ratio=2.0,
            dropout=dropout,
            attn_dropout=dropout,
            drop_path_rate=0.1,
        ),
        False,
    )

    # 3. Vanilla Transformer
    # Use similar total depth (sum of blocks across fractal levels)
    total_blocks = sum([3, 2, 1])
    vt_heads = max(1, hidden_dim // 30)
    while hidden_dim % vt_heads != 0:
        vt_heads = max(1, vt_heads - 1)

    models["VanillaTransformer"] = (
        VanillaTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=total_blocks,
            num_heads=vt_heads,
            mlp_ratio=2.0,
            dropout=dropout,
            attn_dropout=dropout,
            drop_path_rate=0.1,
        ),
        False,
    )

    # 4. Vanilla MLP
    models["VanillaMLP"] = (
        VanillaMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=4,
            dropout=dropout,
        ),
        False,
    )

    return models


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       TASK BUILDERS                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def build_collab_data(
    seed: int = 7,
    n_samples: int = 8000,
    train_ratio: float = 0.8,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Create collaboration task data loaders."""
    cfg = CollabDatasetConfig(n_samples=n_samples, seed=seed)
    dataset = DualAgentCollabDataset(cfg)

    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.input_dim, COLLAB_CLASSES


def build_compete_data(
    seed: int = 23,
    n_samples: int = 10000,
    train_ratio: float = 0.8,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Create competition task data loaders."""
    cfg = CompeteDatasetConfig(n_samples=n_samples, seed=seed)
    dataset = DualAgentCompeteDataset(cfg)

    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.input_dim, COMPETE_CLASSES


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     REPORTING & VISUALIZATION                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


def print_comparison_table(results: List[RunResult], task_name: str) -> None:
    """Print a formatted comparison table for one task."""
    print()
    print("=" * 100)
    print(f"  COMPARISON TABLE — {task_name.upper()}")
    print("=" * 100)

    # Header
    print(f"{'Model':<22s} {'Params':>10s} {'Best Acc':>10s} {'Final Acc':>10s} "
          f"{'Test Loss':>10s} {'Time (s)':>10s}")
    print("-" * 100)

    for r in results:
        print(
            f"{r.model_name:<22s} {r.num_params:>10,d} {r.best_test_acc:>10.4f} "
            f"{r.final_test_acc:>10.4f} {r.final_test_loss:>10.4f} "
            f"{r.train_time_sec:>10.1f}"
        )

    print("-" * 100)

    # Per-class accuracy
    print(f"\n{'Per-Class Accuracy':}")
    print("-" * 100)

    # Get class names from first result
    if results and results[0].per_class_acc:
        class_names = list(results[0].per_class_acc.keys())
        header = f"{'Model':<22s}"
        for cn in class_names:
            header += f" {cn:>14s}"
        print(header)
        print("-" * 100)

        for r in results:
            if r.per_class_acc:
                row = f"{r.model_name:<22s}"
                for cn in class_names:
                    row += f" {r.per_class_acc[cn]:>14.4f}"
                print(row)

    print("-" * 100)


def print_fractom_interpretability(results: List[RunResult]) -> None:
    """Print FracToM-specific interpretability info."""
    fractom_results = [r for r in results if r.depth_weights_mean is not None]
    if not fractom_results:
        return

    print()
    print("=" * 100)
    print("  FracToM INTERPRETABILITY — Mentalizing Depth Usage")
    print("=" * 100)

    for r in fractom_results:
        print(f"\nTask: {r.task_name}")
        print(f"  Depth weights (mean α):  ", end="")
        for k, w in enumerate(r.depth_weights_mean):
            bar = "█" * int(w * 30)
            print(f"\n    Level {k}: {w:.4f}  {bar}", end="")
        print()

        if r.uncertainty_mean:
            print(f"  Epistemic uncertainty (σ):")
            for k, s in sorted(r.uncertainty_mean.items()):
                print(f"    Column {k}: σ = {s:.4f}")


def print_confusion_matrices(results: List[RunResult], class_names: List[str]) -> None:
    """Print confusion matrices for all models."""
    print()
    print("=" * 100)
    print("  CONFUSION MATRICES")
    print("=" * 100)

    for r in results:
        if r.confusion_matrix is None:
            continue
        cm = r.confusion_matrix
        print(f"\n  {r.model_name} — {r.task_name}")
        header = " " * 16 + " ".join([f"P{i:>4d}" for i in range(cm.shape[1])])
        print(f"  {header}")
        for i in range(cm.shape[0]):
            row = " ".join([f"{cm[i, j].item():>5d}" for j in range(cm.shape[1])])
            label = class_names[i] if i < len(class_names) else f"Class{i}"
            print(f"  T{i:>2d} {label:>11s} {row}")


def print_summary(all_results: Dict[str, List[RunResult]]) -> None:
    """Print a final cross-task summary."""
    print()
    print("=" * 100)
    print("  CROSS-TASK SUMMARY")
    print("=" * 100)

    # Collect model names across all tasks
    model_names = set()
    for results in all_results.values():
        for r in results:
            model_names.add(r.model_name)
    model_names = sorted(model_names)

    task_names = sorted(all_results.keys())
    header = f"{'Model':<22s}"
    for tn in task_names:
        header += f" {tn + ' (Best)':>18s}"
    header += f" {'Average':>12s}"
    print(header)
    print("-" * 100)

    for mn in model_names:
        row = f"{mn:<22s}"
        accs = []
        for tn in task_names:
            results = all_results[tn]
            r = next((x for x in results if x.model_name == mn), None)
            if r:
                row += f" {r.best_test_acc:>18.4f}"
                accs.append(r.best_test_acc)
            else:
                row += f" {'N/A':>18s}"
        if accs:
            avg = sum(accs) / len(accs)
            row += f" {avg:>12.4f}"
        print(row)

    print("-" * 100)

    # Highlight winner
    print("\n  Legend:")
    for tn in task_names:
        results = all_results[tn]
        if results:
            best = max(results, key=lambda r: r.best_test_acc)
            print(f"    {tn}: Winner = {best.model_name} ({best.best_test_acc:.4f})")


def save_results_csv(
    all_results: Dict[str, List[RunResult]],
    output_path: str = "eval_results.csv",
) -> None:
    """Save all results to CSV."""
    path = Path(output_path)
    with open(path, "w") as f:
        f.write("task,model,params,best_acc,final_acc,final_loss,time_sec\n")
        for task_name, results in all_results.items():
            for r in results:
                f.write(
                    f"{r.task_name},{r.model_name},{r.num_params},"
                    f"{r.best_test_acc:.6f},{r.final_test_acc:.6f},"
                    f"{r.final_test_loss:.6f},{r.train_time_sec:.1f}\n"
                )
    print(f"\nResults saved to: {path.resolve()}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


def run_task(
    task_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    class_names: List[str],
    args: argparse.Namespace,
    device: str,
) -> List[RunResult]:
    """Run all models on a single task and return results."""
    num_classes = len(class_names)
    results: List[RunResult] = []

    models = build_models(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )

    print(f"\n{'─' * 100}")
    print(f"  Task: {task_name.upper()} ({num_classes} classes, input_dim={input_dim})")
    print(f"{'─' * 100}")

    for model_name, (model, is_fractom) in models.items():
        print(f"\n  ▸ Training {model_name} ({count_params(model):,} params)...")

        set_seed(args.seed)  # reset seed for fair comparison

        result = train_and_evaluate(
            model=model,
            model_name=model_name,
            task_name=task_name,
            class_names=class_names,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            is_fractom=is_fractom,
            fractom_loss_kwargs={
                "lambda_bdi": args.lambda_bdi,
                "lambda_uncertainty": args.lambda_uncertainty,
                "lambda_depth_entropy": args.lambda_depth_entropy,
            },
            eval_every=args.eval_every,
            verbose=args.verbose,
        )
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative evaluation: FracToM vs. FractalGen baselines"
    )

    # tasks
    parser.add_argument(
        "--tasks", nargs="+", default=["collab", "compete"],
        choices=["collab", "compete"],
        help="Task(s) to evaluate on",
    )

    # data
    parser.add_argument("--collab-samples", type=int, default=8000)
    parser.add_argument("--compete-samples", type=int, default=10000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)

    # model
    parser.add_argument("--hidden-dim", type=int, default=120,
                        help="Base hidden dim (must be divisible by 3 for FracToM)")
    parser.add_argument("--depth", type=int, default=3,
                        help="FracToM mentalizing depth")
    parser.add_argument("--dropout", type=float, default=0.1)

    # training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=5)

    # FracToM loss weights
    parser.add_argument("--lambda-bdi", type=float, default=0.01)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.005)
    parser.add_argument("--lambda-depth-entropy", type=float, default=0.01)

    # misc
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", action="store_false", dest="verbose")
    parser.add_argument("--output-csv", type=str, default="eval_results.csv")

    args = parser.parse_args()

    # validation
    if args.hidden_dim % 3 != 0:
        raise ValueError("--hidden-dim must be divisible by 3 (BDI factors)")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print("=" * 100)
    print("  FracToM vs. FractalGen (arXiv:2502.17437) — Comparative Evaluation")
    print("=" * 100)
    print(f"  Device        : {device}")
    print(f"  Tasks         : {args.tasks}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Hidden dim    : {args.hidden_dim}")
    print(f"  Seed          : {args.seed}")
    print(f"  Learning rate : {args.lr}")

    all_results: Dict[str, List[RunResult]] = {}

    # ─── Collaboration Task ───
    if "collab" in args.tasks:
        train_loader, test_loader, input_dim, class_names = build_collab_data(
            seed=args.seed,
            n_samples=args.collab_samples,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
        )
        results = run_task(
            "Collaboration", train_loader, test_loader,
            input_dim, class_names, args, device,
        )
        all_results["Collaboration"] = results

        print_comparison_table(results, "Collaboration")
        print_confusion_matrices(results, class_names)

    # ─── Competition Task ───
    if "compete" in args.tasks:
        train_loader, test_loader, input_dim, class_names = build_compete_data(
            seed=args.seed + 16,
            n_samples=args.compete_samples,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
        )
        results = run_task(
            "Competition", train_loader, test_loader,
            input_dim, class_names, args, device,
        )
        all_results["Competition"] = results

        print_comparison_table(results, "Competition")
        print_confusion_matrices(results, class_names)

    # ─── Cross-task summary ───
    if len(all_results) > 0:
        # FracToM interpretability
        all_fractom = []
        for results in all_results.values():
            all_fractom.extend(results)
        print_fractom_interpretability(all_fractom)

        # Cross-task summary
        print_summary(all_results)

        # Save CSV
        save_results_csv(all_results, args.output_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
