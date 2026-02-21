"""
pretrain.py — Pre-training / Fine-tuning MIND on CoTh Synthetic Data
=====================================================================
Connects ``api.py`` → ``data.py`` → ``mind.py`` (or ``mlx_mind.py``).

Supports:
  - Causal-LM pre-training (next-token prediction)
  - Optional BDI supervision loss (ground-truth BDI alignment)
  - Optional tier-routing loss (cognitive-tier routing guidance)
  - PyTorch (CUDA / MPS / CPU) and Apple MLX backends

Usage
-----
    # Pre-train with default settings
    python pretrain.py --data data/mind_synthetic.jsonl --epochs 5

    # Fine-tune from a checkpoint with BDI supervision
    python pretrain.py --data data/mind_synthetic.jsonl \\
        --checkpoint ckpt/mind.pt --bdi-loss --tier-loss

    # MLX backend on Apple Silicon
    python pretrain.py --mlx --data data/mind_synthetic.jsonl --epochs 10

    # Small config for testing / debugging
    python pretrain.py --data data/mind_synthetic.jsonl --small
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from data import (
    BDISupervisionLoss,
    MindPretrainDataset,
    MindTokenizer,
    TierRoutingLoss,
    collate_fn,
)
from mind import (
    MindCausalLMOutput,
    MindConfig,
    MindForCausalLM,
    analyse_cognitive_architecture,
    cognitive_load_balancing_loss,
    count_active_parameters,
    count_parameters,
    get_tier_summary,
)


# ═══════════════════════════════════════════════════════════════════════
#  Small Config (for debugging / testing)
# ═══════════════════════════════════════════════════════════════════════


def make_small_config(vocab_size: int) -> MindConfig:
    """A tiny MIND config (~15M params) for quick iteration."""
    return MindConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=64,
        dense_intermediate_size=1024,
        expert_intermediate_size=128,
        shared_expert_intermediate_size=512,
        num_cognitive_modules=4,
        experts_per_module=2,
        num_experts_per_tok=2,
        num_sensory_layers=2,
        num_associative_layers=4,
        num_executive_layers=2,
        bdi_factor_dim=64,
        max_position_embeddings=4096,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch Training
# ═══════════════════════════════════════════════════════════════════════


def train_pytorch(args: argparse.Namespace) -> None:
    # ── Device ──────────────────────────────────────────────────────
    if args.cpu:
        device = "cpu"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer = MindTokenizer(
        args.tokenizer, backend="auto",
    )
    print(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size:,})")

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = MindPretrainDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        include_cognition=not args.no_cognition,
        pack_sequences=not args.no_packing,
    )

    # Train / val split
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Dataset: {len(dataset)} samples (train={n_train}, val={n_val})")
    print(f"Sequence length: {args.max_length}")

    # ── Model ───────────────────────────────────────────────────────
    if args.small:
        config = make_small_config(tokenizer.vocab_size)
        print("Using SMALL config for testing")
    else:
        config = MindConfig(vocab_size=tokenizer.vocab_size)

    model = MindForCausalLM(config).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    total = count_parameters(model)
    active = count_active_parameters(config)
    print(f"\nModel: {total:,} params ({total/1e6:.1f}M)")
    print(f"Active per token: {active:,} ({100*active/total:.1f}%)")
    print(get_tier_summary(config))
    print(f"Device: {device}\n")

    # ── Auxiliary losses ────────────────────────────────────────────
    bdi_loss_fn: Optional[BDISupervisionLoss] = None
    tier_loss_fn: Optional[TierRoutingLoss] = None

    if args.bdi_loss:
        bdi_loss_fn = BDISupervisionLoss(
            config.hidden_size, config.bdi_factor_dim,
        ).to(device)
        print(f"BDI supervision loss: enabled (lambda={args.lambda_bdi})")

    if args.tier_loss:
        tier_loss_fn = TierRoutingLoss(
            num_modules=config.num_cognitive_modules,
            strength=args.lambda_tier,
        ).to(device)
        print(f"Tier routing loss: enabled (lambda={args.lambda_tier})")

    # ── Optimizer ───────────────────────────────────────────────────
    all_params = list(model.parameters())
    if bdi_loss_fn:
        all_params += list(bdi_loss_fn.parameters())

    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training loop ───────────────────────────────────────────────
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}")
    print(f"Steps: {total_steps} (warmup={warmup_steps})\n")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bdi_loss_fn:
            bdi_loss_fn.train()

        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tier_ids = batch["tier_ids"].to(device)

            output: MindCausalLMOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss  # LM loss + router aux loss

            # Optional BDI supervision
            if bdi_loss_fn and output.bdi_states:
                bdi_targets_flat = [
                    t for sample_targets in batch["bdi_targets"]
                    for t in sample_targets
                ]
                if bdi_targets_flat:
                    bdi_l = bdi_loss_fn(
                        output.bdi_states,
                        bdi_targets_flat,
                        model.model.embed_tokens,
                        tokenizer,
                    )
                    loss = loss + args.lambda_bdi * bdi_l

            # Optional tier routing loss
            if tier_loss_fn and output.router_probs:
                tier_l = tier_loss_fn(
                    output.router_probs,
                    tier_ids,
                    config.num_cognitive_modules,
                    config.experts_per_module,
                )
                loss = loss + tier_l

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / max(1, epoch_tokens)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step:>6d} | "
                    f"loss={loss.item():.4f} avg={avg:.4f} | "
                    f"lr={lr_now:.2e}"
                )

        # ── Epoch summary ───────────────────────────────────────────
        dt = time.time() - t0
        train_loss = epoch_loss / max(1, epoch_tokens)
        train_ppl = math.exp(min(train_loss, 20.0))

        # Validation
        val_loss, val_ppl = evaluate_pytorch(
            model, val_loader, device,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            if args.save_dir:
                _save_checkpoint(model, config, args.save_dir, epoch, tokenizer)

        mark = " *" if improved else ""
        print(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"train: loss={train_loss:.4f} ppl={train_ppl:.1f} | "
            f"val: loss={val_loss:.4f} ppl={val_ppl:.1f}{mark} | "
            f"{dt:.1f}s"
        )

    print(f"\nBest val loss: {best_val_loss:.4f} "
          f"(ppl={math.exp(min(best_val_loss, 20.0)):.1f})")


@torch.no_grad()
def evaluate_pytorch(
    model: MindForCausalLM,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Evaluate and return (loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        n_tokens = (labels != -100).sum().item()
        total_loss += output.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


# ═══════════════════════════════════════════════════════════════════════
#  MLX Training
# ═══════════════════════════════════════════════════════════════════════


def train_mlx(args: argparse.Namespace) -> None:
    """MLX-accelerated training on Apple Silicon."""
    try:
        import mlx.core as mx
        import mlx.nn as mnn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
    except ImportError:
        print("MLX not installed.  pip install mlx", file=sys.stderr)
        sys.exit(1)

    from mlx_mind import (
        MindConfig as MLXMindConfig,
        MindForCausalLM as MLXMindForCausalLM,
        count_parameters as mlx_count_parameters,
        count_active_parameters as mlx_count_active_parameters,
        analyse_cognitive_architecture as mlx_analyse,
        clip_grad_norm as mlx_clip_grad_norm,
    )

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer = MindTokenizer(
        args.tokenizer, backend="auto",
    )
    print(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size:,})")

    # ── Dataset (build with PyTorch, convert to MLX) ────────────────
    dataset = MindPretrainDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        include_cognition=not args.no_cognition,
        pack_sequences=not args.no_packing,
    )

    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val

    # Convert entire dataset to MLX arrays
    all_input_ids = []
    all_labels = []
    all_masks = []
    for i in range(len(dataset)):
        s = dataset[i]
        all_input_ids.append(s["input_ids"].numpy())
        all_labels.append(s["labels"].numpy())
        all_masks.append(s["attention_mask"].numpy())

    import numpy as np
    all_input_ids_np = np.stack(all_input_ids)
    all_labels_np = np.stack(all_labels)
    all_masks_np = np.stack(all_masks)

    # Split
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(dataset)).numpy()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ids = mx.array(all_input_ids_np[train_idx])
    train_labels = mx.array(all_labels_np[train_idx].astype(np.int32))
    train_masks = mx.array(all_masks_np[train_idx])
    val_ids = mx.array(all_input_ids_np[val_idx])
    val_labels = mx.array(all_labels_np[val_idx].astype(np.int32))
    val_masks = mx.array(all_masks_np[val_idx])

    print(f"Dataset: {len(dataset)} samples (train={n_train}, val={n_val})")

    # ── Model ───────────────────────────────────────────────────────
    if args.small:
        config = make_small_config(tokenizer.vocab_size)
        print("Using SMALL config for testing")
    else:
        config = MLXMindConfig(vocab_size=tokenizer.vocab_size)

    model = MLXMindForCausalLM(config)
    mx.eval(model.parameters())

    total = mlx_count_parameters(model)
    active = mlx_count_active_parameters(config)
    print(f"\nModel: {total:,} params ({total/1e6:.1f}M)")
    print(f"Active per token: {active:,} ({100*active/total:.1f}%)")
    print(f"Backend: MLX (Metal GPU)\n")

    # ── Optimizer ───────────────────────────────────────────────────
    BS = args.batch_size
    total_steps = args.epochs * ((n_train + BS - 1) // BS)
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    warmup_sched = optim.linear_schedule(
        init=1e-7, end=args.lr, steps=max(1, warmup_steps),
    )
    cosine_sched = optim.cosine_decay(
        init=args.lr, decay_steps=max(1, total_steps - warmup_steps),
    )
    schedule = optim.join_schedules(
        schedules=[warmup_sched, cosine_sched],
        boundaries=[warmup_steps],
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.95],
    )

    # ── Train step ──────────────────────────────────────────────────
    def loss_fn(model, ids, labels, masks):
        output = model(ids, attention_mask=masks, labels=labels)
        return output.loss

    loss_and_grad = mnn.value_and_grad(model, loss_fn)

    print(f"Training: {args.epochs} epochs, batch_size={BS}, lr={args.lr}")
    print(f"Steps: {total_steps} (warmup={warmup_steps})\n")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm_mx = mx.argsort(mx.random.uniform(shape=(n_train,)))
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i in range(0, n_train, BS):
            idx = perm_mx[i : i + BS]
            ids_b = train_ids[idx]
            lab_b = train_labels[idx]
            mask_b = train_masks[idx]

            loss, grads = loss_and_grad(model, ids_b, lab_b, mask_b)
            grads = mlx_clip_grad_norm(grads, max_norm=args.max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / max(1, n_batches)
                print(
                    f"  step {global_step:>6d} | "
                    f"loss={loss.item():.4f} avg={avg:.4f}"
                )

        dt = time.time() - t0
        train_loss = epoch_loss / max(1, n_batches)
        train_ppl = math.exp(min(train_loss, 20.0))

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        n_val_samples = val_ids.shape[0]
        for i in range(0, n_val_samples, BS):
            ids_b = val_ids[i : i + BS]
            lab_b = val_labels[i : i + BS]
            mask_b = val_masks[i : i + BS]
            output = model(ids_b, attention_mask=mask_b, labels=lab_b)
            mx.eval(output.loss)
            val_loss_total += output.loss.item()
            val_batches += 1

        val_loss = val_loss_total / max(1, val_batches)
        val_ppl = math.exp(min(val_loss, 20.0))

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        mark = " *" if improved else ""

        print(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"train: loss={train_loss:.4f} ppl={train_ppl:.1f} | "
            f"val: loss={val_loss:.4f} ppl={val_ppl:.1f}{mark} | "
            f"{dt:.1f}s"
        )

    print(f"\nBest val loss: {best_val_loss:.4f} "
          f"(ppl={math.exp(min(best_val_loss, 20.0)):.1f})")


# ═══════════════════════════════════════════════════════════════════════
#  Checkpoint Utilities
# ═══════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: MindForCausalLM,
    config: MindConfig,
    save_dir: str,
    epoch: int,
    tokenizer: MindTokenizer,
) -> None:
    """Save model checkpoint."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mind_epoch{epoch:03d}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train / fine-tune MIND on CoTh synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pretrain.py --data data/mind_synthetic.jsonl --epochs 5
  python pretrain.py --data data/mind_synthetic.jsonl --small --epochs 2
  python pretrain.py --mlx --data data/mind_synthetic.jsonl --epochs 10
  python pretrain.py --data data/mind_synthetic.jsonl --bdi-loss --tier-loss""",
    )

    # ── Data ────────────────────────────────────────────────────────
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSONL from api.py",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="cl100k_base",
        help="Tokenizer model path (.model) or tiktoken encoding name",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--no-cognition", action="store_true",
        help="Exclude cognitive traces (utterances only)",
    )
    parser.add_argument(
        "--no-packing", action="store_true",
        help="Pad each conversation separately (required for BDI loss)",
    )

    # ── Model ───────────────────────────────────────────────────────
    parser.add_argument(
        "--small", action="store_true",
        help="Use small (~15M) config for testing",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Resume from checkpoint (.pt file)",
    )

    # ── Backend ─────────────────────────────────────────────────────
    parser.add_argument("--mlx", action="store_true", help="Use MLX backend")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    # ── Training ────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)

    # ── Auxiliary losses ────────────────────────────────────────────
    parser.add_argument(
        "--bdi-loss", action="store_true",
        help="Enable BDI supervision loss (needs --no-packing)",
    )
    parser.add_argument("--lambda-bdi", type=float, default=0.1)
    parser.add_argument(
        "--tier-loss", action="store_true",
        help="Enable tier-routing guidance loss",
    )
    parser.add_argument("--lambda-tier", type=float, default=0.05)

    # ── Output ──────────────────────────────────────────────────────
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints",
    )

    args = parser.parse_args()

    # Validate
    if args.bdi_loss and not args.no_packing:
        print(
            "Warning: --bdi-loss requires --no-packing for per-conversation "
            "BDI alignment. Enabling --no-packing automatically.",
        )
        args.no_packing = True

    print("=" * 70)
    print("MIND Pre-training / Fine-tuning")
    print("=" * 70)

    if args.mlx:
        train_mlx(args)
    else:
        train_pytorch(args)


if __name__ == "__main__":
    main()
