"""
data.py — Tokenisation and Dataset Pipeline for MIND Pre-training
==================================================================
Bridges ``api.py``'s synthetic CoTh-annotated conversations to the
``(input_ids, labels, attention_mask)`` format that
``MindForCausalLM.forward()`` expects.

Components
----------
1. ``MindTokenizer``  — thin wrapper around a SentencePiece / tiktoken
   tokenizer with MIND-specific special tokens.
2. ``MindPretrainDataset``  — reads JSONL from ``api.py``, tokenizes,
   packs into fixed-length sequences, returns causal-LM-ready batches.
3. ``BDISupervisionLoss``  — optional auxiliary loss matching the model's
   latent BDI states against ground-truth BDI text embeddings.
4. ``TierAwareLoss``  — optional auxiliary loss that encourages certain
   tokens to be routed through the cognitive tier they belong to.
5. Collation and packaging utilities.

Usage
-----
    from data import MindTokenizer, MindPretrainDataset

    tokenizer = MindTokenizer("tokenizer.model")   # SentencePiece
    dataset = MindPretrainDataset(
        "data/mind_synthetic.jsonl", tokenizer, max_length=2048,
    )
    sample = dataset[0]
    # {"input_ids": Tensor(2048,), "labels": Tensor(2048,),
    #  "attention_mask": Tensor(2048,), "tier_ids": Tensor(2048,),
    #  "bdi_targets": [...]}
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

# ═══════════════════════════════════════════════════════════════════════
#  Special Tokens
# ═══════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|conversation|>",
    "<|/conversation|>",
    "<|scenario|>",
    "<|/scenario|>",
    "<|turn|>",
    "<|/turn|>",
    "<|thought|>",
    "<|sensory|>",
    "<|associative|>",
    "<|executive|>",
    "<|bdi|>",
    "<|tom|>",
    "<|pearl|>",
    "<|deceptive|>",
    "<|honest|>",
]

# Tier annotations — tokens between these markers receive tier_id labels
# 0 = general, 1 = sensory, 2 = associative, 3 = executive
TIER_TOKEN_MAP = {
    "<|sensory|>": 1,
    "<|associative|>": 2,
    "<|executive|>": 3,
    "<|bdi|>": 3,       # BDI is executive-tier
    "<|tom|>": 3,       # ToM is executive-tier
    "<|pearl|>": 3,     # Pearl level is executive-tier
    "<|thought|>": 2,   # Inner monologue ≈ associative
}


# ═══════════════════════════════════════════════════════════════════════
#  Tokenizer Wrapper
# ═══════════════════════════════════════════════════════════════════════


class MindTokenizer:
    """Tokenizer with MIND-specific special tokens.

    Supports two backends:
      - ``sentencepiece``: load a ``.model`` file
      - ``tiktoken``: use a tiktoken encoding name (e.g. ``"cl100k_base"``)

    Special tokens are assigned IDs starting from ``vocab_size`` of the
    base tokenizer and are injected before / after encoding.
    """

    def __init__(
        self,
        model_path_or_encoding: str = "cl100k_base",
        backend: str = "auto",
    ):
        self.backend: str
        self._sp = None
        self._tiktoken = None

        if backend == "auto":
            backend = (
                "sentencepiece"
                if model_path_or_encoding.endswith(".model")
                else "tiktoken"
            )

        if backend == "sentencepiece":
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(model_path_or_encoding)
            self._base_vocab_size = self._sp.GetPieceSize()
            self.backend = "sentencepiece"
        else:
            import tiktoken
            self._tiktoken = tiktoken.get_encoding(model_path_or_encoding)
            self._base_vocab_size = self._tiktoken.max_token_value + 1
            self.backend = "tiktoken"

        # Build special token maps
        self._special_to_id: Dict[str, int] = {}
        self._id_to_special: Dict[int, str] = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            tid = self._base_vocab_size + i
            self._special_to_id[tok] = tid
            self._id_to_special[tid] = tok

        self.pad_id = self._special_to_id["<|pad|>"]
        self.bos_id = self._special_to_id["<|bos|>"]
        self.eos_id = self._special_to_id["<|eos|>"]

    @property
    def vocab_size(self) -> int:
        """Total vocab including special tokens."""
        return self._base_vocab_size + len(SPECIAL_TOKENS)

    def special_token_id(self, token: str) -> int:
        return self._special_to_id[token]

    # ── Encode / Decode ─────────────────────────────────────────────

    def _encode_text(self, text: str) -> List[int]:
        """Encode plain text (no special tokens) with the base tokenizer."""
        if self._sp is not None:
            return self._sp.Encode(text)
        assert self._tiktoken is not None
        return self._tiktoken.encode(text)

    def _decode_ids(self, ids: List[int]) -> str:
        """Decode base-tokenizer IDs back to text."""
        if self._sp is not None:
            return self._sp.Decode(ids)
        assert self._tiktoken is not None
        return self._tiktoken.decode(ids)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text, handling embedded special tokens.

        Splits on special tokens, encodes text segments with the base
        tokenizer, and inserts special-token IDs at the right positions.
        """
        import re
        # Build pattern that splits on any special token
        pattern = "(" + "|".join(
            re.escape(t) for t in sorted(
                self._special_to_id.keys(), key=len, reverse=True,
            )
        ) + ")"
        parts = re.split(pattern, text)

        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for part in parts:
            if not part:
                continue
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._encode_text(part))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back to text, restoring special tokens."""
        chunks: List[str] = []
        buf: List[int] = []
        for tid in ids:
            if tid in self._id_to_special:
                if buf:
                    chunks.append(self._decode_ids(buf))
                    buf = []
                chunks.append(self._id_to_special[tid])
            elif tid < self._base_vocab_size:
                buf.append(tid)
            # else: pad or unknown — skip
        if buf:
            chunks.append(self._decode_ids(buf))
        return "".join(chunks)


# ═══════════════════════════════════════════════════════════════════════
#  Conversation → Token Sequence Serialisation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BDITarget:
    """Ground-truth BDI text for one dialogue turn (for supervision)."""
    turn_idx: int
    speaker: str
    belief: str
    desire: str
    intention: str
    mental_models: List[Dict[str, Any]]  # [{target, inferred_bdi, confidence}]
    pearl_level: int
    tom_depth: int
    is_deceptive: bool
    # Token-level positions (filled after tokenization)
    bdi_token_start: int = -1
    bdi_token_end: int = -1


def conversation_to_text(
    conversation: Dict[str, Any],
    include_cognition: bool = True,
) -> Tuple[str, List[BDITarget]]:
    """Serialise a ``SyntheticConversation`` dict to the MIND special-token
    format, and extract per-turn BDI targets for supervision.

    Returns (text, bdi_targets).
    """
    lines: List[str] = []
    bdi_targets: List[BDITarget] = []

    lines.append("<|conversation|>")
    setting = conversation["scenario"]["setting"]
    lines.append(f"<|scenario|>{setting}<|/scenario|>")

    for i, turn in enumerate(conversation["dialogue"]):
        speaker = turn["speaker"]
        lines.append(f"<|turn|>{speaker}: {turn['utterance']}")

        if include_cognition:
            lines.append(f"<|thought|>{turn['private_thought']}")

            # Tier 1 — Sensory
            obs = " | ".join(turn["cognition"]["sensory"]["observations"])
            lines.append(f"<|sensory|>{obs}")

            # Tier 2 — Associative
            assoc_parts = (
                turn["cognition"]["associative"]["schemas"]
                + turn["cognition"]["associative"]["pragmatic_inferences"]
            )
            lines.append(f"<|associative|>{' | '.join(assoc_parts)}")

            # Tier 3 — Executive
            exe = turn["cognition"]["executive"]
            exec_parts = exe["perspective_shifts"] + exe["causal_reasoning"]
            lines.append(f"<|executive|>{' | '.join(exec_parts)}")

            # BDI
            b = turn["self_bdi"]
            lines.append(
                f"<|bdi|>B: {b['belief']} | "
                f"D: {b['desire']} | "
                f"I: {b['intention']}"
            )

            # Mental models
            for mm in turn["mental_models"]:
                conf = mm["confidence"]
                lines.append(
                    f"<|tom|>{mm['tom_chain']} [conf={conf:.1f}]"
                )

            # Pearl level
            lines.append(f"<|pearl|>L{exe['pearl_level']} depth={exe['tom_depth']}")

            # Deception flag
            if turn.get("is_deceptive", False):
                lines.append("<|deceptive|>")
            else:
                lines.append("<|honest|>")

        lines.append("<|/turn|>")

        # Collect BDI supervision target
        bdi_targets.append(BDITarget(
            turn_idx=i,
            speaker=speaker,
            belief=turn["self_bdi"]["belief"],
            desire=turn["self_bdi"]["desire"],
            intention=turn["self_bdi"]["intention"],
            mental_models=turn.get("mental_models", []),
            pearl_level=turn["cognition"]["executive"]["pearl_level"],
            tom_depth=turn["cognition"]["executive"]["tom_depth"],
            is_deceptive=turn.get("is_deceptive", False),
        ))

    lines.append("<|/conversation|>")
    return "\n".join(lines), bdi_targets


# ═══════════════════════════════════════════════════════════════════════
#  Tier-ID Sequence
# ═══════════════════════════════════════════════════════════════════════


def build_tier_ids(
    token_ids: List[int],
    tokenizer: MindTokenizer,
) -> List[int]:
    """Assign a cognitive-tier label to every token.

    Returns a list the same length as `token_ids` where:
        0 = general (utterance, scenario, structural)
        1 = sensory-tier content
        2 = associative-tier content
        3 = executive-tier content

    Tier spans from a tier-opening special token to the next special token.
    """
    tier_ids = [0] * len(token_ids)
    current_tier = 0
    # Precompute reverse lookup: special_id → tier
    id_to_tier: Dict[int, int] = {}
    for tok_str, tier_val in TIER_TOKEN_MAP.items():
        id_to_tier[tokenizer.special_token_id(tok_str)] = tier_val

    # IDs that reset tier back to 0
    reset_ids = {
        tokenizer.special_token_id(t)
        for t in ("<|turn|>", "<|/turn|>", "<|conversation|>",
                   "<|/conversation|>", "<|scenario|>", "<|/scenario|>")
    }

    all_special_ids = set(tokenizer._special_to_id.values())

    for i, tid in enumerate(token_ids):
        if tid in id_to_tier:
            current_tier = id_to_tier[tid]
            tier_ids[i] = current_tier
        elif tid in reset_ids:
            current_tier = 0
            tier_ids[i] = 0
        elif tid in all_special_ids:
            # Other special tokens keep current tier
            tier_ids[i] = current_tier
        else:
            tier_ids[i] = current_tier
    return tier_ids


def locate_bdi_spans(
    token_ids: List[int],
    tokenizer: MindTokenizer,
) -> List[Tuple[int, int]]:
    """Find (start, end) token positions of each <|bdi|>…<|/turn|> span.

    Returns one span per BDI annotation in the conversation.
    """
    bdi_id = tokenizer.special_token_id("<|bdi|>")
    turn_end_id = tokenizer.special_token_id("<|/turn|>")
    spans = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == bdi_id:
            start = i
            # Find the next special token to end the BDI span
            j = i + 1
            all_special_ids = set(tokenizer._special_to_id.values())
            while j < len(token_ids) and token_ids[j] not in all_special_ids:
                j += 1
            spans.append((start, j))
        i += 1
    return spans


# ═══════════════════════════════════════════════════════════════════════
#  Pre-training Dataset
# ═══════════════════════════════════════════════════════════════════════


class MindPretrainDataset(Dataset):
    """Causal-LM dataset from ``api.py``-generated JSONL.

    Each sample is a tokenized, fixed-length sequence ready for
    ``MindForCausalLM.forward(input_ids, labels=labels, attention_mask=mask)``.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: MindTokenizer,
        max_length: int = 2048,
        include_cognition: bool = True,
        pack_sequences: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_cognition = include_cognition

        # Load raw conversations
        conversations = _load_jsonl_raw(jsonl_path)
        print(f"Loaded {len(conversations)} conversations from {jsonl_path}")

        # Tokenize all conversations
        all_ids: List[List[int]] = []
        all_tier_ids: List[List[int]] = []
        all_bdi_targets: List[List[BDITarget]] = []

        for conv in conversations:
            text, bdi_targets = conversation_to_text(conv, include_cognition)
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            tiers = build_tier_ids(ids, tokenizer)
            all_ids.append(ids)
            all_tier_ids.append(tiers)
            all_bdi_targets.append(bdi_targets)

        if pack_sequences:
            # Pack multiple conversations into fixed-length sequences
            self.samples = _pack_sequences(
                all_ids, all_tier_ids, all_bdi_targets,
                max_length, tokenizer,
            )
        else:
            # Truncate / pad each conversation independently
            self.samples = _pad_sequences(
                all_ids, all_tier_ids, all_bdi_targets,
                max_length, tokenizer,
            )

        print(
            f"Created {len(self.samples)} training sequences "
            f"(max_length={max_length}, "
            f"{'packed' if pack_sequences else 'padded'})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(
                s["attention_mask"], dtype=torch.long,
            ),
            "tier_ids": torch.tensor(s["tier_ids"], dtype=torch.long),
            "bdi_targets": s.get("bdi_targets", []),
        }


# ── Packing: concatenate conversations, slice into fixed windows ────


def _pack_sequences(
    all_ids: List[List[int]],
    all_tier_ids: List[List[int]],
    all_bdi_targets: List[List[BDITarget]],
    max_length: int,
    tokenizer: MindTokenizer,
) -> List[Dict[str, Any]]:
    """Concatenate all tokens with EOS separators, then slice into windows."""
    flat_ids: List[int] = []
    flat_tiers: List[int] = []
    for ids, tiers in zip(all_ids, all_tier_ids):
        flat_ids.extend(ids)
        flat_tiers.extend(tiers)

    # Slice into chunks of max_length
    samples: List[Dict[str, Any]] = []
    for start in range(0, len(flat_ids) - 1, max_length):
        end = min(start + max_length, len(flat_ids))
        chunk_ids = flat_ids[start:end]
        chunk_tiers = flat_tiers[start:end]

        # Pad if the last chunk is short
        pad_len = max_length - len(chunk_ids)
        attention_mask = [1] * len(chunk_ids) + [0] * pad_len
        labels = chunk_ids[1:] + [tokenizer.pad_id]
        if pad_len > 0:
            chunk_ids = chunk_ids + [tokenizer.pad_id] * pad_len
            chunk_tiers = chunk_tiers + [0] * pad_len
            labels = labels + [-100] * (pad_len - 1)  # already +1 from shift

        # Labels: same as input_ids shifted left; pad positions get -100
        labels_final = []
        for i, (lid, mask) in enumerate(zip(labels, attention_mask)):
            if mask == 0:
                labels_final.append(-100)
            else:
                labels_final.append(lid)

        # Ensure lengths match
        chunk_ids = chunk_ids[:max_length]
        labels_final = labels_final[:max_length]
        attention_mask = attention_mask[:max_length]
        chunk_tiers = chunk_tiers[:max_length]

        samples.append({
            "input_ids": chunk_ids,
            "labels": labels_final,
            "attention_mask": attention_mask,
            "tier_ids": chunk_tiers,
            "bdi_targets": [],  # BDI spans lost in packing (fine-tune uses pad mode)
        })

    return samples


# ── Padding: one conversation per sample ────────────────────────────


def _pad_sequences(
    all_ids: List[List[int]],
    all_tier_ids: List[List[int]],
    all_bdi_targets: List[List[BDITarget]],
    max_length: int,
    tokenizer: MindTokenizer,
) -> List[Dict[str, Any]]:
    """Truncate or pad each conversation to max_length independently.

    Preserves per-conversation BDI annotation alignment.
    """
    samples: List[Dict[str, Any]] = []

    for ids, tiers, bdi_tgts in zip(all_ids, all_tier_ids, all_bdi_targets):
        # Truncate to max_length
        ids = ids[:max_length]
        tiers = tiers[:max_length]
        seq_len = len(ids)

        # Build labels (shifted copy)
        labels = ids[1:] + [tokenizer.pad_id]

        # Pad
        pad_len = max_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        ids = ids + [tokenizer.pad_id] * pad_len
        tiers = tiers + [0] * pad_len
        labels = labels + [-100] * pad_len

        # Mask padding in labels
        labels = [
            l if m == 1 else -100
            for l, m in zip(labels, attention_mask)
        ]

        # Locate BDI spans in this sequence for supervision
        bdi_spans = locate_bdi_spans(ids, tokenizer)
        for span_idx, (start, end) in enumerate(bdi_spans):
            if span_idx < len(bdi_tgts):
                bdi_tgts[span_idx].bdi_token_start = start
                bdi_tgts[span_idx].bdi_token_end = min(end, max_length)

        samples.append({
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "tier_ids": tiers,
            "bdi_targets": bdi_tgts,
        })

    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Auxiliary Losses for Cognitive Supervision
# ═══════════════════════════════════════════════════════════════════════

# These losses are designed to be added on top of MIND's existing
# cross-entropy LM loss + router load-balancing loss.


class BDISupervisionLoss(torch.nn.Module):
    """Auxiliary loss matching MIND's latent BDI states to ground-truth
    BDI text embeddings.

    For each executive layer's ``BDITensor`` output at BDI-annotated
    positions, computes cosine similarity loss against embeddings of the
    ground-truth BDI text (belief / desire / intention strings).

    The loss encourages the model's internal BDI representations to align
    with semantically meaningful BDI content, providing direct supervision
    for the FracToM integration layers.

    Architecture:
        gt_text → frozen text embeddings → linear projection → factor_dim
        vs.
        model BDITensor at annotated positions

    The text embeddings come from MIND's own embedding layer (no external
    model needed), averaged over tokens (bag-of-embeddings).
    """

    def __init__(self, hidden_size: int, factor_dim: int):
        super().__init__()
        # Project from hidden_size (embedding dim) to factor_dim
        self.text_to_factor = torch.nn.Linear(hidden_size, factor_dim)
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        bdi_states: List,  # List[BDITensor] from model
        bdi_targets: List[BDITarget],
        embed_fn,           # model.model.embed_tokens callable
        tokenizer: MindTokenizer,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute BDI supervision loss.

        Averages cosine loss across all BDI-annotated positions and all
        executive layers that produced BDI states.
        """
        if not bdi_states or not bdi_targets:
            return torch.tensor(0.0)

        device = bdi_states[0].belief.device
        losses = []

        for tgt in bdi_targets:
            if tgt.bdi_token_start < 0:
                continue

            # Embed ground-truth text via model embeddings
            for component, gt_text in [
                ("belief", tgt.belief),
                ("desire", tgt.desire),
                ("intention", tgt.intention),
            ]:
                gt_ids = tokenizer.encode(gt_text)
                if not gt_ids:
                    continue
                gt_tensor = torch.tensor(gt_ids, device=device).unsqueeze(0)
                gt_emb = embed_fn(gt_tensor).mean(dim=1)  # (1, D)
                gt_proj = self.text_to_factor(gt_emb)      # (1, F)

                # Compare against each executive layer's BDI at this position
                for bdi in bdi_states:
                    model_vec = getattr(bdi, component)     # (B, S, F)
                    # Use the position of the BDI annotation
                    pos = min(tgt.bdi_token_start, model_vec.shape[1] - 1)
                    model_at_pos = model_vec[:1, pos, :]    # (1, F)
                    sim = self.cos(model_at_pos, gt_proj)
                    losses.append(1.0 - sim.mean())

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()


class TierRoutingLoss(torch.nn.Module):
    """Auxiliary loss encouraging tier-appropriate expert routing.

    Compares the model's router probabilities at tier-annotated positions
    against a soft target distribution over cognitive modules:
        sensory tokens    → analytical module (module 0)
        associative tokens → linguistic + associative modules (modules 1, 2)
        executive tokens   → social module (module 3)

    This is a *soft* KL-divergence nudge, not a hard constraint.
    """

    def __init__(self, num_modules: int = 4, strength: float = 0.1):
        super().__init__()
        self.strength = strength
        # Soft targets: (tier → module probability distribution)
        # tier 0 = general (uniform), 1 = sensory, 2 = associative, 3 = executive
        targets = torch.zeros(4, num_modules)
        targets[0] = 1.0 / num_modules                    # general: uniform
        targets[1] = torch.tensor([0.6, 0.2, 0.15, 0.05]) # sensory → analytical
        targets[2] = torch.tensor([0.1, 0.35, 0.40, 0.15]) # assoc → ling + assoc
        targets[3] = torch.tensor([0.05, 0.1, 0.15, 0.70]) # exec → social
        self.register_buffer("targets", targets)

    def forward(
        self,
        router_probs: Tuple[Tensor, ...],
        tier_ids: Tensor,
        num_modules: int = 4,
        experts_per_module: int = 4,
    ) -> Tensor:
        """Compute soft routing alignment loss.

        Parameters
        ----------
        router_probs : tuple of (T, E) from MoE layers
        tier_ids : (B, S) — cognitive tier per token
        """
        if not router_probs:
            return torch.tensor(0.0)

        device = router_probs[0].device
        flat_tiers = tier_ids.reshape(-1)  # (B*S,)
        T = flat_tiers.shape[0]

        losses = []
        for rp in router_probs:
            if rp.shape[0] != T:
                continue
            # Aggregate expert probs to module-level: (T, modules)
            E = rp.shape[1]
            if E != num_modules * experts_per_module:
                continue
            module_probs = rp.view(T, num_modules, experts_per_module).sum(dim=-1)
            module_probs = module_probs / (module_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Compute per-token KL(target || predicted)
            for tier_val in [1, 2, 3]:
                mask = (flat_tiers == tier_val)
                if mask.sum() == 0:
                    continue
                target_dist = self.targets[tier_val].to(device)  # (modules,)
                pred = module_probs[mask]  # (N, modules)
                # KL divergence: target * log(target / pred)
                kl = torch.nn.functional.kl_div(
                    pred.log().clamp(min=-20),
                    target_dist.unsqueeze(0).expand_as(pred),
                    reduction="batchmean",
                    log_target=False,
                )
                losses.append(kl)

        if not losses:
            return torch.tensor(0.0, device=device)
        return self.strength * torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════════════
#  Collation
# ═══════════════════════════════════════════════════════════════════════


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collation that stacks tensor fields and lists BDI targets."""
    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "tier_ids": torch.stack([s["tier_ids"] for s in batch]),
        "bdi_targets": [s["bdi_targets"] for s in batch],
    }


# ═══════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════


def _load_jsonl_raw(path: str) -> List[Dict[str, Any]]:
    """Load JSONL as raw dicts (avoids Pydantic dependency at train time)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Quick Sanity Check
# ═══════════════════════════════════════════════════════════════════════


def _demo() -> None:
    """Demonstrate the pipeline with a tiny synthetic example."""
    print("=" * 60)
    print("data.py — Pipeline Sanity Check")
    print("=" * 60)

    # Create a minimal fake conversation matching api.py's schema
    fake_conv = {
        "scenario": {
            "setting": "Two colleagues in a meeting room",
            "agents": [
                {"name": "Alice", "role": "manager",
                 "traits": ["assertive"], "hidden_goal": "get promotion",
                 "private_knowledge": ["budget cut coming"]},
                {"name": "Bob", "role": "engineer",
                 "traits": ["cautious"], "hidden_goal": "keep project alive",
                 "private_knowledge": ["has backup plan"]},
            ],
            "hidden_dynamics": "Alice knows about cuts, Bob doesn't",
            "target_tom_depth": 2,
            "cognitive_demands": ["deception detection"],
        },
        "dialogue": [
            {
                "speaker": "Alice",
                "utterance": "The project looks great, Bob.",
                "private_thought": "I need to gauge his awareness of the cuts.",
                "cognition": {
                    "sensory": {
                        "observations": ["Bob looks relaxed", "No tension cues"],
                        "emotional_cues": ["calm"],
                        "salience": "Bob's relaxed posture",
                    },
                    "associative": {
                        "schemas": ["performance review opener"],
                        "pragmatic_inferences": [
                            "compliment as rapport-building"
                        ],
                        "analogies": ["similar to last quarter's review"],
                    },
                    "executive": {
                        "tom_depth": 2,
                        "perspective_shifts": [
                            "I think Bob believes the project is safe"
                        ],
                        "causal_reasoning": [
                            "If I reveal cuts now, Bob will panic"
                        ],
                        "pearl_level": 1,
                        "metacognition": "80% confident in my read of Bob",
                    },
                },
                "self_bdi": {
                    "belief": "Budget cuts are coming",
                    "desire": "Assess Bob's awareness",
                    "intention": "Probe subtly without revealing",
                },
                "mental_models": [
                    {
                        "target": "Bob",
                        "inferred_bdi": {
                            "belief": "Project is safe",
                            "desire": "Continue current work",
                            "intention": "Present progress normally",
                        },
                        "confidence": 0.8,
                        "tom_chain": "I think Bob believes the project is safe",
                    }
                ],
                "is_deceptive": True,
            },
            {
                "speaker": "Bob",
                "utterance": "Thanks! We're on track for the milestone.",
                "private_thought": "She seems too complimentary. Something's off.",
                "cognition": {
                    "sensory": {
                        "observations": [
                            "Alice smiling but eyes not matching"
                        ],
                        "emotional_cues": ["incongruent affect"],
                        "salience": "Mismatch between words and expression",
                    },
                    "associative": {
                        "schemas": ["false reassurance pattern"],
                        "pragmatic_inferences": [
                            "excessive praise may signal bad news"
                        ],
                        "analogies": [
                            "Like when previous manager praised before layoffs"
                        ],
                    },
                    "executive": {
                        "tom_depth": 2,
                        "perspective_shifts": [
                            "I think Alice thinks I don't suspect anything"
                        ],
                        "causal_reasoning": [
                            "If there were no problems, she wouldn't be this nice"
                        ],
                        "pearl_level": 2,
                        "metacognition": "70% confident something is wrong",
                    },
                },
                "self_bdi": {
                    "belief": "Alice may be hiding bad news",
                    "desire": "Uncover the truth",
                    "intention": "Play along while probing",
                },
                "mental_models": [
                    {
                        "target": "Alice",
                        "inferred_bdi": {
                            "belief": "Bob is unaware of problems",
                            "desire": "Control the narrative",
                            "intention": "Deliver bad news gradually",
                        },
                        "confidence": 0.7,
                        "tom_chain": (
                            "I think Alice thinks I believe "
                            "everything is fine"
                        ),
                    }
                ],
                "is_deceptive": False,
            },
        ],
        "labels": {
            "contains_deception": True,
            "deceptive_turns": [0],
            "false_beliefs": ["Bob may falsely believe project is safe"],
            "max_tom_depth": 2,
            "dominant_pearl_level": 1,
            "social_dynamic": "manipulation",
            "cognitive_phenomena": ["curse of knowledge"],
        },
        "summary": "Alice hides budget cut info while assessing Bob's awareness.",
    }

    # Step 1: Serialize to text
    text, bdi_targets = conversation_to_text(fake_conv)
    print(f"\n--- Serialized text ({len(text)} chars) ---")
    for line in text.split("\n")[:10]:
        print(f"  {line}")
    print(f"  ... ({text.count(chr(10))} lines total)")
    print(f"  BDI targets: {len(bdi_targets)}")

    # Step 2: Tokenize
    tokenizer = MindTokenizer("cl100k_base", backend="tiktoken")
    print(f"\n--- Tokenizer ---")
    print(f"  Backend: tiktoken (cl100k_base)")
    print(f"  Base vocab: {tokenizer._base_vocab_size:,}")
    print(f"  Total vocab (with special): {tokenizer.vocab_size:,}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")

    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(f"\n--- Tokenized ---")
    print(f"  Token count: {len(ids)}")
    print(f"  First 20 IDs: {ids[:20]}")

    # Step 3: Tier IDs
    tiers = build_tier_ids(ids, tokenizer)
    tier_counts = {i: tiers.count(i) for i in range(4)}
    print(f"\n--- Tier distribution ---")
    tier_names = {0: "general", 1: "sensory", 2: "associative", 3: "executive"}
    for t, c in tier_counts.items():
        pct = 100 * c / len(tiers)
        print(f"  Tier {t} ({tier_names[t]}): {c} tokens ({pct:.1f}%)")

    # Step 4: Round-trip decode
    decoded = tokenizer.decode(ids[:50])
    print(f"\n--- Decode first 50 tokens ---")
    print(f"  {decoded[:200]}...")

    # Step 5: Write fake JSONL temporarily for dataset test
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        f.write(json.dumps(fake_conv) + "\n")
        f.write(json.dumps(fake_conv) + "\n")  # duplicate for testing
        tmp_path = f.name

    print(f"\n--- MindPretrainDataset (packed) ---")
    ds_packed = MindPretrainDataset(
        tmp_path, tokenizer, max_length=512, pack_sequences=True,
    )
    sample = ds_packed[0]
    print(f"  Samples: {len(ds_packed)}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape:    {sample['labels'].shape}")
    print(f"  attention_mask:  {sample['attention_mask'].sum().item()} / "
          f"{sample['attention_mask'].shape[0]} active")
    print(f"  tier_ids unique: {sample['tier_ids'].unique().tolist()}")

    print(f"\n--- MindPretrainDataset (padded, per-conversation) ---")
    ds_pad = MindPretrainDataset(
        tmp_path, tokenizer, max_length=512, pack_sequences=False,
    )
    sample2 = ds_pad[0]
    print(f"  Samples: {len(ds_pad)}")
    print(f"  BDI targets: {len(sample2['bdi_targets'])}")
    if sample2["bdi_targets"]:
        t0 = sample2["bdi_targets"][0]
        print(f"    Turn 0: speaker={t0.speaker}, belief='{t0.belief[:50]}...'")
        print(f"    Token span: [{t0.bdi_token_start}, {t0.bdi_token_end})")

    # Step 6: Collation
    batch = collate_fn([ds_packed[i] for i in range(min(2, len(ds_packed)))])
    print(f"\n--- Collated batch ---")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels:    {batch['labels'].shape}")
    print(f"  tier_ids:  {batch['tier_ids'].shape}")

    # Verify labels alignment
    ignore_count = (batch["labels"] == -100).sum().item()
    total = batch["labels"].numel()
    print(f"  Ignored positions (pad/-100): {ignore_count}/{total}")

    # Cleanup
    Path(tmp_path).unlink()

    print(f"\n{'=' * 60}")
    print("Pipeline sanity check passed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _demo()
