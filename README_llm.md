# MIND — Mixture of Intelligent Neural Dynamics

Technical documentation for `mind.py`: a ~1B-parameter Mixture-of-Experts causal language model that organises computation into parallel, hierarchical cognitive processing streams, using FracToM's causal Theory-of-Mind decoder as the integrative backbone.

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Architecture Overview](#2-architecture-overview)
3. [Component-by-Component Walkthrough](#3-component-by-component-walkthrough)
   - 3.1 [Configuration (`MindConfig`)](#31-configuration-mindconfig)
   - 3.2 [Core Primitives](#32-core-primitives)
   - 3.3 [Grouped-Query Attention (`CognitiveAttention`)](#33-grouped-query-attention-cognitiveattention)
   - 3.4 [Feed-Forward Networks (`CognitiveMLP`, `CognitiveExpert`)](#34-feed-forward-networks-cognitivemlp-cognitiveexpert)
   - 3.5 [Hierarchical Router (`CognitiveRouter`)](#35-hierarchical-router-cognitiverouter)
   - 3.6 [Cognitive MoE Block (`CognitiveMoEBlock`)](#36-cognitive-moe-block-cognitivemoeblock)
   - 3.7 [FracToM Integration (`FracToMIntegration`)](#37-fractom-integration-fractomintegration)
   - 3.8 [Decoder Layer (`MindDecoderLayer`)](#38-decoder-layer-minddecoderlayer)
   - 3.9 [Backbone (`MindModel`)](#39-backbone-mindmodel)
   - 3.10 [Language Model Head (`MindForCausalLM`)](#310-language-model-head-mindforcausallm)
   - 3.11 [Load Balancing Loss](#311-load-balancing-loss)
4. [Data Flow](#4-data-flow)
5. [Parameter Budget Breakdown](#5-parameter-budget-breakdown)
6. [Scaling Guide — Extending to Larger Models](#6-scaling-guide--extending-to-larger-models)
   - 6.1 [General Scaling Strategy](#61-general-scaling-strategy)
   - 6.2 [Concrete Recipes (3B, 8B, 30B)](#62-concrete-recipes-3b-8b-30b)
   - 6.3 [Dimension-by-Dimension Scaling Analysis](#63-dimension-by-dimension-scaling-analysis)
   - 6.4 [Infrastructure Requirements](#64-infrastructure-requirements)
   - 6.5 [Training Improvements for Better Performance](#65-training-improvements-for-better-performance)
   - 6.6 [Architectural Improvements for Better Performance](#66-architectural-improvements-for-better-performance)

---

## 1. Design Rationale

MIND fuses two ideas:

1. **Sparse Mixture-of-Experts** — only a fraction of parameters are active per token, enabling large capacity at bounded compute (Fedus et al., 2021; Qwen3-MoE).
2. **Cognitive processing hierarchy** — the brain processes information through tiered, parallel cortical streams: sensory → associative → executive (Mesulam, 1998).

The result is a decoder where early layers are dense and cheap (rapid feature extraction), middle layers route tokens to functionally specialised expert groups (multi-modal integration), and top layers add causal Theory-of-Mind reasoning from FracToM (planning, perspective-taking). This mirrors the progression from primary sensory cortex through association cortex to prefrontal cortex.

**Why not just use Qwen3-MoE unchanged?** Qwen3's MoE treats all experts as interchangeable and all layers identically (flat top-k routing, uniform sparse step). MIND makes three structural departures:

| Qwen3-MoE | MIND |
|-----------|------|
| Flat pool of undifferentiated experts | Experts grouped into 4 cognitive modules |
| Single-level top-k softmax router | Two-stage hierarchical router (module → expert) |
| Uniform layer structure (same MoE everywhere) | Three distinct tiers with different compute profiles |
| No mentalizing / causal reasoning | FracToM BDI + SCM in executive layers |

---

## 2. Architecture Overview

```
Input Token IDs  (B, S)
        │
        ▼
   ┌──────────┐
   │ Embedding │  → (B, S, 1536)    +  RoPE position encoding
   └────┬─────┘
        │
════════╪═══════════════ TIER 1: SENSORY (layers 0–5) ═══════════════
        │    6 layers × [RMSNorm → GQA Attention → RMSNorm → Dense SwiGLU FFN]
        │    No routing overhead. Fast feature extraction.
        │
════════╪═══════════════ TIER 2: ASSOCIATIVE (layers 6–17) ══════════
        │    12 layers × [RMSNorm → GQA Attention → RMSNorm → Cognitive MoE]
        │    Tokens routed to 4/16 specialised experts + 1 shared expert.
        │
════════╪═══════════════ TIER 3: EXECUTIVE (layers 18–23) ═══════════
        │    6 layers × [RMSNorm → GQA Attention → RMSNorm → Cognitive MoE
        │                                                    → FracToM BDI]
        │    Same MoE as Tier 2, plus BDI extraction, epistemic gating,
        │    and cross-depth perspective shifting.
        │
   ┌────┴─────┐
   │  RMSNorm  │
   └────┬─────┘
        │
   ┌────┴─────┐
   │  LM Head  │  → (B, S, 32000)  logits over vocabulary
   └──────────┘
```

**Key numbers (default config):**

| Metric | Value |
|--------|-------|
| Total parameters | ~1.02B |
| Active parameters per token | ~613M (60%) |
| Hidden size | 1536 |
| Layers | 24 (6 sensory + 12 associative + 6 executive) |
| Attention | 16 query heads, 4 KV heads (GQA), head_dim=96 |
| Total experts | 16 (4 modules × 4 experts) |
| Active experts per token | 4 + 1 shared |
| Expert FFN size | 1536 → 384 → 1536 |
| Shared expert FFN size | 1536 → 2048 → 1536 |
| Dense FFN size (sensory) | 1536 → 4096 → 1536 |

---

## 3. Component-by-Component Walkthrough

### 3.1 Configuration (`MindConfig`)

A `@dataclass` holding every architectural hyperparameter. Key design choices encoded here:

- **Tier boundaries** are explicit integers (`num_sensory_layers`, `num_associative_layers`, `num_executive_layers`) that must sum to `num_hidden_layers`. This is enforced in `__post_init__`.
- **Derived properties** (`total_experts`, `sensory_layer_range`, `executive_layer_range`, `layer_tier()`, `executive_depth()`) let any component query its role without passing extra state.
- `head_dim` is set independently (not derived as `hidden_size // num_attention_heads`) to allow non-standard head sizes for QK-norm stability, though the default satisfies `1536 = 16 × 96`.

### 3.2 Core Primitives

**`RMSNorm`** — Root Mean Square normalisation (Zhang & Sennrich, 2019). Cheaper than LayerNorm (no mean subtraction, no bias). Used at every pre-norm position. Computation is cast to float32 for numerical stability, then cast back.

**`RotaryEmbedding`** — Rotary Position Embedding (Su et al., 2021). Precomputes inverse frequencies with `θ = 1,000,000` (high theta extends effective context length). Returns `(cos, sin)` tensors of shape `(B, S, head_dim)` that are applied multiplicatively to Q and K in attention.

**`apply_rotary_pos_emb`** — Applies the rotation: `q_rot = q * cos + rotate_half(q) * sin`. The `rotate_half` splits the head dimension in two and negates the swap, implementing the 2D rotation matrix in closed form.

**`repeat_kv`** — Expands KV heads to match query heads for GQA. With 16 query heads and 4 KV heads, each KV head is repeated 4×. This is a reshape-only operation (no new parameters).

### 3.3 Grouped-Query Attention (`CognitiveAttention`)

Implements the attention mechanism shared across all three tiers. Structure follows Qwen3-MoE exactly:

```
hidden_states (B, S, D=1536)
    │
    ├── q_proj (1536 → 16×96 = 1536) ──→ q_norm (RMSNorm on dim 96) ──→ RoPE
    ├── k_proj (1536 →  4×96 =  384) ──→ k_norm (RMSNorm on dim 96) ──→ RoPE
    └── v_proj (1536 →  4×96 =  384) ──→ (no norm)
                                              │
                                    KV cache concat (if present)
                                              │
                                    repeat_kv (4 → 16 heads)
                                              │
                                    QK^T / √96  +  causal mask
                                              │
                                    softmax (float32) → matmul V
                                              │
                                    o_proj (1536 → 1536)
```

**QK-norm** is the Qwen3 innovation: applying per-head RMSNorm to queries and keys *before* the dot product. This bounds attention logit magnitudes, preventing training instabilities that arise at scale when attention scores become very large.

**KV cache**: during autoregressive decoding, previously computed K and V tensors are concatenated along the sequence dimension. The causal mask is extended accordingly. Each layer stores and returns its own `(K, V)` tuple.

### 3.4 Feed-Forward Networks (`CognitiveMLP`, `CognitiveExpert`)

Both use the **SwiGLU** architecture (Shazeer, 2020):

```
output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
```

- `gate_proj`: `D → intermediate`
- `up_proj`: `D → intermediate`
- `down_proj`: `intermediate → D`
- `SiLU` (Sigmoid Linear Unit): smooth gating activation

`CognitiveMLP` and `CognitiveExpert` are architecturally identical; they differ only in `intermediate_size`:

| Component | intermediate_size | Params per instance | Role |
|-----------|:-:|:-:|------|
| `CognitiveMLP` (sensory dense) | 4096 | 18.9M | Full-capacity FFN for fast feature extraction |
| `CognitiveMLP` (shared expert) | 2048 | 9.4M | Always-active domain-general pathway |
| `CognitiveExpert` (routed) | 384 | 1.8M | Specialised sub-process within a cognitive module |

The parameter trade-off: 16 small routed experts (16 × 1.8M = 28.3M total, but only 4 × 1.8M = 7.1M active) + 1 shared expert (9.4M always active) = 37.7M total, 16.5M active per MoE layer.

### 3.5 Hierarchical Router (`CognitiveRouter`)

This is the key architectural departure from Qwen3. The router factorises the expert selection into two stages:

**Stage 1 — Module routing:**
```
module_logits = module_gate(x)          # (T, 4)     — linear projection
module_probs  = softmax(module_logits)  # (T, 4)     — P(module m | x)
```

**Stage 2 — Expert routing (intra-module):**
```
expert_logits = expert_gate(x)                              # (T, 16)
expert_logits = reshape(T, 4, 4)                            # group by module
expert_probs  = softmax(expert_logits, dim=-1)              # (T, 4, 4)
                                                            # P(expert j | module m, x)
```

**Combined factored distribution:**
```
P(expert j in module m | x) = P(m | x) × P(j | m, x)

combined_probs = (module_probs[:, :, None] * expert_probs).reshape(T, 16)
```

**Top-k selection:** the top 4 experts are selected from the 16-dimensional combined distribution, and their weights are renormalised to sum to 1 (when `norm_topk_prob=True`).

**Why hierarchical?** The factored distribution induces structure: experts within the same module share a module-level relevance score. This encourages the model to develop functionally specialised modules (analytical, linguistic, associative, social) rather than treating all 16 experts as interchangeable. The intra-module softmax ensures experts within a module compete with each other, promoting intra-module differentiation.

**Router parameters:** only two small linear projections (`module_gate`: 1536 → 4, `expert_gate`: 1536 → 16), adding negligible overhead (~30K params per layer).

### 3.6 Cognitive MoE Block (`CognitiveMoEBlock`)

The complete MoE feedforward layer used in Tiers 2 and 3. Combines three components:

1. **`CognitiveRouter`** — selects 4 experts per token
2. **16 `CognitiveExpert` instances** — the specialised routed FFNs
3. **1 `CognitiveMLP` (shared expert)** — always-active, larger-capacity

**Token dispatch** follows the standard MoE pattern:

```python
# For each expert that received ≥1 token:
for expert_idx in active_experts:
    tokens_for_this_expert = gather(x_flat, assignment_mask[expert_idx])
    expert_output = expert(tokens_for_this_expert)
    expert_output *= routing_weight                    # scale by router probability
    scatter_add(final_hidden, expert_output, indices)  # accumulate into output

# Add shared expert output (runs on all tokens)
final_hidden += shared_expert(x_flat)
```

The `expert_mask` is built via `F.one_hot` on the top-k indices, then permuted to `(E, k, T)` for efficient per-expert gathering. This avoids the memory cost of materialising a full `(T, E)` dispatch matrix.

**Returns:** both the output tensor `(B, S, D)` and the full router probability distribution `(B*S, 16)` — the latter is needed for the load balancing loss.

### 3.7 FracToM Integration (`FracToMIntegration`)

Present **only in executive layers** (layers 18–23). Each instance operates at a specific *mentalizing depth* (0–5) and performs five operations:

#### Step 1: Observation Projection
```
obs = obs_proj(hidden_states)     # (B, S, D=1536) → (B, S, F=256)
```
Compresses the full hidden state into a lower-dimensional observation vector suitable for causal reasoning.

#### Step 2: Causal Structural Equations (Obs → B → D → I)
The BDI components are computed via a differentiable Structural Causal Model with learnable edge strengths `s[0..4]` (sigmoid-normalised to [0,1]):

```
Belief    = f_B(obs × s[0])                           # Obs → B
Desire    = f_D(belief × s[1] + obs × s[2])           # B → D,  Obs → D
Intention = f_I(belief × s[3] + desire × s[4])        # B → I,  D → I
```

Each `f_*` is a 2-layer MLP: `Linear(256, 256) → SiLU → Linear(256, 256)`.

The causal graph is:
```
        Obs
       / | \
      s0 s2  \
     /   |    \
    B ──s1──→ D
    │         │
    s3       s4
    │         │
    └──→ I ←──┘
```

#### Step 3: Epistemic Gating
```
σ = softplus(epistemic_head(belief))     # σ > 0, measures belief uncertainty
confidence = 1 / (1 + σ)                 # ∈ (0, 1], high when belief is certain
```
The BDI enrichment is scaled by `confidence`, suppressing hallucinated mental-state attributions when the model is uncertain.

#### Step 4: BDI → Hidden Enrichment
```
bdi_packed = cat(B, D, I) × confidence       # (B, S, 768) scaled
enrichment = bdi_to_hidden(bdi_packed)        # (B, S, 1536)
gate = sigmoid(output_gate(enrichment))       # (B, S, 1536), init ≈ 0.05
hidden_states += gate × enrichment
```
The output gate is initialised near zero (`bias = -3.0`, so `sigmoid(-3) ≈ 0.05`). This ensures training starts with near-identity behaviour — the FracToM pathway must *earn* its influence during training.

#### Step 5: Perspective Shifting (depth > 0 only)
For executive layers beyond depth 0, a gated cross-depth mechanism aggregates the previous layer's BDI state:

```
prior_h = prior_bdi_proj(cat(B_prev, D_prev, I_prev))    # (B, S, 1536)
p_gate = sigmoid(perspective_gate_net(cat(hidden, prior_h)))  # scalar gate
hidden_states += p_gate × prior_h
```

This models the cognitive operation of "taking another's perspective" — each successive executive layer builds on the previous layer's mentalizing output, creating a hierarchy mapped to Pearl's causal levels:

| Depth | Pearl Level | Cognitive Operation |
|:-----:|:-----------:|:------------------:|
| 0 | Association | Observe and predict patterns |
| 1 | Intervention | "What if I change my belief?" |
| 2–5 | Counterfactual | "What would they believe if they'd seen X?" |

### 3.8 Decoder Layer (`MindDecoderLayer`)

Each layer follows the **pre-norm** convention (normalise → sub-layer → residual add):

```
# Sub-layer 1: Attention
residual = x
x = RMSNorm(x)
x = GQA_Attention(x)
x = residual + x

# Sub-layer 2: FFN or MoE
residual = x
x = RMSNorm(x)
x = Dense_FFN(x)  or  MoE(x)     # depends on tier
x = residual + x

# Sub-layer 3: FracToM (executive only)
if executive:
    x, bdi, info = FracToMIntegration(x, prior_bdi)
```

The layer's tier is determined at construction time via `config.layer_tier(layer_idx)`:
- **Sensory** → `CognitiveMLP` (dense SwiGLU, no routing)
- **Associative** → `CognitiveMoEBlock`
- **Executive** → `CognitiveMoEBlock` + `FracToMIntegration`

### 3.9 Backbone (`MindModel`)

Orchestrates the full decoder:

1. **Embedding**: `nn.Embedding(32000, 1536)` maps token IDs to vectors.
2. **RoPE**: computed once and passed to all layers.
3. **Causal mask**: upper-triangular `-inf` mask, combined with padding mask if provided. Uses `masked_fill` (not multiplication) to avoid `0 × -inf = NaN`.
4. **Layer loop**: iterates through all 24 layers, propagating:
   - Hidden states (always)
   - KV cache tuples (when `use_cache=True`)
   - Router probability matrices (from MoE layers, for aux loss)
   - BDI states (across executive layers via `current_bdi` variable)
5. **Final RMSNorm**: applied to the output of the last layer.

**BDI propagation**: the `current_bdi` variable is `None` until the first executive layer produces a BDI state, then it's passed as `prior_bdi` to each subsequent executive layer. This creates the mentalizing chain: layer 18 → 19 → 20 → 21 → 22 → 23.

### 3.10 Language Model Head (`MindForCausalLM`)

Wraps `MindModel` with:

- **LM head**: `nn.Linear(1536, 32000, bias=False)`, weight-tied with the input embedding by default. This halves the embedding parameter cost.
- **Cross-entropy loss**: standard next-token prediction with shift-by-one alignment and `-100` ignore index.
- **Auxiliary load balancing loss**: added to the main loss, weighted by `router_aux_loss_coef` (default 0.001).

**Weight initialisation** follows Qwen3/LLaMA convention:
- All `nn.Linear` weights: `Normal(0, 0.02)`
- All biases: zero
- `RMSNorm` weights: ones
- FracToM output gates: re-initialised after the global init pass to maintain the near-zero startup (`weight=0`, `bias=-3`).

### 3.11 Load Balancing Loss

Implements the Switch Transformer auxiliary loss (Fedus et al., 2021):

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

where:
- $E$ = number of experts (16)
- $f_i$ = fraction of tokens routed to expert $i$ (from top-k hard assignments)
- $p_i$ = mean routing probability assigned to expert $i$ (from soft distribution)

This loss is minimised when routing is perfectly uniform ($f_i = p_i = 1/E$ for all $i$). It penalises expert collapse (one expert getting all tokens) without preventing useful specialisation (the penalty is mild for moderate imbalance).

**Implementation details:**
- Router probabilities from all MoE layers are concatenated before computing the loss (treating the full stack as one large routing event).
- When an attention mask is provided, padding tokens are excluded from both $f_i$ and $p_i$ computation.
- For KV-cache decoding, `S_new` (the number of newly processed tokens) is derived from the router probability tensor shape, and only the corresponding columns of `attention_mask` are used.

---

## 4. Data Flow

Complete forward pass for a batch of shape `(B=2, S=64)`:

```
input_ids (2, 64)
    │
    ▼  embed_tokens
hidden (2, 64, 1536)               ← RoPE cos/sin computed: (2, 64, 96)
    │                               ← Causal mask built: (1, 1, 64, 64)
    │
    │  ┌─── Layer 0–5 (SENSORY) ───────────────────────────────────────┐
    │  │  Each layer:                                                   │
    │  │    Attn: (2,64,1536) → QKV proj → GQA → (2,64,1536)         │
    │  │    FFN:  (2,64,1536) → SwiGLU(4096) → (2,64,1536)           │
    │  │  Returns: hidden, KV cache pair, no router probs, no BDI      │
    │  └────────────────────────────────────────────────────────────────┘
    │
    │  ┌─── Layer 6–17 (ASSOCIATIVE) ──────────────────────────────────┐
    │  │  Each layer:                                                   │
    │  │    Attn: same as sensory                                       │
    │  │    MoE:  (2,64,1536) → flatten to (128,1536)                 │
    │  │          → Router: module_probs (128,4), expert_probs (128,4,4)│
    │  │          → combined (128,16) → top-4 selection                 │
    │  │          → Expert dispatch: 4 active experts × (subset,384)    │
    │  │          → Shared expert: (128,2048)                           │
    │  │          → Sum → reshape (2,64,1536)                          │
    │  │  Returns: hidden, KV, router_probs (128,16), no BDI           │
    │  └────────────────────────────────────────────────────────────────┘
    │
    │  ┌─── Layer 18–23 (EXECUTIVE) ───────────────────────────────────┐
    │  │  Each layer:                                                   │
    │  │    Attn: same                                                  │
    │  │    MoE:  same as associative                                   │
    │  │    FracToM:                                                    │
    │  │      obs = obs_proj(hidden) → (2,64,256)                      │
    │  │      B = f_B(obs*s0)        → (2,64,256)                      │
    │  │      D = f_D(B*s1 + obs*s2) → (2,64,256)                     │
    │  │      I = f_I(B*s3 + D*s4)   → (2,64,256)                     │
    │  │      confidence = 1/(1+softplus(head(B)))                      │
    │  │      enrichment = bdi_to_hidden(cat(B,D,I) * conf)            │
    │  │      hidden += sigmoid(gate(enrichment)) * enrichment          │
    │  │      [if depth>0]: hidden += p_gate * prior_bdi_proj(prior)    │
    │  │  Returns: hidden, KV, router_probs, BDITensor, causal info    │
    │  └────────────────────────────────────────────────────────────────┘
    │
    ▼  final RMSNorm
logits = lm_head(hidden)  ← (2, 64, 32000)
    │
    ▼  loss = CE(logits[:-1], labels[1:]) + 0.001 × load_balance_loss
```

---

## 5. Parameter Budget Breakdown

The following table breaks down where the ~1.02B parameters reside:

| Component | Count | Params Each | Subtotal | % of Total |
|-----------|:-----:|:-----------:|:--------:|:----------:|
| `embed_tokens` | 1 | 49.2M | 49.2M | 4.8% |
| `lm_head` | 1 | (tied) | 0 | — |
| **Attention per layer** | | | | |
| ∟ `q_proj` (1536→1536) | 24 | 2.36M | 56.6M | 5.5% |
| ∟ `k_proj` (1536→384) | 24 | 0.59M | 14.2M | 1.4% |
| ∟ `v_proj` (1536→384) | 24 | 0.59M | 14.2M | 1.4% |
| ∟ `o_proj` (1536→1536) | 24 | 2.36M | 56.6M | 5.5% |
| ∟ `q_norm` + `k_norm` | 24 | 192 | 4.6K | ~0% |
| **Sensory Dense FFN** | | | | |
| ∟ 3 linear layers (1536↔4096) | 6 | 18.9M | 113.2M | 11.1% |
| **MoE per layer** (Tiers 2–3) | | | | |
| ∟ 16 experts × 3 linears (1536↔384) | 18 | 28.3M | 509.6M | 49.9% |
| ∟ 1 shared expert × 3 linears (1536↔2048) | 18 | 9.4M | 169.9M | 16.6% |
| ∟ Router (`module_gate` + `expert_gate`) | 18 | 30.7K | 0.55M | ~0% |
| **Layer norms** (2 per layer) | 48 | 1536 | 73.7K | ~0% |
| **FracToM Integration** | | | | |
| ∟ `obs_proj` + 3 BDI MLPs + `bdi_to_hidden` + gates | 6 | ~3.5M | ~21.2M | 2.1% |
| **Final RMSNorm** | 1 | 1536 | 1.5K | ~0% |
| | | | **~1.02B** | **100%** |

**Active parameters per token:** ~613M (60%). The saving comes from MoE layers where only 4 of 16 experts fire (but the shared expert always fires).

---

## 6. Scaling Guide — Extending to Larger Models

### 6.1 General Scaling Strategy

MoE models scale along two independent axes:

1. **Total parameters** — increasing expert count, hidden size, or depth adds *capacity* (knowledge stored in weights).
2. **Active parameters** — increasing active experts, hidden size, or shared expert size adds *per-token compute* (inference cost).

The ratio between them is the **sparsity ratio**. MIND at 1B has a 60% active ratio. Larger models typically push this lower (Qwen3-30B-A3B activates only 10% of its parameters).

**Scaling laws for MoE** (Krajewski et al., 2024) suggest:
- Total loss scales primarily with *active* parameters (more compute per token = lower loss).
- But MoE models consistently outperform dense models of equivalent active parameter count, because the inactive experts store additional knowledge that is selectively retrieved.

**Scaling priority order:**

```
 Most impact per parameter dollar
┌──────────────────────────────────────────────┐
│ 1. Increase hidden_size (D)                  │  ← affects everything
│ 2. Increase depth (num_hidden_layers)        │  ← more processing steps
│ 3. Increase expert count (total_experts)     │  ← more capacity, same compute
│ 4. Increase expert FFN size                  │  ← more compute per expert
│ 5. Increase active experts (top_k)           │  ← more compute per token
│ 6. Increase shared expert size               │  ← stronger domain-general base
│ 7. Increase BDI factor dim                   │  ← richer mentalizing
└──────────────────────────────────────────────┘
 Least impact per parameter dollar
```

### 6.2 Concrete Recipes (3B, 8B, 30B)

Below are three concrete configs. All maintain the 3-tier cognitive hierarchy and hierarchical routing, modifying only dimensions.

#### MIND-3B (~3B total, ~1B active)

```python
MindConfig(
    hidden_size=2048,
    num_hidden_layers=32,               # 8 sensory + 16 associative + 8 executive
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=128,                       # 2048 / 16
    dense_intermediate_size=5504,       # ≈ 2.7× hidden
    num_cognitive_modules=4,
    experts_per_module=8,               # 32 total experts
    expert_intermediate_size=512,
    shared_expert_intermediate_size=2816,
    num_experts_per_tok=4,
    num_sensory_layers=8,
    num_associative_layers=16,
    num_executive_layers=8,
    bdi_factor_dim=384,
    vocab_size=32000,
)
```

| Metric | Value |
|--------|-------|
| Total params | ~3.1B |
| Active params | ~1.0B (32%) |
| New vs. 1B | 2× hidden, 2× experts, 1.3× depth |

#### MIND-8B (~8B total, ~2.5B active)

```python
MindConfig(
    hidden_size=3072,
    num_hidden_layers=40,               # 8 sensory + 24 associative + 8 executive
    num_attention_heads=24,
    num_key_value_heads=8,
    head_dim=128,
    dense_intermediate_size=8192,
    num_cognitive_modules=4,
    experts_per_module=16,              # 64 total experts
    expert_intermediate_size=768,
    shared_expert_intermediate_size=4096,
    num_experts_per_tok=6,
    num_sensory_layers=8,
    num_associative_layers=24,
    num_executive_layers=8,
    bdi_factor_dim=512,
    vocab_size=32000,
)
```

| Metric | Value |
|--------|-------|
| Total params | ~8.2B |
| Active params | ~2.5B (30%) |
| New vs. 3B | 1.5× hidden, 2× experts, 1.25× depth |

#### MIND-30B (~30B total, ~3B active)

```python
MindConfig(
    hidden_size=4096,
    num_hidden_layers=48,               # 8 sensory + 32 associative + 8 executive
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    dense_intermediate_size=11008,
    num_cognitive_modules=8,            # more cognitive specialisations
    experts_per_module=16,              # 128 total experts
    expert_intermediate_size=768,
    shared_expert_intermediate_size=6144,
    num_experts_per_tok=8,
    num_sensory_layers=8,
    num_associative_layers=32,
    num_executive_layers=8,
    bdi_factor_dim=768,
    vocab_size=152064,                  # Qwen3-level vocabulary
)
```

| Metric | Value |
|--------|-------|
| Total params | ~30B |
| Active params | ~3B (10%) |
| New vs. 8B | 1.3× hidden, 2× experts, 2× modules, 1.2× depth |

### 6.3 Dimension-by-Dimension Scaling Analysis

#### Hidden Size (`D`)

Most impactful knob. Affects every projection in the model (Q, K, V, O, all FFN layers, embeddings, LM head). Scaling `D` by a factor `k`:

- Attention params scale as $\mathcal{O}(k^2)$ (both input and output dimensions grow)
- FFN params scale as $\mathcal{O}(k^2)$ (if intermediate size is proportional to D)
- Embedding scales as $\mathcal{O}(k)$ (vocabulary is fixed)

**Recommendation:** Scale D in steps of 128 or 256 (for hardware alignment). Keep `D = num_attention_heads × head_dim` satisfied. Head dim of 128 is optimal for most GPU tensor-core configurations.

#### Depth (`num_hidden_layers`)

More layers = more sequential processing steps = better at complex reasoning chains. Params scale linearly with depth.

**Tier ratio recommendation:** Keep sensory and executive layers roughly constant (6–10 each) and scale primarily the associative tier. Rationale: feature extraction stabilises quickly, and 6–10 executive layers already span Pearl's full causal hierarchy. The associative tier is where MoE capacity pays off most.

#### Expert Count

Increasing `experts_per_module` or `num_cognitive_modules` adds capacity without adding per-token compute (if `num_experts_per_tok` stays constant). The per-expert dispatch becomes more selective, improving specialisation.

**Scaling pattern:**
- 1B: 4 modules × 4 experts = 16 total
- 3B: 4 modules × 8 experts = 32 total
- 8B: 4 modules × 16 experts = 64 total
- 30B: 8 modules × 16 experts = 128 total

At 30B, consider increasing `num_cognitive_modules` to 8 to introduce finer cognitive specialisations (e.g., splitting "Analytical" into "Logical" and "Mathematical", splitting "Social" into "Empathy" and "Strategic").

#### Active Experts (`num_experts_per_tok`)

Each additional active expert increases compute proportionally. Diminishing returns above ~8 active experts when the combined routing weight for the 9th expert is typically negligible.

**Recommendation:** Scale slowly. 4 active at 1B, 4–6 at 3–8B, 8 at 30B+.

#### Shared Expert Size

The shared expert represents domain-general processing. It should grow proportionally with the model to maintain its role as a working-memory baseline.

**Recommendation:** Keep `shared_expert_intermediate_size ≈ D × 1.3 to D × 1.5`.

#### BDI Factor Dim

Controls the fidelity of mentalizing. Larger BDI dims allow richer belief-desire-intention representations. However, the impact on total param count is small (FracToM uses ~2% of parameters), so this can be scaled generously.

**Recommendation:** `bdi_factor_dim ≈ D / 4 to D / 6`.

### 6.4 Infrastructure Requirements

#### Memory Estimation

Rough formula for total GPU memory during training (mixed precision, without activation checkpointing):

$$\text{Memory (GB)} \approx \frac{\text{Total Params} \times 18}{10^9} + \text{Activations}$$

(18 bytes = 2 for fp16 params + 4 for fp32 master + 4 for fp32 optimizer state × 2 + 4 for fp32 gradient)

| Model | Params | Memory (params only) | Recommended GPUs |
|-------|--------|---------------------|------------------|
| MIND-1B | 1.0B | ~18 GB | 1× A100 80GB |
| MIND-3B | 3.1B | ~56 GB | 1–2× A100 80GB |
| MIND-8B | 8.2B | ~148 GB | 2–4× A100 80GB |
| MIND-30B | 30B | ~540 GB | 8–16× A100 80GB |

#### Parallelism Strategy

| Scale | Strategy |
|-------|----------|
| ≤3B | Data Parallel (DDP). Single-node, 1–2 GPUs. |
| 3–8B | FSDP (Fully Sharded Data Parallel) or DeepSpeed ZeRO-3. Shard parameters, gradients, and optimizer states across GPUs. |
| 8–30B | FSDP + Expert Parallel (EP). Route different experts to different GPUs. The shared expert is replicated. |
| ≥30B | 3D parallelism: Tensor Parallel (TP) within nodes + Expert Parallel + Pipeline Parallel across nodes. Or Megatron-style MoE parallelism. |

**Expert Parallel** is the MoE-specific optimisation: instead of replicating all 128 experts on every GPU, each GPU holds a subset of experts and tokens are routed across GPUs via all-to-all communication. This trades compute for communication but is essential at 30B+.

### 6.5 Training Improvements for Better Performance

These improvements are **orthogonal to architecture scaling** — they improve quality at any size.

#### Data & Tokenisation

1. **Larger vocabulary.** The default 32K vocabulary is small. Scale to 128K–152K (Qwen3 uses 152,064) with BPE/Unigram tokeniser trained on a diverse multilingual corpus. Larger vocab = fewer tokens per sequence = cheaper training + better representation of rare words.

2. **Data quality over quantity.** Scaling laws show that data quality has a larger effect than data quantity beyond a certain point. Apply aggressive deduplication (MinHash), quality filtering (perplexity-based), and domain mixing (match target distribution).

3. **Curriculum learning.** Start with easy/short sequences, progressively increase difficulty and length. For MIND specifically, interleave ToM-relevant data (social scenarios, planning tasks, dialogue) throughout training to activate FracToM pathways.

#### Optimisation

4. **Learning rate.** Follow the $\mu$P (Maximal Update Parametrization, Yang et al., 2022) scaling rule: for a width factor $k$, scale the learning rate by $1/k$ for attention layers and $1/k^2$ for embeddings. This eliminates expensive LR sweeps at each model size.

5. **Batch size warm-up.** Start with small batches, ramp up following a schedule (McCandlish et al., 2018). Prevents early training divergence especially with MoE.

6. **Router z-loss.** Add a penalty on the squared log of the router logits: $\mathcal{L}_z = \lambda_z \frac{1}{T} \sum_t \log^2(\sum_e \exp(z_{t,e}))$. This prevents router logit explosion and stabilises training at scale (Zoph et al., 2022).

7. **Auxiliary balance loss tuning.** `router_aux_loss_coef` may need adjustment at scale. Too low → expert collapse. Too high → over-smoothed routing that prevents specialisation. Start at 0.001 and increase to 0.01 if monitoring reveals collapsed experts.

#### Training Techniques

8. **Gradient checkpointing.** Exchange ~30% more compute for ~70% memory savings. Essential at ≥8B. Apply selectively to associative and executive layers (they're more memory-heavy due to MoE).

9. **Mixed precision.** Use bf16 for forward/backward (better dynamic range than fp16, no loss scaling needed). Keep master weights and optimizer states in fp32.

10. **Sequence packing.** Pack multiple shorter sequences into one fixed-length sample (with appropriate attention masking) to maximise hardware utilisation and eliminate padding waste.

### 6.6 Architectural Improvements for Better Performance

These are optional enhancements to the architecture that go beyond pure scaling.

#### Attention Enhancements

11. **Flash Attention.** Replace the explicit QK^T matmul + softmax + matmul V sequence with FlashAttention-2 (Dao et al., 2023). This is a purely computational optimisation (no accuracy change) that reduces memory from $\mathcal{O}(S^2)$ to $\mathcal{O}(S)$ and improves wall-clock speed by 2–4×.

    ```python
    # In CognitiveAttention.forward(), replace:
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
    # ... softmax ... matmul v ...
    
    # With:
    from flash_attn import flash_attn_func
    out = flash_attn_func(q, k, v, causal=True, softmax_scale=self.scaling)
    ```

12. **Sliding Window Attention.** For long contexts (>8K tokens), apply local sliding-window attention in lower layers and full attention only in upper layers. This is what Qwen3-MoE does with its `sliding_window` config. Reduces the quadratic cost of attention in early feature-extraction layers where long-range dependencies matter less.

13. **Grouped-Query Attention tuning.** The current 16:4 (Q:KV) head ratio can be adjusted. At larger hidden sizes, a 32:8 ratio or even MQA (1 KV head) can reduce KV cache memory during inference without significant quality loss.

#### MoE Enhancements

14. **Fused expert kernels.** Instead of sequential Python loops over active experts, use fused CUDA kernels (e.g., Megablocks, ScatterMoE) that batch all expert computations into a single kernel launch. This can speed up MoE layers by 2–5× on GPU.

    The current implementation:
    ```python
    for idx in expert_hit:
        e = idx[0].item()
        current_state = x_flat[token_idx]
        expert_out = self.experts[e](current_state)
        ...
    ```
    This Python-level loop has kernel launch overhead proportional to the number of active experts. A fused kernel does one launch.

15. **3D expert weight tensors.** Following Qwen3's `Qwen3MoeExperts`, pack all expert weights into 3D tensors `(E, D, intermediate)` and use batched `torch.bmm` or `torch.einsum` for parallel expert computation. This eliminates the Python dispatch loop entirely:

    ```python
    # Instead of ModuleList + loop:
    class FusedExperts(nn.Module):
        def __init__(self, num_experts, hidden_size, intermediate_size):
            super().__init__()
            self.gate_weight = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
            self.up_weight = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
            self.down_weight = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
    
        def forward(self, x, expert_indices):
            # x: (T, k, D), expert_indices: (T, k)
            gate = torch.einsum('tke,eih->tki', x, self.gate_weight[expert_indices])
            up = torch.einsum('tke,eih->tki', x, self.up_weight[expert_indices])
            hidden = F.silu(gate) * up
            return torch.einsum('tki,ehi->tke', hidden, self.down_weight[expert_indices])
    ```

16. **Expert capacity factor.** Add a hard cap on the number of tokens any single expert can process (e.g., `capacity_factor × T / E`). Tokens exceeding capacity overflow to the shared expert. This prevents expert load imbalance from causing OOM during training.

17. **Auxiliary-loss-free routing.** Recent work (Wang et al., 2024) shows that removing the aux loss and instead using a simple bias term per expert can achieve better specialisation without the load balancing loss's tendency to over-smooth routing. Worth experimenting with at ≥8B scale.

#### FracToM Enhancements

18. **Deeper BDI structural equations.** Replace the 2-layer MLP in each structural equation with a deeper or wider network. At 30B, the BDI factor dim can be 768 and each equation can be a 3-layer MLP with residual connections:

    ```python
    class DeepStructuralEquation(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 2), nn.SiLU(),
                nn.Linear(dim * 2, dim * 2), nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )
        def forward(self, x):
            return x + self.net(x)  # residual for stability
    ```

19. **Multi-head perspective shifting.** Replace the single scalar perspective gate with multi-head cross-attention over prior BDI states (all prior layers, not just the immediately previous one):

    ```python
    # Current: scalar gate over prior_bdi[-1]
    # Enhanced: cross-attention over all prior BDI states
    class MultiHeadPerspective(nn.Module):
        def __init__(self, hidden_size, num_heads, num_prior_depths):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads)
            # Query: current hidden state
            # Key/Value: stack of all prior BDI projections
    ```

    This allows executive layers to attend to the full mentalizing history, not just the immediately preceding depth.

20. **Counterfactual intervention mechanism.** At depths ≥ 2 (counterfactual reasoning), add an explicit intervention operation that can modify specific BDI components before propagation:

    ```python
    if self.pearl_level == "counterfactual":
        # Generate counterfactual perturbation
        delta_B = self.counterfactual_head(hidden_states)  # "what if belief were different?"
        belief_cf = belief + delta_B
        # Re-run structural equations with perturbed belief
        desire_cf = self.desire_eq(belief_cf * s[1] + obs * s[2])
        intention_cf = self.intention_eq(belief_cf * s[3] + desire_cf * s[4])
    ```

    This directly implements Pearl's do-calculus in the network: intervening on a variable and propagating the effect through structural equations.

---

## Summary

MIND is a cognitively-grounded MoE architecture that goes beyond standard sparse models by organising experts into functional cognitive modules, routing hierarchically, stratifying the decoder into processing tiers, and integrating causal Theory-of-Mind reasoning in executive layers. To scale it:

1. **Scale hidden size and depth first** — they have the broadest impact.
2. **Scale expert count for capacity** — add more experts to store more knowledge without proportional compute increase.
3. **Add infrastructure support** — Flash Attention, fused expert kernels, Expert Parallel, gradient checkpointing.
4. **Enhance the cognitive mechanisms** — deeper BDI equations, multi-head perspective shifting, counterfactual interventions.
5. **Improve training** — larger vocab, quality data, router z-loss, $\mu$P parametrisation, curriculum learning with ToM-relevant data.
