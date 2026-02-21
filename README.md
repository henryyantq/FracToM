# FracToM — Fractal Theory-of-Mind with Structural Causal Discovery

> A recursive neural architecture that unifies **fractal self-similarity**, **BDI mental-state factoring**, **Pearl's Structural Causal Model** framework, and **FractalGen-inspired training techniques** for hierarchical Theory-of-Mind reasoning.

## Motivation

Standard deep learning architectures used for social reasoning (e.g., opponent modelling, false-belief tasks, intention prediction) lack the structural inductive biases needed for *recursive* mentalizing — the "I think that you think that I think …" reasoning chain central to Theory of Mind (ToM). FracToM addresses this by embedding four theoretical frameworks directly into the network topology:

1. **Fractal Self-Similarity** (Mandelbrot, 1982; Larsson et al., 2017) — mentalizing at every recursive depth uses a structurally identical processing template, differing only in learned weights.
2. **BDI Architecture** (Bratman, 1987) — latent representations are explicitly factored into Belief (epistemic), Desire (motivational), and Intention (conative) subspaces.
3. **Structural Causal Models** (Pearl, 2009; Zheng et al., 2018) — a differentiable causal graph over BDI variables is learned end-to-end, enabling intervention and counterfactual reasoning within the forward pass.
4. **FractalGen-Inspired Training** (Li, Sun, Fan & He, 2025) — three techniques adapted from fractal image generation: per-depth capacity scheduling, guiding-belief FiLM conditioning, and auxiliary deep supervision.

---

## Architecture

FracToM ships with two equivalent implementations:

| File | Backend | Best for |
|------|---------|----------|
| `nn.py` (~2 700 lines) | **PyTorch** | CUDA / MPS / CPU training |
| `mlx_nn.py` (~2 200 lines) | **Apple MLX** | Metal GPU acceleration on Apple Silicon |

Both expose the same `FracToMNet`, `FracToMLoss`, `InterpretabilityReport`, `analyse_mentalizing_depth`, `extract_causal_graph`, and `extract_bdi_activations` APIs. The MLX port uses `nn.value_and_grad` for training, Taylor-series matrix exponentials (order 8) in place of `torch.matrix_exp`, and `mx.array` throughout.

### MIND — ~1 B Mixture-of-Experts Cognitive Language Model

Built on top of FracToM, **MIND** (`mind.py` / `mlx_mind.py`) is a ~1 billion-parameter causal language model that integrates the BDI mental-state framework directly into a Transformer decoder. Key design choices:

| Aspect | Detail |
|--------|--------|
| **Scale** | ~1.07 B total parameters, ~57 % active per token |
| **Layers** | 24 decoder layers organised into a cognitive hierarchy: 6 *sensory*, 12 *associative*, 6 *executive* |
| **Attention** | Grouped-Query Attention (16 query / 4 KV heads) with RoPE |
| **MoE** | Hierarchical 2-stage routing — 4 modules × 4 experts = 16 total experts per MoE layer, top-2 gating |
| **FracToM tie-in** | Executive layers carry a `FracToMIntegration` module that maintains and updates BDI (Belief-Desire-Intention) tensors across the sequence |
| **Backends** | PyTorch (`mind.py`) and Apple MLX (`mlx_mind.py`) — same public API |

> **Detailed documentation** — architecture walk-through, parameter budget, scaling recipes (3 B / 8 B / 30 B), and the full MLX porting guide — is available in **[README_llm.md](README_llm.md)**.

Key components:

### Fractal Mentalizing Columns

The network contains $K{+}1$ parallel columns, where the $k$-th column encodes $k$-th order ToM:

$$M_0(x) = \varphi(x), \qquad M_k(x) = \psi_k\!\bigl(x,\; M_0, \ldots, M_{k-1}\bigr)$$

Each $\psi_k$ is a stack of **SelfSimilarBlocks** — a Transformer-style block (multi-head self-attention + GEGLU FFN) with an additional **BDI re-factoring** projection that decomposes the output back into a `BDIState` triple. Blocks are architecturally identical across columns (self-similar), with independent parameters.

Cross-depth information flows via **PerspectiveShiftAttention**: higher-order columns attend to lower-order column outputs through a learned perspective transform (query/key transforms), modelling the cognitive operation *"simulate what they would perceive"*.

### Epistemic Gating

An **EpistemicGate** attached to each column scales its output by a learned confidence score:

$$\text{confidence}_k = \frac{1}{1 + \sigma_k}, \qquad \sigma_k = \text{softplus}\!\bigl(g(h_k)\bigr)$$

This suppresses unreliable high-order mental-state attributions — deeper mentalizing is noisier because it relies on sparser evidence about the other agent's internal states.

### Structural Causal Model (SCM)

A fully differentiable SCM learns the causal graph among BDI variables:

| Module | Role |
|--------|------|
| **LearnableCausalGraph** | Parameterises a weighted adjacency matrix $A \in [0,1]^{d \times d}$ with a sigmoid over learnable logits. The **NOTEARS** acyclicity constraint $h(A) = \mathrm{tr}(e^{A \circ A}) - d = 0$ (Zheng et al., 2018) is incorporated as a differentiable penalty in the loss. Prior structure (Obs→B, B→D, B→I, D→I) is initialised as a warm start but fully learnable. |
| **StructuralEquationNetwork** | Neural structural equation for each endogenous variable: $V_j = f_j(\mathbf{Pa}_j, \varepsilon_j)$ where $\mathbf{Pa}_j$ is the set of causal parents and $\varepsilon_j$ is exogenous noise. |
| **StructuralCausalModel** | Composes graph + equations, supporting three operations: `forward` (standard inference), `intervene` ($\mathrm{do}(X{=}x)$, severing incoming edges), and `counterfactual` (abduction → action → prediction). |

### Pearl's Causal Hierarchy Routing

The **CausalHierarchyRouter** maps mentalizing depth to Pearl's three causal levels via a learned soft-routing:

| Pearl Level | Cognitive Parallel | Default Column Mapping |
|-------------|-------------------|----------------------|
| **L1 — Association** $P(Y \mid X)$ | Direct perception / pattern matching | Column 0 (depth 0) |
| **L2 — Intervention** $P(Y \mid \mathrm{do}(X))$ | Metacognition / active probing | Column 1 (depth 1) |
| **L3 — Counterfactual** $P(Y_{x'} \mid X, Y)$ | "What if …" reasoning / false-belief | Columns 2+ (depth ≥ 2) |

This mapping reflects established developmental evidence: children acquire associative understanding before causal interventions, and counterfactual reasoning matures last (Gopnik & Wellman, 2012).

### Cross-Depth Causal Discovery

The **CausalDiscoveryModule** discovers causal links *between* mentalizing columns using a pairwise bilinear scoring function with a separate NOTEARS DAG constraint. This reveals which ToM levels causally influence others — a useful probe for studying the computational structure of mentalizing itself.

### FractalGen-Inspired Enhancements

Three techniques inspired by fractal image generation (Li, Sun, Fan & He, 2025; arXiv:2502.17437) are integrated into FracToM, adapted from the visual domain to recursive mentalizing:

#### 1. Per-Depth Capacity Scheduling

When `capacity_schedule="decreasing"`, each column $k$ operates at a progressively smaller hidden dimension:

$$d_k = \text{quantize}\!\left(D \cdot \left(1 - \frac{k}{K+1} \cdot 0.5\right),\; q\right), \qquad q = \text{lcm}(N_{\text{BDI}},\, N_{\text{heads}})$$

where $D$ is the base `hidden_dim` and $q$ ensures divisibility by both the number of BDI factors and attention heads. This mirrors FractalGen's insight that coarser fractal levels need more capacity than finer ones. In the mentalizing context, lower-order ToM (column 0: direct perception) requires richer representations than higher-order ToM (column $K$: abstract recursive belief), reflecting Karmiloff-Smith's (1992) *Representational Redescription* hypothesis.

Linear projection layers (`col_input_projs`, `col_output_projs`) adapt between each column's native dimension and the shared `hidden_dim` space for cross-column communication.

**Effect**: ~45% parameter reduction with `capacity_schedule="decreasing"` vs `"uniform"` (verified by smoke test), without accuracy degradation.

#### 2. Guiding Belief Module (FiLM Conditioning)

The **GuidingBeliefModule** implements Feature-wise Linear Modulation (FiLM; Perez et al., 2018) to inject a coarse "gist" from the observation into each column's representation:

$$h_k' = \gamma_k(x) \odot h_k + \beta_k(x)$$

where $\gamma_k, \beta_k$ are per-column affine transforms learned from a shared gist embedding of the input. This is analogous to FractalGen's *guiding pixel* — a low-resolution signal that steers generation at every fractal level.

**Identity-preserving initialisation**: $\gamma$ weights are initialised to produce $\mathbf{1}$ and $\beta$ to produce $\mathbf{0}$, so the module starts as a no-op and the network can gradually learn to use it. This avoids destabilising pre-trained representations.

#### 3. Auxiliary Deep Supervision

When `auxiliary_heads=True`, each column $k$ gets its own classification head that produces per-column logits:

$$\hat{y}_k = W_k \cdot h_k + b_k$$

A weighted auxiliary loss is added:

$$\mathcal{L}_{\text{aux}} = \frac{\lambda_{\text{aux}}}{K+1} \sum_{k=0}^{K} \ell_{\text{CE}}(\hat{y}_k, y)$$

This is directly inspired by FractalGen's per-level loss — every fractal depth generates its own prediction, which prevents gradient starvation in deeper columns and provides a diagnostic signal for which columns are contributing meaningfully.

### Additional Modules

| Module | Purpose |
|--------|--------|
| **MentalStateEncoder** | Maps raw observations to an initial BDI triple + epistemic uncertainty $\sigma$. Shared trunk → three parallel projection heads. |
| **GuidingBeliefModule** | FiLM-style conditioning: compresses input to a gist vector, then produces per-column $\gamma, \beta$ affine modulation. Identity-preserving init ($\gamma{=}1, \beta{=}0$). |
| **BeliefRevisionModule** | Approximate Bayesian updating after column join: $\text{posterior} = g \cdot \text{evidence} + (1{-}g) \cdot \text{prior}$ where $g$ is a learned gate (Friston, 2010). |
| **MentalizingJoin** | Attention-weighted aggregation $\sum_k \alpha_k(x) \cdot h_k$ of all column outputs, where $\alpha$ is normalised (softmax) and input-dependent. |
| **FractalDropPath** | Stochastically drops entire columns during training (adapted from FractalNet). Drop probability decreases for deeper columns over training, emulating the developmental timeline of ToM acquisition. |
| **ClassificationHead** / **ToMPredictionHead** | Task-specific output layers. The ToM head outputs a predicted BDI state for what another agent is thinking. |

### Full Forward Pass

```text
Input
  │
  ▼
MentalStateEncoder  (obs → BDI₀ + σ₀)
  │
  ├──────────────┬──────────────┬── … ──┐
  ▼              ▼              ▼       ▼
Guiding         Guiding        Guiding  Guiding    ← GuidingBeliefModule
Belief_0        Belief_1       Belief_2 Belief_K     (FiLM: γ·h + β)
  │              │              │       │
  ▼              ▼              ▼       ▼
Column_0       Column_1      Column_2  Column_K   ← FractalMentalizingColumn
(dim=D)        (dim=d₁)      (dim=d₂) (dim=d_K)    (per-depth capacity)
  │              │              │       │
  ├──── Aux_0    ├──── Aux_1    ├─ …    ├── Aux_K  ← Auxiliary Heads
  │              │              │       │            (deep supervision)
  ▼              ▼              ▼       ▼
SCM L1         SCM L2         SCM L3   SCM L3     ← StructuralCausalModel
(Association)  (Intervention) (CF)     (CF)         + CausalHierarchyRouter
  │              │              │       │
  ▼              ▼              ▼       ▼
EpistemicGate  EpistemicGate  EpistemicGate …     ← confidence scaling
  │              │              │       │
  └──────┬───────┴──────────────┴───────┘
         │
         ▼
Cross-Depth Causal Discovery       ← inter-column DAG
         │
         ▼
Attention-Weighted Join             ← α_k soft selection
         │
         ▼
BeliefRevisionModule                ← Bayesian gated update
         │
         ▼
Task Head                           ← classification / ToM prediction
```

### Interpretability Report

Every forward pass with `return_interpretability=True` produces an `InterpretabilityReport` containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `depth_weights` | $(B, K{+}1)$ | Attention weight $\alpha_k$ assigned to each mentalizing depth |
| `bdi_states` | dict | Per-column, per-block BDI triples (column-native dimensions) |
| `projected_bdi_states` | dict | Per-column BDI triples projected to common `factor_dim` (for cross-column comparison) |
| `column_uncertainties` | dict → $(B, 1)$ | Per-column epistemic uncertainty $\sigma_k$ |
| `causal_adjacency` | $(4, 4)$ | Learned BDI causal graph (Obs, B, D, I) |
| `causal_hierarchy_weights` | dict → $(B, 3)$ | Per-column soft routing over Pearl's 3 levels |
| `cross_depth_adjacency` | $(K{+}1, K{+}1)$ | Discovered inter-column causal structure |
| `dag_penalty` | scalar | NOTEARS acyclicity value ($0 =$ valid DAG) |
| `counterfactual_distances` | dict → float | Per-column L2 factual–counterfactual distance |
| `auxiliary_logits` | dict → $(B, C)$ | Per-column classification logits from auxiliary heads |
| `guiding_gists` | dict → $(\gamma, \beta)$ | Per-column FiLM modulation parameters |
| `column_dims` | list of int | Per-column hidden dimensions (reflects capacity schedule) |

### Loss Function (`FracToMLoss`)

The composite loss combines seven terms:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{BDI}} + \lambda_2 \mathcal{L}_{\text{unc}} + \lambda_3 \mathcal{L}_{\text{depth}} + \lambda_4 \mathcal{L}_{\text{DAG}} + \lambda_5 \mathcal{L}_{\text{sparse}} + \lambda_6 \mathcal{L}_{\text{CF}}$$

| Term | Purpose |
|------|---------|
| $\mathcal{L}_{\text{task}}$ | Cross-entropy (classification) or MSE (regression) |
| $\mathcal{L}_{\text{BDI}}$ | Cosine distance between adjacent columns' BDI states (consistency) |
| $\mathcal{L}_{\text{unc}}$ | ECE-inspired calibration: higher $\sigma$ on incorrect predictions |
| $\mathcal{L}_{\text{depth}}$ | Entropy bonus on $\alpha$ to encourage exploring all mentalizing depths |
| $\mathcal{L}_{\text{DAG}}$ | NOTEARS acyclicity constraint on the learned causal graph |
| $\mathcal{L}_{\text{sparse}}$ | L1 on causal adjacency (Occam's razor for causal structure) |
| $\mathcal{L}_{\text{CF}}$ | Ordering penalty: deeper columns should show larger counterfactual distances |

---

## Benchmark Task: Recursive Social Dilemma (`collab_train.py`)

`collab_train.py` defines a synthetic 6-way classification task designed so that each class requires a different depth of recursive mentalizing. Two agents interact over 8 rounds in a resource-sharing game with hidden types (cooperative, competitive, conditional, deceptive), cheap-talk signalling, and nested beliefs.

### Target Classes and Required ToM Depth

| Class | Label | Required Reasoning |
|-------|-------|--------------------|
| 0 | `TRANSPARENT_COOPERATE` | Surface pattern matching (depth 0) |
| 1 | `INFORMED_COOPERATE` | Belief–action alignment checking (depth 1) |
| 2 | `DETECTED_DECEPTION` | Signal–action gap detection via other-modelling (depth 2) |
| 3 | `HIDDEN_DECEPTION` | Subtle trait analysis + late-round patterns (depth 2–3) |
| 4 | `MUTUAL_DEFECTION` | Recognising antagonistic dynamics (depth 1–2) |
| 5 | `RECURSIVE_BELIEF_ERROR` | Full recursive mentalizing — cascading wrong second-order beliefs (depth 3) |

The key design principle: a standard MLP can solve classes 0–1 from surface features, but classes 3 and 5 specifically require FracToM's deeper mentalizing columns and causal reasoning.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with all enhancements (default — PyTorch, auto-selects MPS/CUDA/CPU)
python collab_train.py --epochs 60 --hidden-dim 120 --depth 3

# Train on Apple Silicon with MLX (Metal GPU)
python collab_train.py --mlx --epochs 60 --hidden-dim 120 --depth 3

# Ablation: uniform capacity (no FractalGen scheduling)
python collab_train.py --capacity-schedule uniform --epochs 60

# Ablation: disable guiding belief and auxiliary heads
python collab_train.py --no-guiding-belief --no-auxiliary-heads

# Ablation: train without SCM
python collab_train.py --no-causal-model --epochs 60

# Full control over all hyperparameters
python collab_train.py \
  --causal-model \
  --causal-noise-dim 16 \
  --lambda-dag 0.1 \
  --lambda-causal-sparsity 0.005 \
  --lambda-counterfactual 0.01 \
  --capacity-schedule decreasing \
  --guiding-belief --gist-dim 32 \
  --auxiliary-heads --lambda-auxiliary 0.1 \
  --depth 3

# Same, but on Apple Silicon via MLX
python collab_train.py --mlx \
  --causal-model \
  --capacity-schedule decreasing \
  --guiding-belief --auxiliary-heads \
  --epochs 60 --depth 3
```

### Command-Line Arguments

#### Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | `120` | Hidden dimension (must be divisible by $\text{lcm}(3, \text{heads})$) |
| `--depth` | `3` | Maximum recursive mentalizing depth ($K$); columns = $K{+}1$ |
| `--blocks` | `1` | SelfSimilarBlocks per column |
| `--heads` | `4` | Attention heads |
| `--ff-mult` | `2` | Feed-forward expansion ratio |
| `--dropout` | `0.1` | Dropout rate |
| `--drop-path` | `0.1` | Base column drop-path probability |
| `--capacity-schedule` | `decreasing` | Per-column dim schedule: `"decreasing"` (FractalGen-inspired) or `"uniform"` |
| `--guiding-belief` / `--no-guiding-belief` | on | Enable/disable guiding belief FiLM module |
| `--gist-dim` | `32` | Gist embedding dimension for the guiding belief module |
| `--auxiliary-heads` / `--no-auxiliary-heads` | on | Enable/disable per-column auxiliary classification heads |
| `--causal-model` / `--no-causal-model` | on | Enable/disable Structural Causal Model |
| `--causal-noise-dim` | `16` | Exogenous noise dimension for structural equations |

#### Backend

| Argument | Default | Description |
|----------|---------|-------------|
| `--mlx` | off | Use Apple MLX backend (Metal GPU) instead of PyTorch |
| `--cpu` | off | Force CPU (PyTorch path only) |

#### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `60` | Training epochs |
| `--batch-size` | `128` | Batch size |
| `--lr` | `3e-4` | Learning rate (AdamW) |
| `--weight-decay` | `1e-2` | AdamW weight decay |
| `--samples` | `12000` | Total synthetic samples (min 600 for class balance) |

#### Loss Weights ($\lambda$)

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda-bdi` | `0.01` | BDI consistency between adjacent depths |
| `--lambda-uncertainty` | `0.005` | Uncertainty calibration |
| `--lambda-depth-entropy` | `0.01` | Depth exploration entropy bonus |
| `--lambda-dag` | `0.1` | DAG acyclicity penalty |
| `--lambda-causal-sparsity` | `0.005` | Causal graph L1 sparsity |
| `--lambda-counterfactual` | `0.01` | Counterfactual ordering |
| `--lambda-auxiliary` | `0.1` | Auxiliary deep supervision (per-column classification) |
| `--lambda-auxiliary` | `0.1` | Auxiliary deep supervision (per-column classification) |

---

## Interpretability and Visualisation

After training, the script generates the following plots in `./visualizations/`:

| File | Content |
|------|---------|
| `causal_bdi_adjacency.png` | Heatmap of the learned causal graph over Observation, Belief, Desire, and Intention. Useful for verifying whether the network recovers the expected Obs→B→D→I causal structure. |
| `causal_pearl_hierarchy.png` | Per-column soft routing over Pearl's three causal levels. Expected pattern: Column 0 dominant in Association, Column 1 in Intervention, Column 2+ in Counterfactual. |
| `causal_cross_depth.png` | Inter-column causal adjacency matrix. Reveals which mentalizing levels causally depend on others. |
| `tom_depth_weights.png` | Distribution of dominant ToM depth $\arg\max_k \alpha_k$ across test samples, plus a per-sample heatmap of $\alpha$. |
| `tom_uncertainty.png` | Mean epistemic uncertainty $\bar{\sigma}_k$ by ToM depth. Expected: monotonically increasing with depth. |

The script also prints a textual interpretability summary including learned causal edges (with strength), Pearl hierarchy routing, counterfactual distances, and DAG penalty value.

### Programmatic Access

**PyTorch**

```python
import torch
from nn import FracToMNet, FracToMLoss, extract_causal_graph, analyse_mentalizing_depth

model = FracToMNet(
    input_dim=91, hidden_dim=120, mentalizing_depth=3,
    causal_model=True, causal_noise_dim=16,
    capacity_schedule="decreasing",   # FractalGen-inspired
    guiding_belief=True, gist_dim=32, # FiLM conditioning
    auxiliary_heads=True,             # deep supervision
)

x = torch.randn(32, 91)
logits, report = model(x, return_interpretability=True)

# Per-column capacity
print(report.column_dims)             # [120, 96, 72, 60]

# Auxiliary head logits (deep supervision)
for k, aux in report.auxiliary_logits.items():
    print(f"Column {k}: {aux.shape}")  # (32, num_classes)

# Guiding belief FiLM parameters
for k, (gamma, beta) in report.guiding_gists.items():
    print(f"Column {k}: ‖γ‖={gamma.norm():.3f}  ‖β‖={beta.norm():.3f}")

# Inspect learned causal structure
cg = extract_causal_graph(report)
print(cg["bdi_edges"])               # [("Obs", "Belief", 0.92), ...]
print(cg["hierarchy_weights"])        # {0: tensor([0.80, 0.12, 0.08]), ...}
print(cg["counterfactual_distances"]) # {0: 0.01, 1: 0.15, 2: 0.38}

# Textual summary
print(analyse_mentalizing_depth(report))
```

**Apple MLX** — identical API, different import:

```python
import mlx.core as mx
from mlx_nn import FracToMNet, FracToMLoss, extract_causal_graph, analyse_mentalizing_depth

model = FracToMNet(
    input_dim=91, hidden_dim=120, mentalizing_depth=3,
    causal_model=True, causal_noise_dim=16,
    capacity_schedule="decreasing",
    guiding_belief=True, gist_dim=32,
    auxiliary_heads=True,
)
mx.eval(model.parameters())

x = mx.random.normal((32, 91))
logits, report = model(x, return_interpretability=True)

print(report.column_dims)             # [120, 96, 72, 60]
print(analyse_mentalizing_depth(report))

cg = extract_causal_graph(report)
print(cg["bdi_edges"])
```

---

## Theoretical References

- Bratman, M. E. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Gopnik, A., & Wellman, H. M. (2012). Reconstructing constructivism: causal models, Bayesian learning mechanisms, and the theory theory. *Psychological Bulletin*, 138(6), 1085–1108.
- Karmiloff-Smith, A. (1992). *Beyond Modularity: A Developmental Perspective on Cognitive Science*. MIT Press.
- Larsson, G., Maire, M., & Shakhnarovich, G. (2017). FractalNet: Ultra-deep neural networks without residuals. *ICLR 2017*.
- Li, T., Sun, Y., Fan, H., & He, K. (2025). Autoregressive image generation with randomized quantization and fractal self-similarity. *arXiv:2502.17437*.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI 2018*.
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4), 515–526.
- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: continuous optimization for structure learning. *NeurIPS 2018*.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA, MPS on macOS, or CPU)
- Matplotlib ≥ 3.7
- **Optional — Apple Silicon acceleration:** [MLX](https://ml-explore.github.io/mlx/) ≥ 0.30 (`pip install mlx`)

```bash
pip install -r requirements.txt

# For MLX support on Apple Silicon (M1/M2/M3/M4)
pip install mlx
```

## License

FracToM Non-Commercial Source-Available License (Version 1.0). See [LICENSE](LICENSE) for details.
