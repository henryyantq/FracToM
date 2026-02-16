# FracToM — Fractal Theory-of-Mind Neural Network

## FracToM — 分形心智理论神经网络

> **TL;DR** 融合分形自相似性与心智理论，具有完全可解释性的递归思维神经网络。

---

## Table of Contents | 目录

1. [Overview | 概述](#overview--概述)
2. [Theoretical Foundations | 理论基础](#theoretical-foundations--理论基础)
3. [Architecture | 架构](#architecture--架构)
4. [Quick Start | 快速开始](#quick-start--快速开始)
5. [Scenario Trainers | 场景训练脚本](#scenario-trainers--场景训练脚本)
6. [Adapting to New Datasets | 适配新数据集](#adapting-to-new-datasets--适配新数据集)
7. [Expanding the Architecture | 扩展架构](#expanding-the-architecture--扩展架构)
8. [Advancing the Research | 推进研究](#advancing-the-research--推进研究)
9. [API Reference | API 参考](#api-reference--api-参考)
10. [Training Guide | 训练指南](#training-guide--训练指南)
11. [Interpretability Toolkit | 可解释性工具](#interpretability-toolkit--可解释性工具)
12. [FAQ | 常见问题](#faq--常见问题)
13. [Citation | 引用](#citation--引用)
14. [License | 许可证](#license--许可证)

---

## Overview | 概述

**FracToM** (Fractal Theory-of-Mind) is a novel neural architecture that unifies **fractal self-similarity** with **Theory-of-Mind (ToM)** inspired hierarchical mentalizing.  It produces a network that is simultaneously more interpretable, more theoretically grounded, and structurally richer than standard deep architectures.

**FracToM**（分形心智理论）是一种新颖的神经网络架构，将**分形自相似性**与受**心智理论（ToM）**启发的分层心智化统一起来。它产生的网络同时具备更强的可解释性、更坚实的理论基础，以及比标准深度架构更丰富的结构。

### Key Properties | 核心特性

| Property | Description | 特性 | 说明 |
|----------|-------------|------|------|
| **Fractal self-similarity** | Every mentalizing column shares the same block topology | **分形自相似性** | 每个心智化列共享相同的模块拓扑 |
| **BDI-factored latents** | All internal representations decompose into Belief-Desire-Intention | **BDI 分解潜变量** | 所有内部表示分解为信念-欲望-意图 |
| **Interpretable by construction** | Depth weights, BDI states, and uncertainty are directly readable | **结构性可解释** | 深度权重、BDI 状态和不确定性可直接读取 |
| **Developmental curriculum** | Deeper columns unlock progressively during training | **发展性课程** | 更深的列在训练中逐步解锁 |
| **Bayesian belief revision** | Gated prior-evidence integration at the output | **贝叶斯信念修正** | 在输出端进行门控先验-证据整合 |

### Requirements | 环境要求

```
Python ≥ 3.9
PyTorch ≥ 2.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

安装依赖：

```bash
pip install -r requirements.txt
```

No other dependencies are required for the core architecture.

核心架构不需要其他依赖项。

---

## Theoretical Foundations | 理论基础

FracToM draws on five pillars of cognitive science, philosophy of mind, and mathematics:

FracToM 建立在认知科学、心灵哲学和数学的五大支柱之上：

### 1. Fractal Self-Similarity | 分形自相似性

*Mandelbrot (1982); Larsson et al. (2017)*

Processing blocks at every mentalizing depth share the **same architectural template** (an iterated function system on representations), differing only in learned parameters. This mirrors how fractals exhibit identical structure at every scale.

每个心智化深度的处理模块共享**相同的架构模板**（表示空间上的迭代函数系统），仅在学习参数上有所不同。这反映了分形在每个尺度上展现相同结构的特征。

**Mathematical formulation | 数学表述:**

$$M_0(x) = \varphi(x) \quad \text{(direct perception | 直接感知)}$$

$$M_k(x) = \psi_k(x, M_0, M_1, \ldots, M_{k-1}) \quad \text{(k-th order mentalizing | k 阶心智化)}$$

where $\psi_k$ is *structurally self-similar* to $\psi_j$ for all $j \neq k$.

其中 $\psi_k$ 对所有 $j \neq k$ 与 $\psi_j$ *结构自相似*。

### 2. BDI Architecture | BDI 架构

*Bratman (1987)*

Mental states are factored into three components:

心理状态被分解为三个组成部分：

- **B**eliefs (epistemic) — what the agent thinks is true | **信念**（认知性的）—— 智能体认为什么是真实的
- **D**esires (motivational) — what it wants | **欲望**（动机性的）—— 它想要什么
- **I**ntentions (conative) — what it plans to do | **意图**（意志性的）—— 它计划做什么

Each mentalizing level produces an explicit BDI triple, enabling direct probing without external classifiers.

每个心智化层级产生一个显式的 BDI 三元组，无需外部分类器即可直接探测。

### 3. Hierarchical Mentalizing | 分层心智化

*Premack & Woodruff (1978)*

ToM is inherently recursive — "I believe that *you* believe that *I* intend…". The recursion depth maps directly to fractal column depth:

心智理论本质上是递归的——"我相信*你*相信*我*打算……"。递归深度直接映射到分形列深度：

| Column | Cognitive Interpretation | 列 | 认知解释 |
|--------|-------------------------|-----|---------|
| 0 | Direct stimulus→response (no ToM) | 0 | 直接刺激→反应（无心智理论） |
| 1 | Metacognition / self-model | 1 | 元认知 / 自我模型 |
| 2 | Basic other-modelling | 2 | 基础他人建模 |
| 3 | 2nd-order ToM ("I think you think I think…") | 3 | 二阶心智理论（"我认为你认为我认为……"） |
| k | k-th order mentalizing | k | k 阶心智化 |

### 4. Bayesian Brain / Active Inference | 贝叶斯大脑 / 主动推理

*Friston (2010)*

The `BeliefRevisionModule` implements approximate Bayesian updating:

`BeliefRevisionModule` 实现近似贝叶斯更新：

$$\text{posterior} = g \odot \text{evidence} + (1-g) \odot \text{prior}$$

where $g$ is a learned gate approximating a log-likelihood ratio.

其中 $g$ 是一个学习到的门控，近似对数似然比。

### 5. Fractal Drop-Path | 分形丢弃路径

*Adapted from FractalNet (Larsson et al., 2017)*

Entire mentalizing columns can be stochastically dropped during training, which:

在训练过程中可以随机丢弃整个心智化列，这实现了：

1. Regularises the network | 正则化网络
2. Forces every depth to carry independent meaning | 迫使每个深度承载独立意义
3. Mirrors cognitive development where deeper mentalizing emerges gradually | 模拟更深层心智化逐渐出现的认知发展过程

---

## Architecture | 架构

```
Input 输入
  │
  ▼
┌────────────────────┐
│  MentalStateEncoder │  observation → initial BDI embedding
│  心智状态编码器      │  观测 → 初始 BDI 嵌入
└────────┬───────────┘
         │
   ┌─────┴─────┬─────────┬── … ──┐
   ▼           ▼         ▼       ▼
Column_0   Column_1  Column_2  Column_K    ← FractalMentalizingColumn
(depth 0)  (depth 1) (depth 2) (depth K)     分形心智化列（自相似 ψ 模块）
   │           │         │       │
   └─────┬─────┴─────────┴── … ──┘
         ▼
┌────────────────────────┐
│  EpistemicGate ×(K+1)  │  confidence = 1/(1+σ_k)
│  认知门控 ×(K+1)        │  置信度 = 1/(1+σ_k)
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ Attention-Weighted Join │  α_k(x) soft selection
│ 注意力加权联合           │  α_k(x) 软选择
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  BeliefRevisionModule  │  Bayesian gated update
│  信念修正模块           │  贝叶斯门控更新
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│     Task Head          │  Classification / ToM Prediction
│     任务头              │  分类 / 心智理论预测
└────────────────────────┘
```

### Module Inventory | 模块清单

| Module | File Location | Purpose |
|--------|--------------|---------|
| `FractalDropPath` | `nn.py` | Stochastic column dropping with developmental curriculum |
| `GatedResidual` | `nn.py` | GRU-style gated residual connection |
| `BDIState` | `nn.py` | Structured Belief-Desire-Intention container |
| `MentalStateEncoder` | `nn.py` | Input → BDI triple + epistemic uncertainty |
| `SelfSimilarBlock` | `nn.py` | Fractal primitive: self-attention + cross-attention + GEGLU + BDI |
| `PerspectiveShiftAttention` | `nn.py` | Orthogonal perspective transform + cross-depth attention |
| `FractalMentalizingColumn` | `nn.py` | K-th order mentalizing column (stacked ψ blocks) |
| `EpistemicGate` | `nn.py` | Confidence-scaled gating per column |
| `BeliefRevisionModule` | `nn.py` | Approximate Bayesian prior-evidence integration |
| `MentalizingJoin` | `nn.py` | Input-dependent soft selection over mentalizing depths |
| `FracToMNet` | `nn.py` | Complete architecture combining all sub-modules |
| `FracToMLoss` | `nn.py` | Composite loss (task + BDI + uncertainty + entropy) |
| `SequenceFracToM` | `nn.py` | Sequence-input wrapper with Transformer encoder |

| 模块 | 文件位置 | 用途 |
|------|---------|------|
| `FractalDropPath` | `nn.py` | 带发展性课程的随机列丢弃 |
| `GatedResidual` | `nn.py` | GRU 风格的门控残差连接 |
| `BDIState` | `nn.py` | 结构化的信念-欲望-意图容器 |
| `MentalStateEncoder` | `nn.py` | 输入 → BDI 三元组 + 认知不确定性 |
| `SelfSimilarBlock` | `nn.py` | 分形基元：自注意力 + 交叉注意力 + GEGLU + BDI |
| `PerspectiveShiftAttention` | `nn.py` | 正交视角变换 + 跨深度注意力 |
| `FractalMentalizingColumn` | `nn.py` | K 阶心智化列（堆叠的 ψ 模块） |
| `EpistemicGate` | `nn.py` | 每列的置信度缩放门控 |
| `BeliefRevisionModule` | `nn.py` | 近似贝叶斯先验-证据整合 |
| `MentalizingJoin` | `nn.py` | 基于输入的心智化深度软选择 |
| `FracToMNet` | `nn.py` | 结合所有子模块的完整架构 |
| `FracToMLoss` | `nn.py` | 复合损失（任务 + BDI + 不确定性 + 熵） |
| `SequenceFracToM` | `nn.py` | 带 Transformer 编码器的序列输入包装器 |

---

## Quick Start | 快速开始

### Minimal Example | 最小示例

```python
import torch
from nn import FracToMNet, FracToMLoss, analyse_mentalizing_depth

# 1) Build model | 构建模型
model = FracToMNet(
    input_dim=128,
    hidden_dim=256,
    mentalizing_depth=3,   # columns 0–3 | 0–3 列
    num_classes=10,
)

# 2) Forward pass with interpretability | 带可解释性的前向传播
x = torch.randn(32, 128)
logits, report = model(x, return_interpretability=True)

# 3) Inspect mentalizing depth allocation | 检查心智化深度分配
print(analyse_mentalizing_depth(report))

# 4) Train with composite loss | 使用复合损失训练
criterion = FracToMLoss()
targets = torch.randint(0, 10, (32,))
loss, breakdown = criterion(logits, targets, report)
loss.backward()
```

### Run the Built-in Demo | 运行内置演示

```bash
python nn.py
```

This trains a small FracToM on a synthetic Sally-Anne false-belief task and prints interpretability analysis.

这将在合成的 Sally-Anne 错误信念任务上训练一个小型 FracToM，并打印可解释性分析。

---

## Scenario Trainers | 场景训练脚本

This repository includes two ready-to-run scenario trainers built on top of `nn.py`.

本仓库包含两个基于 `nn.py` 的可直接运行场景训练脚本。

### 1) Collaboration Trainer: `collab_train.py`

Trains FracToM on a synthetic **dual-agent collaboration** dataset.

在合成的**双智能体协作**数据集上训练 FracToM。

#### What it does

- Generates random collaborative episodes with private goals and noisy beliefs
- Trains and evaluates FracToM (accuracy, confusion matrix, loss breakdown)
- Exports ToM hierarchy visualizations automatically

#### Run

```bash
python collab_train.py
```

Quick smoke test:

```bash
python collab_train.py --epochs 2 --samples 1000 --batch-size 64 --eval-every 1
```

#### Visualization outputs

By default, figures are written to `visualizations/`:

- `tom_depth_weights.png`
- `tom_uncertainty.png`

Configure with:

```bash
python collab_train.py --viz-dir collab_viz --viz-heatmap-samples 80
```

### 2) Competition Trainer: `compete_train.py`

Trains FracToM on a synthetic **dual-agent competition** dataset and runs a
post-training competition simulator against random and majority baselines.

在合成的**双智能体竞争**数据集上训练 FracToM，并在训练后与随机/多数类基线进行竞争仿真对比。

#### What it does

- Generates random competitive episodes (objectives, capability gaps, risk/scarcity)
- Trains and evaluates FracToM with interpretability summaries
- Runs competition simulation and reports win rate + utility
- Exports ToM hierarchy visualizations automatically

#### Run

```bash
python compete_train.py
```

Quick smoke test:

```bash
python compete_train.py --epochs 3 --samples 1500 --batch-size 64 --eval-every 1 --sim-episodes 300
```

#### Visualization outputs

By default, figures are written to `visualizations_compete/`:

- `tom_depth_weights.png`
- `tom_uncertainty.png`

Configure with:

```bash
python compete_train.py --viz-dir compete_viz --viz-heatmap-samples 80
```

### Script Argument Summary

| Script | Key Args | Purpose |
|---|---|---|
| `collab_train.py` | `--samples --epochs --batch-size --viz-dir --viz-heatmap-samples` | Collaboration training + ToM visualization |
| `compete_train.py` | `--samples --epochs --batch-size --sim-episodes --viz-dir --viz-heatmap-samples` | Competition training + simulator + ToM visualization |

---

## Adapting to New Datasets | 适配新数据集

### Step 1: Determine Input Modality | 步骤一：确定输入模态

FracToM provides two entry points depending on your data:

FracToM 根据数据类型提供两种入口：

| Data Type | Entry Point | Example |
|-----------|-------------|---------|
| Fixed-dimensional vectors (images, tabular, game states) | `FracToMNet` | Encoded observations, feature vectors |
| Variable-length token sequences (text, dialogue) | `SequenceFracToM` | Stories, conversations, instructions |

| 数据类型 | 入口 | 示例 |
|---------|------|------|
| 固定维度向量（图像、表格、游戏状态） | `FracToMNet` | 编码观测、特征向量 |
| 变长标记序列（文本、对话） | `SequenceFracToM` | 故事、对话、指令 |

### Step 2: Configure Dimensions | 步骤二：配置维度

```python
# For vector inputs | 向量输入
model = FracToMNet(
    input_dim=YOUR_FEATURE_DIM,    # must match your data | 必须匹配数据
    hidden_dim=256,                 # ↑ for more capacity | ↑ 增加容量
    mentalizing_depth=3,            # ↑ for deeper ToM | ↑ 更深的心智理论
    blocks_per_column=2,            # ↑ for more processing per level | ↑ 每层更多处理
    num_heads=8,                    # must divide hidden_dim | 必须整除 hidden_dim
    num_classes=YOUR_NUM_CLASSES,   # omit for regression | 回归时省略
)

# For sequence inputs | 序列输入
model = SequenceFracToM(
    vocab_size=YOUR_VOCAB_SIZE,
    embed_dim=256,
    max_seq_len=512,
    seq_encoder_layers=4,
    mentalizing_depth=3,
    num_classes=YOUR_NUM_CLASSES,
)
```

### Step 3: Write a Data Loader | 步骤三：编写数据加载器

```python
from torch.utils.data import Dataset, DataLoader

class MyToMDataset(Dataset):
    """
    Template for a ToM dataset.
    ToM 数据集模板。
    """
    def __init__(self, data_path: str):
        # Load your data here | 在此加载数据
        self.samples = ...
        self.labels = ...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]          # (input_dim,) tensor
        y = self.labels[idx]           # scalar label
        return x, y

train_loader = DataLoader(MyToMDataset("train.json"), batch_size=64, shuffle=True)
```

### Step 4: Training Loop Template | 步骤四：训练循环模板

```python
import torch
from nn import FracToMNet, FracToMLoss

model = FracToMNet(input_dim=128, hidden_dim=256, mentalizing_depth=3, num_classes=5)
criterion = FracToMLoss(lambda_bdi=0.01, lambda_uncertainty=0.005, lambda_depth_entropy=0.01)
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    model.train()
    # Developmental curriculum: anneal from 1.0 → 0.0
    # 发展性课程：从 1.0 退火到 0.0
    model.set_curriculum(1.0 - epoch / EPOCHS)

    for xb, yb in train_loader:
        logits, report = model(xb, return_interpretability=True)
        loss, breakdown = criterion(logits, yb, report)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    # Log breakdown for monitoring | 记录分解用于监控
    # breakdown = {"task": ..., "bdi_consistency": ..., "uncertainty_cal": ..., ...}
```

### Step 5: Adapt for Specific Task Types | 步骤五：适配特定任务类型

#### Classification (分类)

Set `num_classes` in the constructor. Use `nn.CrossEntropyLoss` (default).

在构造函数中设置 `num_classes`。使用 `nn.CrossEntropyLoss`（默认）。

#### Regression (回归)

Omit `num_classes` (output is `hidden_dim` vector). Add your own regression head:

省略 `num_classes`（输出为 `hidden_dim` 向量）。添加自定义回归头：

```python
model = FracToMNet(input_dim=128, hidden_dim=256, mentalizing_depth=3)
regression_head = torch.nn.Linear(256, 1)  # single scalar output | 单标量输出

# In forward | 前向传播中
h = model(x)                    # (batch, 256)
prediction = regression_head(h)  # (batch, 1)
```

#### ToM Prediction (心智理论预测)

Use `ToMPredictionHead` to predict another agent's BDI state:

使用 `ToMPredictionHead` 预测另一个智能体的 BDI 状态：

```python
from nn import FracToMNet, ToMPredictionHead

model = FracToMNet(input_dim=128, hidden_dim=256, mentalizing_depth=3)
tom_head = model.get_tom_head()

h = model(x)                    # (batch, 256)
predicted_bdi = tom_head(h)     # BDIState with .belief, .desire, .intention
```

#### Multi-Agent Scenarios (多智能体场景)

For games or social scenarios with multiple agents:

对于多智能体的游戏或社交场景：

```python
# Encode each agent's observation separately | 分别编码每个智能体的观测
obs_self = encode_observation(agent_self)     # (batch, input_dim)
obs_other = encode_observation(agent_other)   # (batch, input_dim)

# Concatenate or use separate models | 拼接或使用独立模型
combined = torch.cat([obs_self, obs_other], dim=-1)
model = FracToMNet(input_dim=input_dim * 2, ...)
```

### Recommended Hyperparameters by Dataset Scale | 按数据集规模推荐的超参数

| Dataset Size | `hidden_dim` | `mentalizing_depth` | `blocks_per_column` | `num_heads` |
|-------------|-------------|--------------------|--------------------|------------|
| < 1K samples | 64–96 | 1–2 | 1 | 4 |
| 1K–10K | 128–256 | 2–3 | 1–2 | 4–8 |
| 10K–100K | 256–512 | 3–4 | 2–3 | 8 |
| > 100K | 512–1024 | 3–5 | 2–4 | 8–16 |

| 数据集规模 | `hidden_dim` | `mentalizing_depth` | `blocks_per_column` | `num_heads` |
|-----------|-------------|--------------------|--------------------|------------|
| < 1K 样本 | 64–96 | 1–2 | 1 | 4 |
| 1K–10K | 128–256 | 2–3 | 1–2 | 4–8 |
| 10K–100K | 256–512 | 3–4 | 2–3 | 8 |
| > 100K | 512–1024 | 3–5 | 2–4 | 8–16 |

---

## Expanding the Architecture | 扩展架构

### 1. Adding New Mental-State Factors | 添加新的心理状态因子

The current BDI triple can be extended to richer ontologies. For example, adding **Emotions** (E) for a BDIE model:

当前的 BDI 三元组可以扩展为更丰富的本体论。例如，添加**情感**（E）形成 BDIE 模型：

```python
# In MentalStateEncoder.__init__:
# 在 MentalStateEncoder.__init__ 中：
self.proj_emotion = nn.Linear(hidden, factor_dim)

# In forward:
# 在 forward 中：
bdi = BDIEState(
    belief=self.proj_belief(h),
    desire=self.proj_desire(h),
    intention=self.proj_intention(h),
    emotion=self.proj_emotion(h),
)
```

Update `num_bdi_factors` to 4. Ensure `hidden_dim % 4 == 0`.

将 `num_bdi_factors` 更新为 4。确保 `hidden_dim % 4 == 0`。

### 2. Adding New Column Types | 添加新的列类型

Columns can be specialised beyond pure mentalizing depth. Ideas:

列可以超越纯粹的心智化深度进行专门化。思路：

- **Affective column** — specialised for emotional processing | **情感列** —— 专门处理情感
- **Temporal column** — maintains state across timesteps | **时间列** —— 跨时间步维护状态
- **Counterfactual column** — reasons about hypothetical scenarios | **反事实列** —— 推理假设场景

```python
class AffectiveColumn(FractalMentalizingColumn):
    """Column specialised for emotional processing.
    专门处理情感的列。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valence_head = nn.Linear(self.blocks[0].dim, 2)  # positive/negative
        self.arousal_head = nn.Linear(self.blocks[0].dim, 1)  # arousal level

    def forward(self, x, lower_level_outputs=None):
        h, bdis = super().forward(x, lower_level_outputs)
        valence = torch.softmax(self.valence_head(h), dim=-1)
        arousal = torch.sigmoid(self.arousal_head(h))
        return h, bdis, valence, arousal
```

### 3. Visual Encoder Frontend | 视觉编码器前端

For image-based tasks (e.g., your maze environment):

对于基于图像的任务（如迷宫环境）：

```python
import torchvision.models as models

class VisualFracToM(nn.Module):
    """FracToM with a CNN visual encoder frontend.
    带 CNN 视觉编码器前端的 FracToM。
    """
    def __init__(self, num_classes: int, mentalizing_depth: int = 3):
        super().__init__()
        # Use a pretrained backbone | 使用预训练骨干网络
        resnet = models.resnet18(pretrained=True)
        self.visual_enc = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 512  # ResNet-18 output | ResNet-18 输出

        self.fractom = FracToMNet(
            input_dim=feature_dim,
            hidden_dim=256,
            mentalizing_depth=mentalizing_depth,
            num_classes=num_classes,
        )

    def forward(self, images, return_interpretability=False):
        features = self.visual_enc(images).flatten(1)  # (B, 512)
        return self.fractom(features, return_interpretability)
```

### 4. Recurrent / Temporal Extension | 循环 / 时间扩展

For tasks requiring memory over time (e.g., multi-turn dialogue, sequential decision-making):

对于需要时间记忆的任务（如多轮对话、序列决策）：

```python
class TemporalFracToM(nn.Module):
    """Maintains a belief state across timesteps.
    跨时间步维护信念状态。
    """
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.fractom = FracToMNet(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
        self.belief_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq, return_interpretability=False):
        """
        x_seq: (batch, timesteps, input_dim)
        """
        B, T, _ = x_seq.shape
        h_belief = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        reports = []

        for t in range(T):
            out = self.fractom(x_seq[:, t], return_interpretability=True)
            h_t, report = out
            h_belief = self.belief_gru(h_t, h_belief)
            reports.append(report)

        return h_belief, reports
```

### 5. Custom Loss Components | 自定义损失组件

The `FracToMLoss` can be extended with domain-specific regularisers:

`FracToMLoss` 可以通过特定领域的正则化器进行扩展：

```python
class ExtendedFracToMLoss(FracToMLoss):
    """Adds a ToM-specific auxiliary loss.
    添加特定于心智理论的辅助损失。
    """
    def __init__(self, *args, lambda_tom_aux: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_tom_aux = lambda_tom_aux

    def forward(self, logits, targets, report, tom_targets=None):
        loss, breakdown = super().forward(logits, targets, report)

        if tom_targets is not None:
            # Auxiliary loss: predict another agent's belief
            # 辅助损失：预测另一个智能体的信念
            bdi_col2 = report.bdi_states[2][-1]  # column 2 (other-model)
            tom_loss = F.mse_loss(bdi_col2.belief, tom_targets)
            loss = loss + self.lambda_tom_aux * tom_loss
            breakdown["tom_aux"] = tom_loss.item()

        return loss, breakdown
```

---

## Advancing the Research | 推进研究

### Research Direction 1: Shared-Weight Columns | 研究方向一：共享权重列

**Idea**: Make the fractal self-similarity *strict* — all columns share the same weight matrices, differentiated only by a learned depth embedding.

**思路**：使分形自相似性*严格化* —— 所有列共享相同的权重矩阵，仅通过学习到的深度嵌入进行区分。

```python
class StrictFractalColumn(nn.Module):
    """All mentalizing levels share a single block's weights.
    所有心智化层级共享单个模块的权重。
    """
    def __init__(self, dim, factor_dim, max_depth, **kwargs):
        super().__init__()
        # Single shared block | 单个共享模块
        self.shared_block = SelfSimilarBlock(dim, factor_dim, **kwargs)
        # Depth embedding conditions the block | 深度嵌入调节模块
        self.depth_embed = nn.Embedding(max_depth + 1, dim)

    def forward(self, x, depth_index):
        # Condition input on depth | 根据深度调节输入
        x = x + self.depth_embed(torch.tensor(depth_index, device=x.device))
        return self.shared_block(x)
```

**Benefit**: Dramatically fewer parameters; stronger inductive bias for self-similarity.

**优势**：大幅减少参数量；更强的自相似性归纳偏置。

### Research Direction 2: Dynamic Depth Selection | 研究方向二：动态深度选择

**Idea**: Instead of always running all K+1 columns, use a learned halting mechanism (like ACT — Adaptive Computation Time) to dynamically decide how deep to mentalize for each input.

**思路**：不总是运行所有 K+1 列，而是使用学习到的停止机制（如 ACT —— 自适应计算时间）动态决定每个输入的心智化深度。

```python
class AdaptiveDepthFracToM(FracToMNet):
    """Adaptively decides mentalizing depth per sample.
    针对每个样本自适应决定心智化深度。
    """
    def __init__(self, *args, halt_threshold=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.halt_threshold = halt_threshold
        self.halt_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, return_interpretability=False):
        B = x.shape[0]
        bdi_init, _ = self.encoder(x)
        h_input = self.input_proj(bdi_init.pack())

        # Accumulate halting probability | 累积停止概率
        cum_halt = torch.zeros(B, 1, device=x.device)
        running = torch.ones(B, 1, device=x.device)
        weighted_sum = torch.zeros(B, self.hidden_dim, device=x.device)

        for k, (col, egate) in enumerate(zip(self.columns, self.epistemic_gates)):
            lower = [weighted_sum] if k > 0 else None
            h_col, _ = col(h_input, lower)
            h_col, _ = egate(h_col)

            halt_p = torch.sigmoid(self.halt_proj(h_col))
            still_running = (cum_halt < self.halt_threshold).float()
            running = running * still_running

            weighted_sum = weighted_sum + running * halt_p * h_col
            cum_halt = cum_halt + running * halt_p

        # ... continue with belief revision and task head
        # ... 继续信念修正和任务头
```

### Research Direction 3: Multi-Agent Mentalizing | 研究方向三：多智能体心智化

**Idea**: Extend a single agent's self-similar columns to model *multiple other agents simultaneously*, each with their own perspective column.

**思路**：将单个智能体的自相似列扩展为*同时建模多个其他智能体*，每个智能体有自己的视角列。

```python
class MultiAgentFracToM(nn.Module):
    """Maintains separate mentalizing columns per observed agent.
    为每个被观察的智能体维护单独的心智化列。
    """
    def __init__(self, input_dim, hidden_dim, num_agents, mentalizing_depth=2, **kwargs):
        super().__init__()
        # Self-model (shared) | 自我模型（共享）
        self.self_model = FracToMNet(input_dim, hidden_dim, mentalizing_depth=0, **kwargs)
        # One FracToM per other agent | 每个其他智能体一个 FracToM
        self.agent_models = nn.ModuleList([
            FracToMNet(input_dim, hidden_dim, mentalizing_depth, **kwargs)
            for _ in range(num_agents)
        ])
        self.fusion = nn.Linear(hidden_dim * (1 + num_agents), hidden_dim)

    def forward(self, self_obs, other_obs_list):
        h_self = self.self_model(self_obs)
        h_others = [m(obs) for m, obs in zip(self.agent_models, other_obs_list)]
        combined = torch.cat([h_self] + h_others, dim=-1)
        return self.fusion(combined)
```

### Research Direction 4: Neuroscience-Aligned Probing | 研究方向四：神经科学对齐的探测

**Idea**: Use the BDI-factored representations for direct comparison with fMRI data from ToM tasks (TPJ, mPFC activations).

**思路**：使用 BDI 分解表示与来自心智理论任务的 fMRI 数据（TPJ、mPFC 激活）进行直接比较。

```python
from nn import extract_bdi_activations

# After training, extract BDI representations per depth
# 训练后，提取每个深度的 BDI 表示
_, report = model(test_data, return_interpretability=True)
bdi_acts = extract_bdi_activations(report)

# Compare column 2 beliefs with TPJ fMRI activation patterns
# 将列 2 的信念与 TPJ fMRI 激活模式进行比较
col2_beliefs = bdi_acts[2]["belief"]  # (N_samples, factor_dim)
# → Compute RSA (Representational Similarity Analysis) with fMRI
# → 与 fMRI 计算 RSA（表征相似性分析）
```

### Research Direction 5: Combining with LLMs | 研究方向五：与大语言模型结合

**Idea**: Use FracToM as a structured "mentalizing head" on top of a frozen LLM encoder, giving the LLM explicit ToM structure.

**思路**：在冻结的 LLM 编码器之上使用 FracToM 作为结构化的"心智化头"，赋予 LLM 显式的心智理论结构。

```python
from transformers import AutoModel

class LLMFracToM(nn.Module):
    def __init__(self, llm_name="bert-base-uncased", mentalizing_depth=3, num_classes=2):
        super().__init__()
        self.llm = AutoModel.from_pretrained(llm_name)
        for p in self.llm.parameters():
            p.requires_grad = False  # freeze LLM | 冻结 LLM

        llm_dim = self.llm.config.hidden_size
        self.fractom = FracToMNet(
            input_dim=llm_dim,
            hidden_dim=256,
            mentalizing_depth=mentalizing_depth,
            num_classes=num_classes,
        )

    def forward(self, input_ids, attention_mask, return_interpretability=False):
        with torch.no_grad():
            llm_out = self.llm(input_ids, attention_mask=attention_mask)
        pooled = llm_out.last_hidden_state[:, 0]  # [CLS] token
        return self.fractom(pooled, return_interpretability)
```

### Research Direction 6: Formal Verification of Mentalizing | 研究方向六：心智化的形式化验证

**Idea**: Because BDI states are explicit, formal properties can be checked:

**思路**：由于 BDI 状态是显式的，可以检查形式化属性：

- **Belief consistency**: Are beliefs at depth k logically consistent with observations? | **信念一致性**：深度 k 的信念是否与观测逻辑一致？
- **Intention-desire alignment**: Do intentions serve desires? | **意图-欲望对齐**：意图是否服务于欲望？
- **Depth monotonicity**: Does adding mentalizing depth always help or plateau? | **深度单调性**：增加心智化深度是否总是有帮助或趋于饱和？

```python
def check_belief_consistency(report, observations):
    """Verify that deeper beliefs are consistent with shallower ones.
    验证更深层的信念与更浅层的信念一致。
    """
    bdi = extract_bdi_activations(report)
    for k in range(1, len(bdi)):
        cos_sim = F.cosine_similarity(
            bdi[k]["belief"], bdi[k-1]["belief"], dim=-1
        )
        print(f"  Belief consistency (depth {k-1}→{k}): {cos_sim.mean():.4f}")
```

---

## API Reference | API 参考

### `FracToMNet`

```python
FracToMNet(
    input_dim: int,              # Raw observation dimensionality | 原始观测维度
    hidden_dim: int = 256,       # Internal representation width | 内部表示宽度
    mentalizing_depth: int = 3,  # Max ToM order (K); creates K+1 columns | 最大 ToM 阶数
    num_bdi_factors: int = 3,    # BDI factor count (currently 3) | BDI 因子数量
    blocks_per_column: int = 2,  # SelfSimilarBlocks per column | 每列的自相似模块数
    num_heads: int = 8,          # Attention heads | 注意力头数
    ff_mult: int = 4,            # Feed-forward multiplier | 前馈乘数
    dropout: float = 0.1,        # Dropout rate | Dropout 率
    drop_path: float = 0.15,     # Base column drop probability | 基础列丢弃概率
    num_classes: int | None = None,  # Set for classification | 分类时设置
)
```

**Methods | 方法:**

| Method | Description | 方法 | 说明 |
|--------|-------------|------|------|
| `forward(x, return_interpretability=False)` | Main forward pass | `forward(x, return_interpretability=False)` | 主前向传播 |
| `set_curriculum(factor)` | Set developmental drop-path factor (1→0) | `set_curriculum(factor)` | 设置发展性丢弃路径因子 (1→0) |
| `get_tom_head(factor_dim, dropout)` | Build a matching ToMPredictionHead | `get_tom_head(factor_dim, dropout)` | 构建匹配的 ToM 预测头 |

### `FracToMLoss`

```python
FracToMLoss(
    task_loss_fn: nn.Module = None,          # Default: CrossEntropyLoss | 默认：交叉熵损失
    lambda_bdi: float = 0.01,                # BDI consistency weight | BDI 一致性权重
    lambda_uncertainty: float = 0.005,       # Uncertainty calibration weight | 不确定性校准权重
    lambda_depth_entropy: float = 0.01,      # Depth entropy regularisation | 深度熵正则化
)
```

**Returns | 返回:** `(loss: Tensor, breakdown: Dict[str, float])`

The `breakdown` dict contains: `task`, `bdi_consistency`, `uncertainty_cal`, `depth_entropy_reg`, `total`.

`breakdown` 字典包含：`task`、`bdi_consistency`、`uncertainty_cal`、`depth_entropy_reg`、`total`。

### `InterpretabilityReport`

| Attribute | Shape | Description | 属性 | 形状 | 说明 |
|-----------|-------|-------------|------|------|------|
| `depth_weights` | `(B, K+1)` | α_k per sample — which mentalizing depth was used | `depth_weights` | `(B, K+1)` | 每个样本的 α_k — 使用了哪个心智化深度 |
| `bdi_states` | `dict[int, list[BDIState]]` | Per-column, per-block BDI triples | `bdi_states` | `dict[int, list[BDIState]]` | 每列、每模块的 BDI 三元组 |
| `column_uncertainties` | `dict[int, Tensor(B,1)]` | Epistemic uncertainty σ_k per column | `column_uncertainties` | `dict[int, Tensor(B,1)]` | 每列的认知不确定性 σ_k |

### `SequenceFracToM`

```python
SequenceFracToM(
    vocab_size: int,
    embed_dim: int = 256,
    max_seq_len: int = 512,
    seq_encoder_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    **fractom_kwargs,            # Forwarded to FracToMNet | 转发到 FracToMNet
)
```

### Utility Functions | 工具函数

```python
# Human-readable mentalizing depth analysis | 人类可读的心智化深度分析
analyse_mentalizing_depth(report: InterpretabilityReport) -> str

# Extract BDI activations for downstream analysis | 提取 BDI 激活用于下游分析
extract_bdi_activations(report: InterpretabilityReport) -> Dict[int, Dict[str, Tensor]]
```

---

## Training Guide | 训练指南

### Developmental Curriculum Schedule | 发展性课程计划

The curriculum controls how aggressively deeper columns are dropped during training. This mimics how children acquire ToM:

课程控制在训练过程中更深层列被丢弃的激进程度。这模拟了儿童获得心智理论的过程：

| Training Phase | Curriculum Factor | Effect | 训练阶段 | 课程因子 | 效果 |
|---------------|------------------|--------|---------|---------|------|
| 0–25% | 1.0 → 0.75 | Deep columns mostly dropped; learn basic perception | 0–25% | 1.0 → 0.75 | 深层列大多被丢弃；学习基础感知 |
| 25–50% | 0.75 → 0.50 | Column 1 (metacognition) emerges | 25–50% | 0.75 → 0.50 | 列 1（元认知）出现 |
| 50–75% | 0.50 → 0.25 | Column 2 (other-model) stabilises | 50–75% | 0.50 → 0.25 | 列 2（他人模型）稳定 |
| 75–100% | 0.25 → 0.0 | All columns contribute; fine-tuning | 75–100% | 0.25 → 0.0 | 所有列贡献；微调 |

```python
for epoch in range(total_epochs):
    factor = 1.0 - epoch / total_epochs
    model.set_curriculum(factor)
```

### Monitoring Training | 监控训练

Watch these signals from the loss breakdown:

从损失分解中关注这些信号：

- **`task`** — should decrease steadily | 应稳步下降
- **`bdi_consistency`** — should be low but not zero (some divergence is healthy) | 应较低但不为零（一些分歧是健康的）
- **`depth_entropy_reg`** — should stay moderate (avoid collapse to one depth) | 应保持适中（避免坍缩到单一深度）
- **`uncertainty_cal`** — should decrease (model becomes better calibrated) | 应下降（模型变得更好校准）

### Tips | 技巧

1. **Learning rate**: 1e-4 to 5e-4 with AdamW works well. | **学习率**：使用 AdamW 时 1e-4 到 5e-4 效果良好。
2. **Gradient clipping**: Always clip to 1.0; the multi-column architecture can have gradient variance. | **梯度裁剪**：始终裁剪到 1.0；多列架构可能有梯度方差。
3. **Batch size**: ≥32 recommended for stable epistemic uncertainty estimates. | **批大小**：建议 ≥32 以获得稳定的认知不确定性估计。
4. **Weight decay**: 0.01–0.05; BDI heads benefit from mild regularisation. | **权重衰减**：0.01–0.05；BDI 头受益于温和的正则化。
5. **`mentalizing_depth`**: Start with 2–3. More than 4 rarely helps and increases computation. | **`mentalizing_depth`**：从 2–3 开始。超过 4 很少有帮助且增加计算量。

---

## Interpretability Toolkit | 可解释性工具

### Mentalizing Depth Analysis | 心智化深度分析

```python
model.eval()
with torch.no_grad():
    _, report = model(X_test, return_interpretability=True)

# Print formatted analysis | 打印格式化分析
print(analyse_mentalizing_depth(report))
```

**Example output | 示例输出:**
```
Mentalizing Depth Analysis (400 samples, 4 levels)
============================================================
  Level 0 (Direct (no ToM)     ): 0.257  ██████████
  Level 1 (Metacognition       ): 0.250  ██████████
  Level 2 (Basic other-model   ): 0.249  █████████
  Level 3 (2nd-order ToM       ): 0.243  █████████

Dominant mentalizing level per sample:
  Level 0: 120/400 samples (30.0%)
  Level 1: 89/400 samples (22.2%)
  Level 2: 98/400 samples (24.5%)
  Level 3: 93/400 samples (23.2%)

Epistemic uncertainty (σ) per column:
  Column 0: σ = 0.0346
  Column 1: σ = 0.0343
  Column 2: σ = 0.0161
  Column 3: σ = 0.0020
```

### BDI Activation Probing | BDI 激活探测

```python
bdi_acts = extract_bdi_activations(report)

for k, acts in bdi_acts.items():
    print(f"Column {k}:")
    print(f"  Belief  mean norm: {acts['belief'].norm(dim=-1).mean():.3f}")
    print(f"  Desire  mean norm: {acts['desire'].norm(dim=-1).mean():.3f}")
    print(f"  Intention mean norm: {acts['intention'].norm(dim=-1).mean():.3f}")
```

### Visualising Depth Weights | 可视化深度权重

```python
import matplotlib.pyplot as plt

alpha = report.depth_weights.numpy()  # (B, K+1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution of dominant depth | 主导深度分布
axes[0].hist(alpha.argmax(axis=1), bins=range(alpha.shape[1]+1), align='left')
axes[0].set_xlabel("Dominant mentalizing depth | 主导心智化深度")
axes[0].set_ylabel("Count | 数量")
axes[0].set_title("Which depth does the model rely on? | 模型依赖哪个深度？")

# Per-sample depth profile | 每个样本的深度分布
axes[1].imshow(alpha[:50], aspect='auto', cmap='viridis')
axes[1].set_xlabel("Mentalizing depth | 心智化深度")
axes[1].set_ylabel("Sample | 样本")
axes[1].set_title("Depth weights per sample | 每样本深度权重")

plt.tight_layout()
plt.savefig("depth_analysis.png", dpi=150)
```

### Comparing Conditions | 比较条件

```python
# Split test data by condition | 按条件分割测试数据
false_belief_mask = ...  # samples requiring ToM | 需要心智理论的样本
true_belief_mask = ...   # samples NOT requiring ToM | 不需要心智理论的样本

_, report_fb = model(X_test[false_belief_mask], return_interpretability=True)
_, report_tb = model(X_test[true_belief_mask], return_interpretability=True)

print("False-belief trials (need ToM) | 错误信念试验（需要心智理论）:")
print(f"  Mean depth weight on column 2: {report_fb.depth_weights[:, 2].mean():.3f}")
print("True-belief trials (no ToM needed) | 真实信念试验（不需要心智理论）:")
print(f"  Mean depth weight on column 2: {report_tb.depth_weights[:, 2].mean():.3f}")
# Expect: higher weight on deeper columns for false-belief trials
# 预期：错误信念试验在更深列上有更高权重
```

---

## FAQ | 常见问题

### Q: Why fractal architecture instead of a simple deep network? | 问：为什么用分形架构而不是简单的深度网络？

**A (EN):** Fractal self-similarity provides an inductive bias that mirrors the recursive nature of Theory of Mind. In cognitive science, mentalizing is inherently self-similar: "I think that you think that she thinks…" has the same *structure* at every depth, just applied recursively. A fractal architecture encodes this prior directly, leading to better sample efficiency on ToM tasks and interpretable depth allocation.

**A (中):** 分形自相似性提供了一种归纳偏置，反映了心智理论的递归本质。在认知科学中，心智化本质上是自相似的："我认为你认为她认为……"在每个深度都有相同的*结构*，只是递归应用。分形架构直接编码了这一先验，在心智理论任务上带来更好的样本效率和可解释的深度分配。

### Q: How does this compare to existing ToM models? | 问：这与现有的心智理论模型相比如何？

**A (EN):** Most prior ToM neural architectures (e.g., ToMNet by Rabinowitz et al., 2018) treat ToM as a black-box prediction problem. FracToM differs by: (1) making mentalizing depth an *architectural* feature, not just a training objective; (2) providing BDI-factored intermediate states for probing; (3) using epistemic gating to gracefully degrade unreliable higher-order attributions.

**A (中):** 大多数先前的 ToM 神经架构（如 Rabinowitz 等人 2018 年的 ToMNet）将心智理论视为黑盒预测问题。FracToM 的不同之处在于：(1) 将心智化深度作为*架构*特征而非仅仅是训练目标；(2) 提供 BDI 分解的中间状态用于探测；(3) 使用认知门控优雅地降低不可靠的高阶归因。

### Q: Can I use this for non-ToM tasks? | 问：我可以将其用于非心智理论任务吗？

**A (EN):** Yes. With `mentalizing_depth=0`, FracToM reduces to a standard gated Transformer encoder with BDI-factored latents. The fractal columns add capacity without changing the output interface, so the same model can be used for general classification/regression tasks — you simply get additional interpretability for free.

**A (中):** 可以。当 `mentalizing_depth=0` 时，FracToM 退化为具有 BDI 分解潜变量的标准门控 Transformer 编码器。分形列在不改变输出接口的情况下增加容量，因此相同的模型可用于一般的分类/回归任务——你只是额外免费获得了可解释性。

### Q: How do I choose `mentalizing_depth`? | 问：如何选择 `mentalizing_depth`？

**A (EN):** As a rule of thumb: use depth 0–1 for tasks without social reasoning, depth 2–3 for tasks involving modelling one other agent's perspective, and depth 3–4 for tasks requiring recursive mentalizing (e.g., strategic games, negotiation). Beyond depth 4, cognitive science suggests diminishing returns — even humans struggle with 4th-order ToM.

**A (中):** 经验法则：对于不涉及社会推理的任务使用深度 0-1，对于涉及建模另一个智能体视角的任务使用深度 2-3，对于需要递归心智化的任务（如策略游戏、谈判）使用深度 3-4。超过深度 4，认知科学表明收益递减——即使人类也难以处理四阶心智理论。

### Q: What if my task has no clear BDI decomposition? | 问：如果我的任务没有明确的 BDI 分解怎么办？

**A (EN):** The BDI heads are self-supervised through the composite loss (BDI consistency, uncertainty calibration). They will learn *some* factored decomposition even without explicit BDI labels. You can treat them as a structured bottleneck that encourages disentangled representations — beneficial even when the factors don't map perfectly to philosophical BDI.

**A (中):** BDI 头通过复合损失（BDI 一致性、不确定性校准）进行自监督。即使没有显式的 BDI 标签，它们也会学习*某种*分解表示。你可以将它们视为鼓励解耦表示的结构化瓶颈——即使因子不完美映射到哲学 BDI 也是有益的。

### Q: How do I integrate FracToM with the existing maze agent? | 问：如何将 FracToM 与现有的迷宫智能体集成？

**A (EN):** The maze agent in `agent.py` uses an LLM via API calls. To use FracToM as a learned component, you would encode maze states as vectors (grid flattening or CNN), feed them through `FracToMNet`, and use the output to predict actions. The `InterpretabilityReport` can then explain *why* the agent chose a particular direction.

**A (中):** `agent.py` 中的迷宫智能体通过 API 调用使用 LLM。要将 FracToM 用作学习组件，你需要将迷宫状态编码为向量（网格展平或 CNN），通过 `FracToMNet` 传递它们，并使用输出预测动作。`InterpretabilityReport` 然后可以解释智能体*为什么*选择了特定方向。

```python
# Example: Maze state → FracToM → Action
# 示例：迷宫状态 → FracToM → 动作

import torch
from nn import FracToMNet

GRID_SIZE = 16
NUM_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT

# Flatten maze grid as input | 将迷宫网格展平作为输入
model = FracToMNet(
    input_dim=GRID_SIZE * GRID_SIZE,  # flattened grid | 展平的网格
    hidden_dim=128,
    mentalizing_depth=2,
    num_classes=NUM_ACTIONS,
)

# Encode maze state | 编码迷宫状态
grid_tensor = torch.zeros(1, GRID_SIZE * GRID_SIZE)
# ... fill with cell types ...

logits, report = model(grid_tensor, return_interpretability=True)
action_idx = logits.argmax(-1).item()
actions = ["UP", "DOWN", "LEFT", "RIGHT"]
print(f"Predicted action | 预测动作: {actions[action_idx]}")
```

---

## Citation | 引用

If you use FracToM in your research, please cite:

如果你在研究中使用 FracToM，请引用：

```bibtex
@software{fractom2026,
  title   = {FracToM: Fractal Theory-of-Mind Neural Network},
  author  = {Yan, T.},
  year    = {2026},
  url     = {https://github.com/henryyantq/FracToM},
  note    = {A fractal neural architecture for interpretable hierarchical mentalizing}
}
```

---

## License | 许可证

This project is released for academic research purposes.

本项目用于学术研究目的发布。
