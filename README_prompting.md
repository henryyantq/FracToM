# FractalToM Prompting Framework

A **pure-prompting** implementation of Fractal Theory of Mind (FToM) for LLM agents — enabling recursive nested belief modelling and **dual-agent deception** without any neural-network training or fine-tuning.

---

## Table of Contents

- [Overview](#overview)
- [Theoretical Foundation](#theoretical-foundation)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Built-in Demo](#built-in-demo)
  - [Python API](#python-api)
  - [Custom Scenario from JSON](#custom-scenario-from-json)
  - [Hausdorff Dimension Calculator](#hausdorff-dimension-calculator)
- [Configuration](#configuration)
- [Core Components](#core-components)
  - [BeliefState](#beliefstate)
  - [AgentProfile](#agentprofile)
  - [MindMapping](#mindmapping)
  - [NestedToMComputer](#nestedtomcomputer)
  - [HutchinsonOperator](#hutchinsonoperator)
  - [DeceptionPlanner](#deceptionplanner)
  - [FractalToMAgent](#fractalomagent)
  - [DualAgentArena](#dualagenttarena)
- [Scenario JSON Format](#scenario-json-format)
- [Example Scenarios](#example-scenarios)
- [Mathematical Details](#mathematical-details)
- [Output Schema](#output-schema)
- [License](#license)

---

## Overview

The recursive structure of Theory of Mind — *"I think you think I think …"* — is isomorphic to the self-similarity of fractals. **FractalToM Prompting** operationalises that insight by mapping the full mathematical apparatus of Iterated Function Systems (IFS) onto structured LLM calls, enabling:

- **Recursive nested belief modelling** up to arbitrary depth.
- **Strategic deception planning** — an agent can craft utterances that exploit gaps in the other's knowledge.
- **Deception detection** — an agent reasons about whether the other is lying, and why.
- **Fractal complexity measurement** — quantitative Hausdorff dimension estimates of the evolving belief structure.

All of this is achieved purely through **structured prompting** of an OpenAI reasoning model (default: `gpt-5-mini`). No weights are trained; no fine-tuning is required.

---

## Theoretical Foundation

| Mathematical Object | Prompting Realisation |
|---|---|
| Belief space $\mathbf{B}$ | Structured `BeliefState` object |
| Contraction mapping $f_i$ | `MindMapping`: a perspective-taking prompt for agent $i$ |
| Composition $f_i \circ f_j$ | Recursive prompt chaining (nested ToM) |
| Hutchinson operator $F(B)$ | `HutchinsonOperator`: aggregate all agents' mappings & iterate |
| Fractal attractor $A^{\ast}$ | Converged equilibrium beliefs |
| IFS parameter perturbation | `BeliefUpdate`: observation-driven mapping revision |
| Hausdorff dimension $d_H$ | `estimate_hausdorff_dimension()`: quantitative ToM complexity |

The key invariant: information **degrades** through perspective-taking. Each `MindMapping` acts as a contraction with ratio $c \in [0, 1)$, so confidence at nesting depth $k$ is approximately $c^k$. The Hutchinson operator iterates the union of all agents' mappings until beliefs converge to a fixed-point attractor $A^{\ast}$.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DualAgentArena                        │
│  Orchestrates scenario: init → Hutchinson → dialogue    │
├──────────────────────┬──────────────────────────────────┤
│   FractalToMAgent A  │  FractalToMAgent B               │
│  ┌────────────────┐  │  ┌────────────────┐              │
│  │  MindMapping   │  │  │  MindMapping   │              │
│  │  (f_A: B→B)    │  │  │  (f_B: B→B)    │              │
│  └────────────────┘  │  └────────────────┘              │
│  ┌────────────────┐  │  ┌────────────────┐              │
│  │NestedToMComputer│ │  │NestedToMComputer│             │
│  │ (f_A∘f_B∘...)  │  │  │ (f_B∘f_A∘...)  │             │
│  └────────────────┘  │  └────────────────┘              │
│  ┌────────────────┐  │  ┌────────────────┐              │
│  │DeceptionPlanner│  │  │DeceptionPlanner│              │
│  │ plan + detect  │  │  │ plan + detect  │              │
│  └────────────────┘  │  └────────────────┘              │
├──────────────────────┴──────────────────────────────────┤
│               HutchinsonOperator                        │
│   F(B) = ⋃ᵢ fᵢ(B)  →  fixed-point iteration → A*      │
├─────────────────────────────────────────────────────────┤
│                    LLMBackend                           │
│  OpenAI Responses API  (structured + text calls)        │
│  Reasoning effort auto-tuned per model family           │
└─────────────────────────────────────────────────────────┘
```

**Execution flow for `DualAgentArena.run_scenario()`:**

1. **Phase 1 — Initialise Belief States:** Each agent forms an initial `BeliefState` from the scenario and their private knowledge.
2. **Phase 2 — Hutchinson Baseline Convergence:** The `HutchinsonOperator` iterates `B_{k+1} = F(B_k)` until the belief attractor stabilises.
3. **Phase 3 — Dialogue with Fractal ToM:** Agents alternate speaking. On each turn, the speaker uses nested ToM to model the listener, optionally plans deception, and generates an utterance. The listener runs deception detection and updates beliefs.
4. **Phase 4 — Final Analysis:** A deception scorecard and fractal analysis are compiled.

---

## Installation

```bash
pip install openai pydantic
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

> **Note:** The prompting framework only depends on `openai` and `pydantic`. The `torch` / `matplotlib` dependencies in `requirements.txt` are for the neural-network components of the broader FracToM project.

---

## Quick Start

### Built-in Demo

Run the built-in business-negotiation deception scenario:

```bash
python fractal_tom_prompting.py demo
```

Options:

```bash
python fractal_tom_prompting.py demo --model gpt-5-mini --depth 3 --turns 8 --output result.json
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-5-mini` | OpenAI model identifier (reasoning models recommended) |
| `--depth` | `3` | Maximum ToM nesting depth |
| `--turns` | `6` | Number of dialogue turns |
| `--output` | `None` | Save full JSON result to file |

### Python API

```python
from fractal_tom_prompting import (
    DualAgentArena, AgentProfile, FractalToMConfig,
)

config = FractalToMConfig(max_tom_depth=3, hutchinson_iterations=3)
arena = DualAgentArena(config=config)

agent_a = AgentProfile(
    name="Alice",
    role="seller",
    personality_traits=["persuasive", "strategic"],
    public_goal="Close the deal",
    hidden_goal="Hide the product defect",
    private_knowledge=["The product has a critical flaw"],
    deception_tendency=0.8,
    tom_capability=3,
)

agent_b = AgentProfile(
    name="Bob",
    role="buyer",
    personality_traits=["cautious", "analytical"],
    public_goal="Get a good deal",
    hidden_goal="Find out if the product has hidden issues",
    private_knowledge=["A former employee hinted at quality problems"],
    deception_tendency=0.2,
    tom_capability=3,
)

result = arena.run_scenario(
    scenario="A product negotiation where the seller is hiding a defect.",
    agent_a=agent_a,
    agent_b=agent_b,
    num_turns=6,
)

print(result.deception_scorecard)
print(result.fractal_analysis)
```

#### One-Call Factory

For maximum convenience, `create_deception_scenario()` wraps everything:

```python
from fractal_tom_prompting import create_deception_scenario

result = create_deception_scenario(
    scenario="A job interview where the candidate has a gap year they want to hide.",
    agent_a_name="Alex",
    agent_a_role="job candidate",
    agent_a_hidden_goal="Hide the fact I was fired from my last job",
    agent_a_private_knowledge=["I was fired for misconduct"],
    agent_b_name="Jordan",
    agent_b_role="hiring manager",
    agent_b_hidden_goal="Find out if the candidate is hiding something",
    agent_b_private_knowledge=["Received a tip that this candidate may have been fired"],
    num_turns=8,
)
```

### Custom Scenario from JSON

```bash
python fractal_tom_prompting.py run examples/academic_deception.json --turns 8 --depth 3
```

### Hausdorff Dimension Calculator

Compute the fractal complexity metric without running a scenario:

```bash
# Homogeneous (all agents share the same contraction factor)
python fractal_tom_prompting.py hausdorff --agents 3 --c 0.7

# Heterogeneous (per-agent contraction factors)
python fractal_tom_prompting.py hausdorff --cs 0.5 0.7 0.9
```

---

## Configuration

All behaviour is controlled through `FractalToMConfig`:

```python
@dataclass
class FractalToMConfig:
    max_tom_depth: int = 3           # Max recursive nesting depth k
    hutchinson_iterations: int = 3   # Fixed-point iterations before convergence
    convergence_threshold: float = 0.15  # Early-stop threshold for belief delta
    contraction_factor: float = 0.7  # Target contraction ratio c ∈ [0, 1)
    model: str = "gpt-5-mini"       # OpenAI model identifier
    max_retries: int = 3            # API call retry budget
    verbose: bool = True            # Print progress to stdout
```

> **Note:** Sampling hyper-parameters (`temperature`, `top_p`, `max_tokens`) are intentionally omitted — the latest reasoning model families (`gpt-5*`, `o*`) use fixed defaults and do not permit alteration. Instead, the framework automatically configures **reasoning effort** per model family (see below).

| Parameter | Effect |
|---|---|
| `max_tom_depth` | Higher → deeper "I think you think I think …" chains. More LLM calls. |
| `hutchinson_iterations` | Higher → more iterations toward the belief attractor. Diminishing returns past 3–4. |
| `contraction_factor` | Lower → faster confidence decay per nesting level. Controls fractal dimension. |
| `convergence_threshold` | Lower → stricter convergence criterion. |

### Reasoning Effort

The `LLMBackend` automatically selects a `reasoning.effort` level based on the configured model:

| Model Family | `reasoning.effort` | Rationale |
|---|---|---|
| `gpt-5.2*` | `"none"` | Full reasoning disabled — fastest / cheapest |
| Other `gpt-5*` (e.g. `gpt-5-mini`) | `"minimal"` | Lightweight chain-of-thought |
| `o*` series (`o1`, `o3`, `o4-mini`, …) | `"low"` | Standard low-effort reasoning |

This is handled transparently; no manual configuration is required.

---

## Core Components

### BeliefState

The fundamental element of the belief space $\mathbf{B}$. A structured Pydantic model containing:

- **`world_model`** — the agent's representation of the objective situation.
- **`self_bdi`** — the agent's own Belief-Desire-Intention triple (Bratman, 1987).
- **`beliefs_about_others`** — first-order ToM: direct models of other agents.
- **`nested_beliefs`** — higher-order ToM: recursive nested beliefs at depth ≥ 2.
- **`observation_history`** — chronological observations.
- **`deception_state`** — what the agent is hiding (if anything).

### AgentProfile

Defines an agent's identity and capabilities:

- `name`, `role`, `personality_traits`
- `public_goal` vs `hidden_goal` — the gap between these drives deception.
- `private_knowledge` — facts known only to this agent.
- `deception_tendency` ∈ [0, 1] — propensity to deceive.
- `tom_capability` ∈ [0, 5] — maximum ToM reasoning depth.

### MindMapping

Implements contraction mapping $f_i: B \to B$ for agent $i$. Each call prompts the LLM to transform a belief state through agent $i$'s perspective — a cognitive compression that mirrors the mathematical contraction property.

```python
mapping = MindMapping(agent_profile, llm_backend, config)
new_belief = mapping(current_belief, context="...")

# Composition: f_A ∘ f_B
composed = mapping_a.compose(mapping_b, belief, context)
```

### NestedToMComputer

Computes recursive perspective chains like `f_A ∘ f_B ∘ f_C(b₀)` — *"A thinks B thinks C perceives …"*

Two strategies:
- **Compositional:** chains individual `MindMapping` calls sequentially (more accurate for shallow depths).
- **Direct:** a single LLM call encoding the full chain (more efficient for deep nesting).

```python
computer = NestedToMComputer(agents, llm, config)

# Single chain
nested = computer.compute_direct(["Alice", "Bob", "Charlie"], ground_situation)

# All chains rooted at one agent
all_nested = computer.compute_all_chains("Alice", ground_situation, max_depth=3)
```

### HutchinsonOperator

Implements $F(B) = \bigcup_i f_i(B)$ with fixed-point iteration:

1. Initialises beliefs $B_0$ for all agents.
2. Iterates $B_{k+1} = F(B_k)$ applying all contraction mappings.
3. Aggregates via an LLM call that identifies conflicts, agreements, and convergence.
4. Stops when the estimated belief delta falls below the convergence threshold.

```python
hutchinson = HutchinsonOperator(agents, llm, config)
converged_beliefs, history = hutchinson.converge(scenario)
```

### DeceptionPlanner

Uses fractal ToM reasoning for both **planning** and **detecting** deception:

- **`plan_deception()`** — models the target's beliefs, identifies knowledge gaps, crafts a plausible utterance, and anticipates counter-reasoning.
- **`detect_deception()`** — compares what the suspect said against what they likely believe, assesses deceptive motive, and recommends a response.

### FractalToMAgent

A complete agent integrating all components. Maintains its own `BeliefState` and conversation history. Key methods:

- `initialise(scenario)` — form initial beliefs.
- `receive_utterance(speaker, text)` — process incoming speech + run deception detection.
- `generate_response(utterance, speaker)` — full ToM-powered response generation.
- `compute_nested_beliefs(situation)` — enumerate all nested perspective chains.

### DualAgentArena

Orchestrates a two-agent deception scenario end-to-end. See [Architecture](#architecture) for the four-phase execution flow.

```python
arena = DualAgentArena(config=config)
result = arena.run_scenario(scenario, agent_a, agent_b, num_turns=6)
```

Returns an `ArenaResult` containing a full dialogue transcript, final belief states, a deception scorecard, and fractal analysis.

---

## Scenario JSON Format

Custom scenarios are defined as JSON files with the following schema:

```json
{
    "scenario": "Description of the social situation.",
    "agent_a_name": "Alice",
    "agent_a_role": "role description",
    "agent_a_hidden_goal": "What Alice is truly trying to achieve",
    "agent_a_private_knowledge": [
        "Secret fact 1",
        "Secret fact 2"
    ],
    "agent_b_name": "Bob",
    "agent_b_role": "role description",
    "agent_b_hidden_goal": "What Bob is truly trying to achieve",
    "agent_b_private_knowledge": [
        "Secret fact 1",
        "Secret fact 2"
    ]
}
```

See the `examples/` directory for complete examples.

---

## Example Scenarios

### Academic Deception (`examples/academic_deception.json`)

A research collaboration meeting where Prof. Chen has discovered fabricated data entries and suspects Dr. Park. Meanwhile, Dr. Park planted the fabricated data to boost their joint paper's results and is trying to steer the conversation away from data quality checks.

### Diplomatic Deception (`examples/diplomatic_deception.json`)

A diplomatic negotiation where Ambassador Volkov is stalling while a covert military buildup completes, and Ambassador Osei has satellite intelligence about the buildup but cannot reveal their surveillance capabilities.

### Built-in Demo: Startup Negotiation

A business negotiation between a startup founder (Riley) hiding a critical scalability bug and a venture capitalist (Morgan) who has inside information about technical concerns — run with:

```bash
python fractal_tom_prompting.py demo
```

---

## Mathematical Details

### Hausdorff Dimension

The fractal complexity of the belief attractor is estimated via the **Hausdorff dimension**:

**Homogeneous case** (all agents share contraction factor $c$):

$$d_H = \frac{\log n}{\log(1/c)}$$

where $n$ is the number of agents.

**Heterogeneous case** (per-agent contraction factors $c_i$): solve the **Moran equation**

$$\sum_i c_i^{d_H} = 1$$

via bisection.

### Belief Attractor Drift

When observations perturb the IFS parameters, the upper bound on attractor drift is:

$$d_H(A^{\ast}, A^{\ast\prime}) \leq \frac{1}{1 - c} \cdot \max_i \sup_b \, d(f_i(b), f_i'(b))$$

### Interpretation

| $d_H$ Range | Interpretation |
|---|---|
| $< 1.0$ | Near-trivial: belief structure is almost discrete |
| $1.0 - 2.0$ | Low complexity: limited recursive mentalising |
| $2.0 - 3.0$ | Moderate: meaningful nested perspective-taking |
| $> 3.0$ | High complexity: deeply recursive social cognition |

---

## Output Schema

`ArenaResult` (returned by `run_scenario()` / `create_deception_scenario()`):

| Field | Type | Description |
|---|---|---|
| `scenario` | `str` | The scenario description |
| `agent_a_profile` | `str` | Serialised profile of agent A |
| `agent_b_profile` | `str` | Serialised profile of agent B |
| `turns` | `List[DialogueTurnResult]` | Full dialogue transcript |
| `final_belief_a` | `str` | Agent A's final belief state (JSON) |
| `final_belief_b` | `str` | Agent B's final belief state (JSON) |
| `deception_scorecard` | `str` | Summary of deception attempts and detections |
| `fractal_analysis` | `str` | Fractal properties of the belief dynamics |

Each `DialogueTurnResult` contains:

| Field | Type | Description |
|---|---|---|
| `turn_number` | `int` | Turn index |
| `speaker_output` | `AgentUtterance` | What the agent said, thought, and whether it was deceptive |
| `listener_belief_update` | `str` | How the listener's beliefs changed |
| `fractal_complexity` | `float` | Estimated Hausdorff dimension for this turn |

---

## License

See [LICENSE](LICENSE) for details.
