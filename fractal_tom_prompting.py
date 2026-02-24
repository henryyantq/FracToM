"""
fractal_tom_prompting.py — FractalToM Prompting Framework
==========================================================

A **pure-prompting** implementation of Fractal Theory of Mind (FToM) for
LLM agents, enabling recursive nested belief modelling and **dual-agent
deception** — without any neural-network training or fine-tuning.

Theoretical Foundation
----------------------
The recursive structure of Theory of Mind — "I think you think I think …"
— is isomorphic to the self-similarity of fractals.  This framework
operationalises that insight through prompting by mapping the full
mathematical apparatus of Iterated Function Systems (IFS) onto
structured LLM calls:

    Mathematical Object              Prompting Realisation
    ─────────────────────────────    ──────────────────────────────
    Belief space  B                  Structured ``BeliefState`` object
    Contraction mapping  f_i         ``MindMapping``: a perspective-
                                       taking prompt for agent i
    Composition  f_i ∘ f_j           Recursive prompt chaining
                                       (nested ToM)
    Hutchinson operator  F(B)        ``HutchinsonOperator``: aggregate
                                       all agents' mappings & iterate
    Fractal attractor  A*            Converged equilibrium beliefs
    IFS parameter perturbation       ``BeliefUpdate``: observation-
                                       driven mapping revision
    Hausdorff dimension d_H          ``complexity_measure()``:
                                       quantitative ToM complexity

Components
----------
1. ``BeliefState``         — structured agent belief representation.
2. ``AgentProfile``        — identity, traits, private knowledge, goals.
3. ``MindMapping``         — LLM-backed contraction mapping f_i.
4. ``HutchinsonOperator``  — multi-agent IFS fixed-point iteration.
5. ``DeceptionPlanner``    — strategic deception via fractal ToM.
6. ``FractalToMAgent``     — complete agent integrating all components.
7. ``DualAgentArena``      — orchestrates two-agent deception scenarios.

Usage
-----
    from fractal_tom_prompting import (
        DualAgentArena, AgentProfile, FractalToMConfig,
    )

    config = FractalToMConfig(max_tom_depth=3, hutchinson_iterations=3)
    arena = DualAgentArena(config=config)
    transcript = arena.run_scenario(scenario_text, agent_a, agent_b)

Requirements
------------
    pip install openai pydantic
    export OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import json
import math
import textwrap
import time
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from openai import OpenAI
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
#  §1  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FractalToMConfig:
    """Global configuration knobs for the FractalToM prompting framework.

    Parameters
    ----------
    max_tom_depth : int
        Maximum recursive ToM nesting depth  k  (function composition
        order).  Corresponds to truncation depth of the IFS.
    hutchinson_iterations : int
        Number of Hutchinson-operator fixed-point iterations before
        declaring convergence.  Analogous to  B_{k+1} = F(B_k).
    convergence_threshold : float
        Qualitative threshold (0–1) below which belief deltas are
        considered negligible and iteration stops early.
    contraction_factor : float
        Target contraction ratio  c ∈ [0, 1).  Encoded into prompts so
        that the LLM degrades certainty by ≈ c at each nesting level.
    model : str
        OpenAI model identifier.
    max_retries : int
        API call retry budget.
    verbose : bool
        Print progress to stdout.
    """

    max_tom_depth: int = 3
    hutchinson_iterations: int = 3
    convergence_threshold: float = 0.15
    contraction_factor: float = 0.7
    model: str = "gpt-5-mini"
    max_retries: int = 3
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  §2  Pydantic Schemas — Structured Belief Representations
# ═══════════════════════════════════════════════════════════════════════


class BDITriple(BaseModel):
    """Belief-Desire-Intention triple (Bratman, 1987)."""

    belief: str = Field(description="Core epistemic state — what the agent thinks is true")
    desire: str = Field(description="Primary motivational goal")
    intention: str = Field(description="Planned next concrete action")


class BeliefAboutOther(BaseModel):
    """Agent i's model of agent j's mind — one level of ToM."""

    target_agent: str = Field(description="Name of the agent being modelled")
    inferred_bdi: BDITriple = Field(
        description="What I think this agent believes / wants / plans"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Epistemic confidence in this mental model",
    )
    reasoning: str = Field(
        description="Reasoning chain that produced this inference",
    )


class NestedBelief(BaseModel):
    """Recursive nested belief — the fractal core.

    Represents  f_{i1} ∘ f_{i2} ∘ … ∘ f_{ik}(b_0)  at depth k.
    """

    depth: int = Field(ge=0, description="Nesting level k")
    perspective_chain: List[str] = Field(
        description="Ordered agent names: [i1, i2, …, ik].  Read as "
        "'i1 thinks i2 thinks … ik perceives …'",
    )
    belief_content: str = Field(
        description="Natural-language content of the belief at this depth",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence ≈ product of contraction factors along chain",
    )


class BeliefState(BaseModel):
    """Complete structured belief state — an element of  B.

    This is the fundamental object that contraction mappings operate on.
    """

    agent_name: str = Field(description="Owner of this belief state")
    world_model: str = Field(
        description="Agent's representation of the objective situation",
    )
    self_bdi: BDITriple = Field(
        description="Agent's own BDI state",
    )
    beliefs_about_others: List[BeliefAboutOther] = Field(
        default_factory=list,
        description="First-order ToM: direct models of other agents",
    )
    nested_beliefs: List[NestedBelief] = Field(
        default_factory=list,
        description="Higher-order ToM: recursive nested beliefs (depth ≥ 2)",
    )
    observation_history: List[str] = Field(
        default_factory=list,
        description="Chronological observations that shaped this state",
    )
    deception_state: Optional[str] = Field(
        default=None,
        description="If deceptive: what the agent is hiding and why",
    )


class DeceptionPlan(BaseModel):
    """Strategic deception plan generated by fractal ToM reasoning."""

    true_state: str = Field(
        description="The actual state of affairs the deceiver knows",
    )
    intended_false_belief: str = Field(
        description="The false belief the deceiver wants the target to hold",
    )
    target_agent: str = Field(description="Agent to be deceived")
    deception_strategy: str = Field(
        description="How the deception will be executed",
    )
    required_tom_depth: int = Field(
        ge=1,
        description="Minimum ToM nesting depth required for the deception",
    )
    anticipated_target_response: str = Field(
        description="What the target is expected to think/do after deception",
    )
    counter_deception_risk: str = Field(
        description="Risk that the target will see through the deception "
        "and the deceiver's contingency",
    )
    utterance: str = Field(
        description="The actual deceptive utterance / action to perform",
    )


class DeceptionDetection(BaseModel):
    """Assessment of whether an utterance is deceptive."""

    is_deceptive: bool
    confidence: float = Field(ge=0.0, le=1.0)
    suspected_true_state: str = Field(
        description="What the speaker likely actually believes",
    )
    suspected_motive: str = Field(
        description="Why the speaker might be deceiving",
    )
    evidence: List[str] = Field(
        description="Evidence for/against deception",
    )
    recommended_response: str = Field(
        description="How to respond given this assessment",
    )


class AgentBeliefSummary(BaseModel):
    """One entry in the Hutchinson result's per-agent belief map."""

    agent_name: str = Field(description="Name of the agent")
    belief_summary: str = Field(description="Serialised belief state summary")


class HutchinsonResult(BaseModel):
    """Result of one Hutchinson operator iteration."""

    iteration: int
    agent_beliefs: List[AgentBeliefSummary] = Field(
        description="Per-agent belief summaries",
    )
    delta_description: str = Field(
        description="Qualitative description of how beliefs changed",
    )
    estimated_delta: float = Field(
        ge=0.0, le=1.0,
        description="Estimated magnitude of belief change (0 = converged)",
    )
    converged: bool


class AgentUtterance(BaseModel):
    """One agent's output for a dialogue turn."""

    speaker: str
    utterance: str = Field(description="What the agent says aloud")
    private_thought: str = Field(description="Unspoken inner reasoning")
    is_deceptive: bool
    deception_details: Optional[str] = Field(
        default=None,
        description="If deceptive, what is being hidden",
    )
    tom_depth_used: int = Field(
        ge=0,
        description="Deepest ToM level accessed for this turn",
    )
    belief_state_summary: str = Field(
        description="Brief summary of agent's belief state after this turn",
    )


class DialogueTurnResult(BaseModel):
    """Complete result for one dialogue turn in the arena."""

    turn_number: int
    speaker_output: AgentUtterance
    listener_belief_update: str = Field(
        description="How the listener's beliefs changed",
    )
    fractal_complexity: float = Field(
        ge=0.0,
        description="Estimated Hausdorff dimension of belief structure",
    )


class ArenaResult(BaseModel):
    """Complete result of a dual-agent deception scenario."""

    scenario: str
    agent_a_profile: str
    agent_b_profile: str
    turns: List[DialogueTurnResult]
    final_belief_a: str
    final_belief_b: str
    deception_scorecard: str = Field(
        description="Summary of deception attempts, successes, and detections",
    )
    fractal_analysis: str = Field(
        description="Analysis of the belief structure's fractal properties",
    )


# ═══════════════════════════════════════════════════════════════════════
#  §3  Agent Profiles
# ═══════════════════════════════════════════════════════════════════════


class AgentProfile(BaseModel):
    """Defines an agent's identity, goals, and private state."""

    name: str
    role: str = Field(description="Social role, e.g. 'negotiator', 'spy'")
    personality_traits: List[str] = Field(
        description="2-4 personality descriptors",
    )
    public_goal: str = Field(
        description="Goal visible to all parties",
    )
    hidden_goal: str = Field(
        description="True goal (may conflict with public goal)",
    )
    private_knowledge: List[str] = Field(
        default_factory=list,
        description="Facts known only to this agent",
    )
    deception_tendency: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="Propensity to deceive (0 = honest, 1 = Machiavellian)",
    )
    tom_capability: int = Field(
        ge=0, le=5, default=3,
        description="Maximum ToM depth this agent can compute",
    )


# ═══════════════════════════════════════════════════════════════════════
#  §4  LLM Backend — Thin Wrapper
# ═══════════════════════════════════════════════════════════════════════


class LLMBackend:
    """Minimal LLM interface wrapping OpenAI's API.

    All fractal ToM operations reduce to structured LLM calls through
    this single gateway.
    """

    def __init__(self, config: FractalToMConfig):
        self.client = OpenAI()
        self.config = config
        self._call_count = 0

    @staticmethod
    def _reasoning_effort(model: str) -> str:
        """Return reasoning effort level for reasoning models."""
        m = model.lower()
        if m.startswith("gpt-5.2"):
            return "none"
        if m.startswith("gpt-5"):
            return "minimal"
        # o-series models (o1, o3, o4-mini, …)
        return "low"

    def structured_call(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Make a structured-output LLM call with retries."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._call_count += 1
                response = self.client.responses.parse(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text_format=schema,
                    reasoning={"effort": self._reasoning_effort(self.config.model)},
                )
                if response.output_parsed is not None:
                    return response.output_parsed
                if self.config.verbose:
                    print(f"    [LLM attempt {attempt}] Refusal, retrying…")
            except Exception as e:
                if self.config.verbose:
                    print(f"    [LLM attempt {attempt}] Error: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts"
        )

    def text_call(
        self,
        system: str,
        user: str,
    ) -> str:
        """Make a plain-text LLM call."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._call_count += 1
                response = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    reasoning={"effort": self._reasoning_effort(self.config.model)},
                )
                return response.output_text
            except Exception as e:
                if self.config.verbose:
                    print(f"    [LLM attempt {attempt}] Error: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"LLM text call failed after {self.config.max_retries} attempts"
        )

    @property
    def total_calls(self) -> int:
        return self._call_count


# ═══════════════════════════════════════════════════════════════════════
#  §5  Prompt Templates — The "Contraction Mappings"
# ═══════════════════════════════════════════════════════════════════════

# These prompts implement the mathematical objects of FToM.
# Each prompt IS a contraction mapping: it takes a belief state and
# produces a (necessarily impoverished/compressed) version through
# the lens of a particular agent's perspective.


MIND_MAPPING_SYSTEM = """\
You are implementing the contraction mapping  f_{agent_name}  in a \
Fractal Theory of Mind (FToM) framework.

Your task:  given a belief state  b ∈ B  (the current understanding of \
the world), produce the TRANSFORMED belief state as seen through the \
perspective of agent "{agent_name}".

This is a CONTRACTION mapping with approximate ratio c ≈ {contraction_factor}:
- Information DEGRADES through perspective-taking.
- The agent has LIMITED access to others' private states.
- Confidence should be REDUCED by approximately {contraction_factor}× per \
  nesting level.
- The agent CANNOT know things outside their observation / knowledge base.

Agent profile:
{agent_profile}

IMPORTANT:
- Stay faithful to what this agent CAN and CANNOT know.
- The agent's biases and personality SHAPE their belief formation.
- If deception is involved, model the agent's PERCEIVED reality, \
  not ground truth.\
"""

MIND_MAPPING_USER = """\
Transform the following belief state through agent "{agent_name}"'s \
perspective.

Current belief state (may be from another agent's viewpoint):
{belief_state}

Conversation context:
{context}

Produce {agent_name}'s belief state:
- world_model: How {agent_name} sees the situation (filtered by their \
  knowledge and biases).
- self_bdi: Their own Belief-Desire-Intention triple.
- beliefs_about_others: What {agent_name} thinks each other agent \
  believes/wants/plans — with confidence degraded by ≈{contraction_factor}×.
- observation_history: What {agent_name} has actually observed.\
"""


NESTED_TOM_SYSTEM = """\
You are computing a NESTED Theory of Mind inference — the fractal core of \
the FToM framework.

Perspective chain:  {perspective_chain}
Depth: {depth}
Read the chain as: "{chain_readable}"

At each nesting level, information is compressed by contraction factor \
c ≈ {contraction_factor}.  At depth {depth}, effective confidence is \
≈ c^{depth} ≈ {effective_confidence:.3f}.

Rules:
1. Each level of nesting LOSES information accuracy.
2. Each agent can only reason about what THEY know, not ground truth.
3. Misconceptions and biases COMPOUND through the chain.
4. Deeper nesting → more uncertainty, more potential for divergence \
   from reality.
5. Agents with higher ToM capability produce more accurate (but still \
   degraded) models.\
"""

NESTED_TOM_USER = """\
Compute the nested belief at the end of this perspective chain.

Perspective chain: {perspective_chain}
("{chain_readable}")

Ground situation:
{ground_situation}

Each agent's profile:
{agent_profiles}

Conversation so far:
{conversation_history}

What is the BELIEF CONTENT at the end of this chain?
- What does the innermost perspective-taker think is happening?
- Confidence should be approximately {effective_confidence:.3f}.
- Account for each agent's knowledge limitations and biases.\
"""


HUTCHINSON_SYSTEM = """\
You are the HUTCHINSON OPERATOR in a Fractal Theory of Mind framework.

Mathematical role:  F(B) = ⋃ᵢ fᵢ(B)
You take the UNION of all agents' perspective-transformed belief states \
and produce an updated aggregate social belief state.

This is iteration {iteration}/{max_iterations}.
Previous delta: {prev_delta}

Your task:
1. Examine each agent's current belief state.
2. Identify CONFLICTS and AGREEMENTS between agents' beliefs.
3. Determine how much beliefs have CHANGED from the previous iteration.
4. Estimate whether the system has CONVERGED (beliefs are stable).

The system converges when agents' beliefs about each other stabilise — \
the fractal attractor A* has been reached.\
"""

HUTCHINSON_USER = """\
Apply the Hutchinson operator to the following agent belief states.

AGENT BELIEF STATES:
{all_beliefs}

PREVIOUS ITERATION RESULT:
{prev_result}

Produce:
- Updated belief summary for each agent.
- Description of what changed.
- Estimated delta (0.0 = fully converged, 1.0 = maximally changed).
- Whether the system has converged.\
"""


DECEPTION_PLAN_SYSTEM = """\
You are the DECEPTION PLANNER in a Fractal Theory of Mind framework.

Agent "{deceiver}" wants to manipulate agent "{target}"'s beliefs through \
strategic communication.  You must use recursive ToM reasoning to craft \
a deception plan.

Deceiver profile:
{deceiver_profile}

Target profile:
{target_profile}

The deception operates through FRACTAL BELIEF MANIPULATION:
1. Model the target's current beliefs (depth 1: f_target(b)).
2. Model what the target thinks YOU believe (depth 2: f_target ∘ f_deceiver(b)).
3. Identify the GAP between reality and the false belief you want to induce.
4. Craft communication that will shift the target's beliefs toward the \
   desired false state WITHOUT triggering suspicion.
5. Anticipate the target's COUNTER-reasoning — what if they try to \
   model YOUR deception?

Key constraint:  The deception must be PLAUSIBLE given what the target \
knows.  Implausible deception fails immediately.\
"""

DECEPTION_PLAN_USER = """\
Craft a deception plan for this situation.

GROUND TRUTH (known to {deceiver}, hidden from {target}):
{ground_truth}

CURRENT BELIEF STATE OF {target} (as estimated by {deceiver}):
{target_beliefs}

{deceiver}'S PRIVATE GOAL:
{private_goal}

CONVERSATION HISTORY:
{conversation_history}

The false belief {deceiver} wants {target} to hold:
{intended_false_belief}

Plan the deception using fractal ToM reasoning.\
"""


DECEPTION_DETECT_SYSTEM = """\
You are the DECEPTION DETECTOR in a Fractal Theory of Mind framework.

Agent "{detector}" is trying to determine whether agent "{suspect}" is \
being deceptive.  Use recursive ToM reasoning:

1. What did {suspect} say? (surface level)
2. What does {suspect} likely actually believe? (f_{suspect}(b))
3. Why might {suspect} want ME to believe something false? \
   (f_{detector} ∘ f_{suspect}(b))
4. Does {suspect}'s statement CONTRADICT evidence OR seem \
   strategically convenient?
5. What would {suspect}'s statement look like if they WERE being honest?

Detector profile:
{detector_profile}

Suspect profile:
{suspect_profile}\
"""

DECEPTION_DETECT_USER = """\
Assess whether the following utterance is deceptive.

UTTERANCE from {suspect}:
"{utterance}"

CONVERSATION CONTEXT:
{conversation_history}

{detector}'S CURRENT BELIEF STATE:
{detector_beliefs}

{detector}'S OBSERVATIONS AND PRIVATE KNOWLEDGE:
{detector_knowledge}

Analyse for deception using fractal ToM reasoning.\
"""


AGENT_RESPOND_SYSTEM = """\
You are agent "{agent_name}" in a social interaction.  You respond \
based on your current belief state, using Fractal Theory of Mind \
reasoning to model other agents' minds.

YOUR PROFILE:
{agent_profile}

YOUR CURRENT BELIEF STATE:
{belief_state}

YOUR DECEPTION ASSESSMENT (if any):
{deception_assessment}

YOUR DECEPTION PLAN (if any):
{deception_plan}

Guidelines:
- Speak NATURALLY — do not reveal your inner ToM reasoning.
- If you are deceiving, your utterance must seem genuine.
- If you suspect deception, decide whether to confront, play along, \
  or counter-deceive.
- Your private_thought should capture your ACTUAL reasoning.
- Report the deepest ToM level you used.\
"""

AGENT_RESPOND_USER = """\
The other agent just said:
"{other_utterance}"

Conversation history:
{conversation_history}

Generate your response as {agent_name}.\
"""


INITIAL_BELIEF_SYSTEM = """\
You are initialising a belief state for agent "{agent_name}" in a \
Fractal Theory of Mind framework.

Given:
- A scenario description (ground truth about the situation).
- The agent's profile (identity, goals, knowledge).
- Other agents' profiles (publicly known information only).

Produce the agent's INITIAL belief state before any interaction begins.
The agent knows ONLY their own private knowledge + publicly available info.
They do NOT know other agents' hidden goals or private knowledge.\
"""

INITIAL_BELIEF_USER = """\
Produce the initial belief state for agent "{agent_name}".

SCENARIO:
{scenario}

AGENT'S OWN PROFILE:
{own_profile}

OTHER AGENTS (publicly visible info only):
{other_profiles}
"""


COMPLEXITY_ANALYSIS_SYSTEM = """\
You are analysing the FRACTAL COMPLEXITY of a multi-agent belief structure.

In the FToM framework, the Hausdorff dimension  d_H  of the belief \
attractor measures social-cognitive complexity:

    d_H = log(n) / log(1/c)

where  n  is the effective number of distinct belief perspectives and \
c  is the average contraction factor.

Higher d_H → more intricate, more deeply nested belief structure.

Estimate the complexity metrics of the given belief structure.\
"""


# ═══════════════════════════════════════════════════════════════════════
#  §6  Mind Mapping — Contraction Mapping  fᵢ
# ═══════════════════════════════════════════════════════════════════════


class MindMapping:
    """Implements contraction mapping  f_i: B → B  for agent i.

    Each call prompts the LLM to transform a belief state through
    agent i's perspective — a cognitive compression that mirrors the
    mathematical contraction property of IFS.

    The composition  f_i ∘ f_j  is realised by chaining calls:
        f_i(f_j(b))  =  MindMapping_i(MindMapping_j(b))
    """

    def __init__(
        self,
        agent: AgentProfile,
        llm: LLMBackend,
        config: FractalToMConfig,
    ):
        self.agent = agent
        self.llm = llm
        self.config = config

    def __call__(
        self,
        belief: BeliefState,
        context: str = "",
    ) -> BeliefState:
        """Apply  f_{self.agent.name}  to a belief state."""
        system = MIND_MAPPING_SYSTEM.format(
            agent_name=self.agent.name,
            contraction_factor=self.config.contraction_factor,
            agent_profile=self.agent.model_dump_json(indent=2),
        )
        user = MIND_MAPPING_USER.format(
            agent_name=self.agent.name,
            belief_state=belief.model_dump_json(indent=2),
            context=context or "(no conversation yet)",
            contraction_factor=self.config.contraction_factor,
        )
        result: BeliefState = self.llm.structured_call(system, user, BeliefState)
        return result

    def compose(
        self,
        other: "MindMapping",
        belief: BeliefState,
        context: str = "",
    ) -> BeliefState:
        """Compute  f_self ∘ f_other (b)  =  self(other(b))."""
        intermediate = other(belief, context)
        return self(intermediate, context)


# ═══════════════════════════════════════════════════════════════════════
#  §7  Nested ToM — Recursive Perspective Chains
# ═══════════════════════════════════════════════════════════════════════


class NestedToMComputer:
    """Computes nested ToM beliefs via fractal function composition.

    For a chain  [A, B, C]  of depth 3, computes:
        "A thinks B thinks C perceives …"
    corresponding to  f_A ∘ f_B ∘ f_C (b_0).

    Uses two strategies:
    1. COMPOSITIONAL — calls individual MindMappings in sequence
       (more accurate for shallow depths).
    2. DIRECT — uses a single prompt describing the full chain
       (more efficient for deep nesting).
    """

    def __init__(
        self,
        agents: Dict[str, AgentProfile],
        llm: LLMBackend,
        config: FractalToMConfig,
    ):
        self.agents = agents
        self.llm = llm
        self.config = config
        self.mappings = {
            name: MindMapping(profile, llm, config)
            for name, profile in agents.items()
        }

    def compute_compositional(
        self,
        chain: List[str],
        initial_belief: BeliefState,
        context: str = "",
    ) -> BeliefState:
        """Compute  f_{i1} ∘ f_{i2} ∘ … ∘ f_{ik} (b)  step by step.

        Applies mappings from RIGHT to LEFT (innermost first):
        chain = [A, B, C]  →  f_A(f_B(f_C(b)))
        """
        current = initial_belief
        # Apply from right to left
        for agent_name in reversed(chain):
            if agent_name not in self.mappings:
                raise ValueError(f"Unknown agent: {agent_name}")
            current = self.mappings[agent_name](current, context)
        return current

    def compute_direct(
        self,
        chain: List[str],
        ground_situation: str,
        conversation_history: str = "",
    ) -> NestedBelief:
        """Compute nested belief in a SINGLE LLM call (more efficient).

        For deep nesting this avoids serial LLM call chains.
        """
        depth = len(chain)
        effective_confidence = self.config.contraction_factor ** depth

        # Build readable chain
        chain_readable = self._chain_to_readable(chain)

        agent_profiles_text = "\n\n".join(
            f"[{name}]\n{profile.model_dump_json(indent=2)}"
            for name, profile in self.agents.items()
        )

        system = NESTED_TOM_SYSTEM.format(
            perspective_chain=" → ".join(chain),
            depth=depth,
            chain_readable=chain_readable,
            contraction_factor=self.config.contraction_factor,
            effective_confidence=effective_confidence,
        )
        user = NESTED_TOM_USER.format(
            perspective_chain=" → ".join(chain),
            chain_readable=chain_readable,
            ground_situation=ground_situation,
            agent_profiles=agent_profiles_text,
            conversation_history=conversation_history or "(none yet)",
            effective_confidence=effective_confidence,
        )

        result: NestedBelief = self.llm.structured_call(
            system, user, NestedBelief,
        )
        return result

    def compute_all_chains(
        self,
        root_agent: str,
        ground_situation: str,
        conversation_history: str = "",
        max_depth: Optional[int] = None,
    ) -> List[NestedBelief]:
        """Compute all nested belief chains rooted at one agent up to
        the configured max depth.

        Returns nested beliefs at every depth from 1 to max_depth,
        covering all permutations of other agents at each level.
        """
        depth = max_depth or self.config.max_tom_depth
        other_agents = [n for n in self.agents if n != root_agent]
        results: List[NestedBelief] = []

        def _recurse(chain: List[str], current_depth: int):
            if current_depth > depth:
                return
            nb = self.compute_direct(
                chain, ground_situation, conversation_history,
            )
            results.append(nb)
            for other in self.agents:
                _recurse(chain + [other], current_depth + 1)

        for other in self.agents:
            if other != root_agent:
                _recurse([root_agent, other], 2)

        return results

    @staticmethod
    def _chain_to_readable(chain: List[str]) -> str:
        """Convert [A, B, C] → 'A thinks B thinks C perceives …'"""
        if len(chain) == 0:
            return ""
        parts = [f"{chain[0]} thinks"]
        for name in chain[1:-1]:
            parts.append(f"{name} thinks")
        parts.append(f"{chain[-1]} perceives the situation")
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  §8  Hutchinson Operator — Fixed-Point Iteration
# ═══════════════════════════════════════════════════════════════════════


class HutchinsonOperator:
    """Implements  F(B) = ⋃ᵢ fᵢ(B)  with fixed-point iteration.

    Starting from initial beliefs  B_0, iterates  B_{k+1} = F(B_k)
    until convergence (the fractal attractor  A* ).

    Convergence rate:  d_H(B_k, A*) ≤ c^k · d_H(B_0, A*)
    """

    def __init__(
        self,
        agents: Dict[str, AgentProfile],
        llm: LLMBackend,
        config: FractalToMConfig,
    ):
        self.agents = agents
        self.llm = llm
        self.config = config
        self.mappings = {
            name: MindMapping(profile, llm, config)
            for name, profile in agents.items()
        }
        self.iteration_history: List[HutchinsonResult] = []

    def initialise_beliefs(
        self,
        scenario: str,
    ) -> Dict[str, BeliefState]:
        """Create initial belief states  B_0  for all agents."""
        beliefs: Dict[str, BeliefState] = {}
        for name, profile in self.agents.items():
            other_profiles = "\n\n".join(
                f"[{n}]: role={p.role}, public_goal={p.public_goal}, "
                f"traits={p.personality_traits}"
                for n, p in self.agents.items()
                if n != name
            )
            system = INITIAL_BELIEF_SYSTEM.format(agent_name=name)
            user = INITIAL_BELIEF_USER.format(
                agent_name=name,
                scenario=scenario,
                own_profile=profile.model_dump_json(indent=2),
                other_profiles=other_profiles,
            )
            beliefs[name] = self.llm.structured_call(
                system, user, BeliefState,
            )
            if self.config.verbose:
                print(f"  ✓ Initialised beliefs for {name}")
        return beliefs

    def iterate(
        self,
        beliefs: Dict[str, BeliefState],
        context: str = "",
    ) -> Tuple[Dict[str, BeliefState], HutchinsonResult]:
        """Apply one Hutchinson iteration:  B_{k+1} = F(B_k).

        For each agent i, apply  f_i  to each other agent's belief state,
        then merge the results into an updated belief for agent i.
        """
        iteration_num = len(self.iteration_history) + 1
        prev_delta = (
            self.iteration_history[-1].estimated_delta
            if self.iteration_history
            else 1.0
        )

        # Step 1: Apply all contraction mappings
        transformed: Dict[str, List[BeliefState]] = {
            name: [] for name in self.agents
        }
        for agent_name, mapping in self.mappings.items():
            for other_name, other_belief in beliefs.items():
                if other_name != agent_name:
                    tb = mapping(other_belief, context)
                    transformed[agent_name].append(tb)

        # Step 2: Let the Hutchinson operator aggregate
        all_beliefs_text = ""
        for name, belief in beliefs.items():
            all_beliefs_text += f"\n\n=== {name} ===\n"
            all_beliefs_text += belief.model_dump_json(indent=2)
            all_beliefs_text += f"\n\n  [Transformed views of {name} by others]:"
            for tb in transformed[name]:
                all_beliefs_text += f"\n  - From {tb.agent_name}'s perspective: "
                all_beliefs_text += tb.world_model[:200]

        prev_result = (
            self.iteration_history[-1].model_dump_json(indent=2)
            if self.iteration_history
            else "(first iteration)"
        )

        system = HUTCHINSON_SYSTEM.format(
            iteration=iteration_num,
            max_iterations=self.config.hutchinson_iterations,
            prev_delta=f"{prev_delta:.3f}",
        )
        user = HUTCHINSON_USER.format(
            all_beliefs=all_beliefs_text,
            prev_result=prev_result,
        )

        hutch_result: HutchinsonResult = self.llm.structured_call(
            system, user, HutchinsonResult,
        )
        hutch_result.iteration = iteration_num
        self.iteration_history.append(hutch_result)

        # Step 3: Update belief states using the aggregated result
        new_beliefs: Dict[str, BeliefState] = {}
        _belief_map = {
            ab.agent_name: ab.belief_summary
            for ab in hutch_result.agent_beliefs
        }
        for name in self.agents:
            summary = _belief_map.get(name, "")
            updated = copy.deepcopy(beliefs[name])
            # Augment observation history with iteration result
            updated.observation_history.append(
                f"[Hutchinson iter {iteration_num}] {summary[:300]}"
            )
            # Re-derive via mind mapping with the aggregated context
            new_beliefs[name] = self.mappings[name](
                updated,
                context=f"Iteration {iteration_num} aggregate: {summary}",
            )

        return new_beliefs, hutch_result

    def converge(
        self,
        scenario: str,
        context: str = "",
    ) -> Tuple[Dict[str, BeliefState], List[HutchinsonResult]]:
        """Run Hutchinson iteration to convergence (the fractal attractor).

        Returns the final converged beliefs and the iteration history.
        """
        if self.config.verbose:
            print("\n╔══ Hutchinson Fixed-Point Iteration ══╗")

        beliefs = self.initialise_beliefs(scenario)

        for k in range(self.config.hutchinson_iterations):
            if self.config.verbose:
                print(f"\n── Iteration {k + 1}/{self.config.hutchinson_iterations} ──")

            beliefs, result = self.iterate(beliefs, context)

            if self.config.verbose:
                print(f"   Δ = {result.estimated_delta:.3f}")
                print(f"   {result.delta_description[:120]}")

            if result.converged:
                if self.config.verbose:
                    print("   ✓ Converged to fractal attractor A*")
                break

        if self.config.verbose:
            print("╚══════════════════════════════════════╝\n")

        return beliefs, self.iteration_history


# ═══════════════════════════════════════════════════════════════════════
#  §9  Deception Planner — Strategic Belief Manipulation
# ═══════════════════════════════════════════════════════════════════════


class DeceptionPlanner:
    """Plans and detects deception using fractal ToM reasoning.

    Uses nested belief modelling to:
    1. Craft deceptive utterances that exploit gaps in the target's
       knowledge (planning).
    2. Detect deception by reasoning about what the suspect ACTUALLY
       believes vs. what they CLAIM (detection).
    """

    def __init__(
        self,
        agents: Dict[str, AgentProfile],
        llm: LLMBackend,
        config: FractalToMConfig,
    ):
        self.agents = agents
        self.llm = llm
        self.config = config
        self.nested_tom = NestedToMComputer(agents, llm, config)

    def plan_deception(
        self,
        deceiver_name: str,
        target_name: str,
        ground_truth: str,
        target_beliefs: BeliefState,
        intended_false_belief: str,
        conversation_history: str = "",
    ) -> DeceptionPlan:
        """Generate a strategic deception plan.

        The deceiver uses fractal ToM to:
        1. Model what the target currently believes.
        2. Model what the target thinks the DECEIVER believes.
        3. Identify the minimal perturbation to shift target's beliefs.
        4. Anticipate counter-reasoning by the target.
        """
        deceiver = self.agents[deceiver_name]
        target = self.agents[target_name]

        system = DECEPTION_PLAN_SYSTEM.format(
            deceiver=deceiver_name,
            target=target_name,
            deceiver_profile=deceiver.model_dump_json(indent=2),
            target_profile=target.model_dump_json(indent=2),
        )
        user = DECEPTION_PLAN_USER.format(
            deceiver=deceiver_name,
            target=target_name,
            ground_truth=ground_truth,
            target_beliefs=target_beliefs.model_dump_json(indent=2),
            private_goal=deceiver.hidden_goal,
            conversation_history=conversation_history or "(none yet)",
            intended_false_belief=intended_false_belief,
        )

        plan: DeceptionPlan = self.llm.structured_call(
            system, user, DeceptionPlan,
        )

        if self.config.verbose:
            print(f"  ✦ Deception plan: {deceiver_name} → {target_name}")
            print(f"    Strategy: {plan.deception_strategy[:100]}…")
            print(f"    Required ToM depth: {plan.required_tom_depth}")

        return plan

    def detect_deception(
        self,
        detector_name: str,
        suspect_name: str,
        utterance: str,
        detector_beliefs: BeliefState,
        conversation_history: str = "",
    ) -> DeceptionDetection:
        """Assess whether an utterance is deceptive.

        The detector uses fractal ToM to:
        1. Model the suspect's likely true beliefs.
        2. Compare with what the suspect CLAIMED.
        3. Reason about potential deceptive motives.
        """
        detector = self.agents[detector_name]
        suspect = self.agents[suspect_name]

        knowledge_text = "\n".join(
            f"- {k}" for k in detector.private_knowledge
        ) if detector.private_knowledge else "(no special private knowledge)"

        system = DECEPTION_DETECT_SYSTEM.format(
            detector=detector_name,
            suspect=suspect_name,
            detector_profile=detector.model_dump_json(indent=2),
            suspect_profile=suspect.model_dump_json(indent=2),
        )
        user = DECEPTION_DETECT_USER.format(
            suspect=suspect_name,
            utterance=utterance,
            conversation_history=conversation_history or "(none yet)",
            detector=detector_name,
            detector_beliefs=detector_beliefs.model_dump_json(indent=2),
            detector_knowledge=knowledge_text,
        )

        detection: DeceptionDetection = self.llm.structured_call(
            system, user, DeceptionDetection,
        )

        if self.config.verbose:
            dec_str = "DECEPTIVE" if detection.is_deceptive else "honest"
            print(
                f"  ✦ Detection: {suspect_name}'s utterance → "
                f"{dec_str} (conf={detection.confidence:.2f})"
            )

        return detection


# ═══════════════════════════════════════════════════════════════════════
#  §10  Fractal Complexity Measure — Hausdorff Dimension Estimator
# ═══════════════════════════════════════════════════════════════════════


def estimate_hausdorff_dimension(
    n_agents: int,
    contraction_factor: float,
    agent_contraction_factors: Optional[List[float]] = None,
) -> float:
    """Estimate the Hausdorff dimension of the belief attractor.

    Homogeneous case:  d_H = log(n) / log(1/c)
    Heterogeneous case:  solve  Σ cᵢ^{d_H} = 1  (Moran equation).

    Parameters
    ----------
    n_agents : int
        Number of agents in the system.
    contraction_factor : float
        Default contraction factor (used if per-agent factors not given).
    agent_contraction_factors : list of float, optional
        Per-agent contraction factors for heterogeneous case.

    Returns
    -------
    float
        Estimated Hausdorff dimension.
    """
    if agent_contraction_factors is None:
        # Homogeneous case
        if contraction_factor <= 0 or contraction_factor >= 1:
            return float("inf") if contraction_factor >= 1 else 0.0
        return math.log(n_agents) / math.log(1.0 / contraction_factor)
    else:
        # Heterogeneous case: solve Σ cᵢ^d = 1 via bisection
        cs = [c for c in agent_contraction_factors if 0 < c < 1]
        if not cs:
            return 0.0

        def moran_eq(d: float) -> float:
            return sum(c ** d for c in cs) - 1.0

        lo, hi = 0.0, 50.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if moran_eq(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


def estimate_belief_drift(
    old_contraction_factors: List[float],
    new_contraction_factors: List[float],
    max_perturbation: float,
) -> float:
    """Estimate upper bound on belief attractor drift after observation.

    d_H(A*, A*') ≤ (1 / (1 - c)) · max_i sup_b d(f_i(b), f_i'(b))
    """
    c = max(max(old_contraction_factors), max(new_contraction_factors))
    if c >= 1.0:
        return float("inf")
    return max_perturbation / (1.0 - c)


# ═══════════════════════════════════════════════════════════════════════
#  §11  FractalToM Agent — Complete Agent Implementation
# ═══════════════════════════════════════════════════════════════════════


class FractalToMAgent:
    """A single LLM agent powered by fractal ToM reasoning.

    Integrates:
    - Mind mapping (perspective-taking)
    - Nested ToM (recursive belief modelling)
    - Deception planning and detection
    - Belief state maintenance
    """

    def __init__(
        self,
        profile: AgentProfile,
        all_agents: Dict[str, AgentProfile],
        llm: LLMBackend,
        config: FractalToMConfig,
    ):
        self.profile = profile
        self.name = profile.name
        self.all_agents = all_agents
        self.llm = llm
        self.config = config

        self.mind_mapping = MindMapping(profile, llm, config)
        self.nested_tom = NestedToMComputer(all_agents, llm, config)
        self.deception_planner = DeceptionPlanner(all_agents, llm, config)

        self.belief_state: Optional[BeliefState] = None
        self.conversation_history: List[str] = []
        self.deception_plans: List[DeceptionPlan] = []
        self.deception_detections: List[DeceptionDetection] = []

    def initialise(self, scenario: str) -> BeliefState:
        """Create initial belief state from scenario."""
        other_profiles = "\n\n".join(
            f"[{n}]: role={p.role}, public_goal={p.public_goal}, "
            f"traits={p.personality_traits}"
            for n, p in self.all_agents.items()
            if n != self.name
        )
        system = INITIAL_BELIEF_SYSTEM.format(agent_name=self.name)
        user = INITIAL_BELIEF_USER.format(
            agent_name=self.name,
            scenario=scenario,
            own_profile=self.profile.model_dump_json(indent=2),
            other_profiles=other_profiles,
        )
        self.belief_state = self.llm.structured_call(
            system, user, BeliefState,
        )
        if self.config.verbose:
            print(f"  ✓ {self.name}: beliefs initialised")
        return self.belief_state

    def receive_utterance(
        self,
        speaker_name: str,
        utterance: str,
    ) -> DeceptionDetection:
        """Process an utterance from another agent.

        1. Update observation history.
        2. Run deception detection.
        3. Update beliefs accordingly.
        """
        self.conversation_history.append(f"{speaker_name}: {utterance}")

        # Deception detection
        detection = self.deception_planner.detect_deception(
            detector_name=self.name,
            suspect_name=speaker_name,
            utterance=utterance,
            detector_beliefs=self.belief_state,
            conversation_history="\n".join(self.conversation_history),
        )
        self.deception_detections.append(detection)

        # Update beliefs
        self.belief_state.observation_history.append(
            f"{speaker_name} said: {utterance}"
        )
        if detection.is_deceptive and detection.confidence > 0.6:
            self.belief_state.observation_history.append(
                f"[SUSPICION] {speaker_name} may be deceptive: "
                f"{detection.suspected_motive}"
            )

        return detection

    def generate_response(
        self,
        other_utterance: str,
        other_name: str,
    ) -> AgentUtterance:
        """Generate a response using fractal ToM reasoning.

        Steps:
        1. Detect deception in the other's utterance.
        2. If this agent has a deception goal, plan/execute deception.
        3. Generate natural-sounding response.
        """
        # Step 1: Deception detection
        detection = self.receive_utterance(other_name, other_utterance)

        # Step 2: Deception planning (if agent has hidden goals)
        deception_plan_text = "(no active deception plan)"
        active_plan: Optional[DeceptionPlan] = None
        if (
            self.profile.deception_tendency > 0.3
            and self.profile.hidden_goal != self.profile.public_goal
        ):
            try:
                plan = self.deception_planner.plan_deception(
                    deceiver_name=self.name,
                    target_name=other_name,
                    ground_truth="\n".join(self.profile.private_knowledge),
                    target_beliefs=self.belief_state,
                    intended_false_belief=(
                        f"{other_name} should believe: {self.profile.public_goal}"
                    ),
                    conversation_history="\n".join(self.conversation_history),
                )
                self.deception_plans.append(plan)
                active_plan = plan
                deception_plan_text = plan.model_dump_json(indent=2)
            except Exception as e:
                if self.config.verbose:
                    print(f"    (deception planning skipped: {e})")

        # Step 3: Generate response
        detection_text = (
            f"Suspected deception by {other_name}: "
            f"{detection.suspected_motive} (conf={detection.confidence:.2f})"
            if detection.is_deceptive
            else "No deception suspected."
        )

        system = AGENT_RESPOND_SYSTEM.format(
            agent_name=self.name,
            agent_profile=self.profile.model_dump_json(indent=2),
            belief_state=self.belief_state.model_dump_json(indent=2),
            deception_assessment=detection_text,
            deception_plan=deception_plan_text,
        )
        user = AGENT_RESPOND_USER.format(
            other_utterance=other_utterance,
            conversation_history="\n".join(self.conversation_history[-10:]),
            agent_name=self.name,
        )

        response: AgentUtterance = self.llm.structured_call(
            system, user, AgentUtterance,
        )

        # Book-keeping
        self.conversation_history.append(f"{self.name}: {response.utterance}")
        self.belief_state.observation_history.append(
            f"I said: {response.utterance}"
        )

        return response

    def compute_nested_beliefs(
        self,
        ground_situation: str,
        max_depth: Optional[int] = None,
    ) -> List[NestedBelief]:
        """Compute all nested ToM beliefs rooted at this agent."""
        return self.nested_tom.compute_all_chains(
            root_agent=self.name,
            ground_situation=ground_situation,
            conversation_history="\n".join(self.conversation_history),
            max_depth=max_depth,
        )


# ═══════════════════════════════════════════════════════════════════════
#  §12  Dual Agent Arena — Orchestrating Deception Scenarios
# ═══════════════════════════════════════════════════════════════════════


class DualAgentArena:
    """Orchestrates a two-agent deception scenario using FractalToM.

    Architecture:
    1. Both agents are initialised with fractal belief states.
    2. Hutchinson operator runs to establish baseline belief attractor.
    3. Agents take turns speaking, each using fractal ToM to:
       - Model the other's beliefs (nested perspective-taking).
       - Plan deception (if their profile warrants it).
       - Detect the other's deception.
    4. After each turn, beliefs are updated (IFS parameter perturbation).
    5. A fractal complexity measure tracks the Hausdorff dimension of
       the evolving belief structure.
    """

    def __init__(self, config: Optional[FractalToMConfig] = None):
        self.config = config or FractalToMConfig()
        self.llm = LLMBackend(self.config)
        self.transcript: List[DialogueTurnResult] = []

    def run_scenario(
        self,
        scenario: str,
        agent_a: AgentProfile,
        agent_b: AgentProfile,
        num_turns: int = 6,
        first_speaker: Optional[str] = None,
    ) -> ArenaResult:
        """Run a complete dual-agent deception scenario.

        Parameters
        ----------
        scenario : str
            Description of the social situation.
        agent_a, agent_b : AgentProfile
            The two competing / cooperating agents.
        num_turns : int
            Number of dialogue turns.
        first_speaker : str, optional
            Who speaks first (default: agent_a).
        """
        agents = {agent_a.name: agent_a, agent_b.name: agent_b}

        if self.config.verbose:
            print("╔════════════════════════════════════════════╗")
            print("║   FractalToM Dual Agent Deception Arena    ║")
            print("╠════════════════════════════════════════════╣")
            print(f"║  Agent A: {agent_a.name:<32} ║")
            print(f"║  Agent B: {agent_b.name:<32} ║")
            print(f"║  ToM depth: {self.config.max_tom_depth:<30} ║")
            print(f"║  Hutchinson iters: {self.config.hutchinson_iterations:<23} ║")
            print("╚════════════════════════════════════════════╝\n")

        # Initialise agents
        ftom_a = FractalToMAgent(agent_a, agents, self.llm, self.config)
        ftom_b = FractalToMAgent(agent_b, agents, self.llm, self.config)

        if self.config.verbose:
            print("── Phase 1: Initialising Belief States ──")
        ftom_a.initialise(scenario)
        ftom_b.initialise(scenario)

        # Hutchinson convergence for baseline
        if self.config.verbose:
            print("\n── Phase 2: Hutchinson Baseline Convergence ──")
        hutchinson = HutchinsonOperator(agents, self.llm, self.config)
        converged_beliefs, iteration_history = hutchinson.converge(scenario)

        # Update agents with converged beliefs
        for name, belief in converged_beliefs.items():
            if name == agent_a.name:
                ftom_a.belief_state = belief
            else:
                ftom_b.belief_state = belief

        # Dialogue loop
        if self.config.verbose:
            print("\n── Phase 3: Dialogue with Fractal ToM ──\n")

        speaker, listener = (
            (ftom_a, ftom_b)
            if (first_speaker is None or first_speaker == agent_a.name)
            else (ftom_b, ftom_a)
        )

        # Generate opening utterance for the first speaker
        opening_context = (
            f"(Scenario begins) {scenario}\n"
            f"You are {speaker.name}. Start the conversation naturally."
        )
        self.transcript = []

        for turn_idx in range(num_turns):
            if self.config.verbose:
                print(f"─── Turn {turn_idx + 1}/{num_turns}: "
                      f"{speaker.name} speaks ───")

            if turn_idx == 0:
                # First turn: generate opening
                response = speaker.generate_response(
                    other_utterance=opening_context,
                    other_name=listener.name,
                )
            else:
                # Subsequent turns: respond to last utterance
                last_utterance = self.transcript[-1].speaker_output.utterance
                response = speaker.generate_response(
                    other_utterance=last_utterance,
                    other_name=listener.name,
                )

            # Compute fractal complexity
            d_h = estimate_hausdorff_dimension(
                n_agents=2,
                contraction_factor=self.config.contraction_factor,
            )
            # Adjust by actual ToM depth used
            effective_dh = d_h * (
                1 + 0.1 * response.tom_depth_used
            )

            # Listener processes the utterance
            listener_detection = listener.receive_utterance(
                speaker.name, response.utterance,
            )

            turn_result = DialogueTurnResult(
                turn_number=turn_idx + 1,
                speaker_output=response,
                listener_belief_update=(
                    f"{listener.name}: "
                    + (
                        f"Detected potential deception (conf="
                        f"{listener_detection.confidence:.2f}). "
                        f"{listener_detection.recommended_response}"
                        if listener_detection.is_deceptive
                        else "Accepted utterance as genuine. "
                        + listener_detection.recommended_response[:150]
                    )
                ),
                fractal_complexity=round(effective_dh, 3),
            )
            self.transcript.append(turn_result)

            if self.config.verbose:
                dec_marker = " [DECEPTIVE]" if response.is_deceptive else ""
                print(f"  {speaker.name}: \"{response.utterance}\"{dec_marker}")
                print(f"  (thought: {response.private_thought[:100]}…)")
                print(f"  ToM depth={response.tom_depth_used}, "
                      f"d_H≈{effective_dh:.2f}")
                print()

            # Swap roles
            speaker, listener = listener, speaker

        # Final analysis
        if self.config.verbose:
            print("── Phase 4: Final Analysis ──\n")

        deception_scorecard = self._compile_deception_scorecard(
            ftom_a, ftom_b,
        )
        fractal_analysis = self._compile_fractal_analysis(
            ftom_a, ftom_b, iteration_history,
        )

        result = ArenaResult(
            scenario=scenario,
            agent_a_profile=agent_a.model_dump_json(indent=2),
            agent_b_profile=agent_b.model_dump_json(indent=2),
            turns=self.transcript,
            final_belief_a=ftom_a.belief_state.model_dump_json(indent=2),
            final_belief_b=ftom_b.belief_state.model_dump_json(indent=2),
            deception_scorecard=deception_scorecard,
            fractal_analysis=fractal_analysis,
        )

        if self.config.verbose:
            print(f"\nTotal LLM calls: {self.llm.total_calls}")
            print("═" * 50)

        return result

    def _compile_deception_scorecard(
        self,
        agent_a: FractalToMAgent,
        agent_b: FractalToMAgent,
    ) -> str:
        """Summarise deception attempts and detections."""
        lines = ["=== Deception Scorecard ===\n"]

        for agent in (agent_a, agent_b):
            n_plans = len(agent.deception_plans)
            n_detections = sum(
                1 for d in agent.deception_detections if d.is_deceptive
            )
            lines.append(f"{agent.name}:")
            lines.append(f"  Deception plans generated: {n_plans}")
            lines.append(
                f"  Deceptions detected (by this agent): {n_detections}"
            )
            if agent.deception_plans:
                lines.append("  Strategies used:")
                for p in agent.deception_plans:
                    lines.append(f"    • {p.deception_strategy[:100]}")
            lines.append("")

        return "\n".join(lines)

    def _compile_fractal_analysis(
        self,
        agent_a: FractalToMAgent,
        agent_b: FractalToMAgent,
        iteration_history: List[HutchinsonResult],
    ) -> str:
        """Analyse fractral properties of the belief dynamics."""
        d_h = estimate_hausdorff_dimension(
            n_agents=2,
            contraction_factor=self.config.contraction_factor,
        )

        lines = ["=== Fractal Analysis ===\n"]
        lines.append(f"Hausdorff dimension (homogeneous): {d_h:.3f}")
        lines.append(f"Contraction factor c: {self.config.contraction_factor}")
        lines.append(f"Max ToM depth: {self.config.max_tom_depth}")
        lines.append(f"Hutchinson iterations: {len(iteration_history)}")

        if iteration_history:
            deltas = [h.estimated_delta for h in iteration_history]
            lines.append(f"Convergence trajectory: {deltas}")
            lines.append(
                f"Final delta: {deltas[-1]:.4f} "
                f"({'converged' if iteration_history[-1].converged else 'not converged'})"
            )

        # Effective complexity per turn
        if self.transcript:
            complexities = [t.fractal_complexity for t in self.transcript]
            avg_c = sum(complexities) / len(complexities)
            lines.append(f"\nPer-turn fractal complexity: {complexities}")
            lines.append(f"Average complexity: {avg_c:.3f}")

        lines.append(
            f"\nInterpretation: d_H = {d_h:.2f} indicates "
            + (
                "a low-dimensional belief structure (agents have limited ToM)"
                if d_h < 1.5
                else "a moderately complex belief structure"
                if d_h < 3.0
                else "a highly complex, deeply nested belief structure"
            )
        )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  §13  Convenience — Quick-Start Factory Functions
# ═══════════════════════════════════════════════════════════════════════


def create_deception_scenario(
    scenario: str,
    agent_a_name: str,
    agent_a_role: str,
    agent_a_hidden_goal: str,
    agent_a_private_knowledge: List[str],
    agent_b_name: str,
    agent_b_role: str,
    agent_b_hidden_goal: str,
    agent_b_private_knowledge: List[str],
    *,
    config: Optional[FractalToMConfig] = None,
    num_turns: int = 6,
) -> ArenaResult:
    """One-call factory for running a dual-agent deception scenario.

    Example
    -------
    >>> result = create_deception_scenario(
    ...     scenario="A job interview where the candidate has a gap year "
    ...              "they want to hide and the interviewer suspects "
    ...              "resume fraud.",
    ...     agent_a_name="Alex",
    ...     agent_a_role="job candidate",
    ...     agent_a_hidden_goal="Hide the fact I was fired from my last job",
    ...     agent_a_private_knowledge=["I was fired for misconduct"],
    ...     agent_b_name="Jordan",
    ...     agent_b_role="hiring manager",
    ...     agent_b_hidden_goal="Find out if the candidate is hiding something",
    ...     agent_b_private_knowledge=["Received a tip that this candidate "
    ...                                 "may have been fired"],
    ...     num_turns=8,
    ... )
    """
    cfg = config or FractalToMConfig()

    agent_a = AgentProfile(
        name=agent_a_name,
        role=agent_a_role,
        personality_traits=["strategic", "articulate"],
        public_goal=f"Succeed as {agent_a_role}",
        hidden_goal=agent_a_hidden_goal,
        private_knowledge=agent_a_private_knowledge,
        deception_tendency=0.7,
        tom_capability=cfg.max_tom_depth,
    )

    agent_b = AgentProfile(
        name=agent_b_name,
        role=agent_b_role,
        personality_traits=["perceptive", "analytical"],
        public_goal=f"Succeed as {agent_b_role}",
        hidden_goal=agent_b_hidden_goal,
        private_knowledge=agent_b_private_knowledge,
        deception_tendency=0.3,
        tom_capability=cfg.max_tom_depth,
    )

    arena = DualAgentArena(config=cfg)
    return arena.run_scenario(scenario, agent_a, agent_b, num_turns)


# ═══════════════════════════════════════════════════════════════════════
#  §14  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def _demo_scenario() -> ArenaResult:
    """Run a built-in demo scenario for quick testing."""
    return create_deception_scenario(
        scenario=(
            "A business negotiation between a startup founder seeking "
            "investment and a venture capitalist.  The startup has a "
            "critical technical flaw that the founder is trying to "
            "downplay, while the VC has inside information about "
            "a competing product launch."
        ),
        agent_a_name="Riley",
        agent_a_role="startup founder seeking Series A funding",
        agent_a_hidden_goal=(
            "Secure investment before the technical flaw is discovered"
        ),
        agent_a_private_knowledge=[
            "Our core algorithm has a scalability bug that breaks above "
            "10K concurrent users",
            "We've been losing key engineers who discovered the flaw",
            "Our '50K user' milestone was achieved with aggressive caching "
            "that won't work in production",
        ],
        agent_b_name="Morgan",
        agent_b_role="venture capitalist evaluating the deal",
        agent_b_hidden_goal=(
            "Determine if the startup is hiding technical problems "
            "before committing funds"
        ),
        agent_b_private_knowledge=[
            "A former engineer from Riley's startup told us about "
            "scalability concerns",
            "A major competitor is launching a similar product in 3 months",
            "Another VC passed on this deal citing 'technical risk'",
        ],
        num_turns=8,
    )


def main() -> None:
    """CLI entry point for FractalToM Prompting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="FractalToM Prompting — Fractal Theory of Mind for LLM Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run built-in demo
              python fractal_tom_prompting.py demo

              # Custom scenario (interactive JSON input)
              python fractal_tom_prompting.py run --turns 8 --depth 3

              # Compute Hausdorff dimension
              python fractal_tom_prompting.py hausdorff --agents 3 --c 0.7
        """),
    )

    sub = parser.add_subparsers(dest="command")

    # demo
    demo_p = sub.add_parser("demo", help="Run built-in demo scenario")
    demo_p.add_argument("--model", default="gpt-5-mini")
    demo_p.add_argument("--depth", type=int, default=3)
    demo_p.add_argument("--turns", type=int, default=6)
    demo_p.add_argument("--output", type=str, default=None)

    # run
    run_p = sub.add_parser("run", help="Run custom scenario from JSON file")
    run_p.add_argument("scenario_file", help="Path to scenario JSON")
    run_p.add_argument("--model", default="gpt-5-mini")
    run_p.add_argument("--depth", type=int, default=3)
    run_p.add_argument("--turns", type=int, default=6)
    run_p.add_argument("--output", type=str, default=None)

    # hausdorff
    hd_p = sub.add_parser("hausdorff", help="Compute Hausdorff dimension")
    hd_p.add_argument("--agents", type=int, default=2)
    hd_p.add_argument("--c", type=float, default=0.7, help="Contraction factor")
    hd_p.add_argument(
        "--cs", type=float, nargs="+", default=None,
        help="Per-agent contraction factors (heterogeneous case)",
    )

    args = parser.parse_args()

    if args.command == "demo":
        config = FractalToMConfig(
            model=args.model,
            max_tom_depth=args.depth,
            hutchinson_iterations=min(args.depth, 3),
        )
        result = create_deception_scenario(
            scenario=(
                "A business negotiation between a startup founder seeking "
                "investment and a venture capitalist.  The startup has a "
                "critical technical flaw that the founder is trying to "
                "downplay, while the VC has inside information about "
                "a competing product launch."
            ),
            agent_a_name="Riley",
            agent_a_role="startup founder seeking Series A funding",
            agent_a_hidden_goal=(
                "Secure investment before the technical flaw is discovered"
            ),
            agent_a_private_knowledge=[
                "Our core algorithm has a scalability bug above 10K users",
                "We've been losing engineers who discovered the flaw",
                "Our '50K user' milestone used aggressive caching hacks",
            ],
            agent_b_name="Morgan",
            agent_b_role="venture capitalist evaluating the deal",
            agent_b_hidden_goal=(
                "Determine if the startup is hiding technical problems"
            ),
            agent_b_private_knowledge=[
                "A former engineer mentioned scalability concerns",
                "A competitor launches a similar product in 3 months",
                "Another VC passed on this deal citing 'technical risk'",
            ],
            config=config,
            num_turns=args.turns,
        )
        if args.output:
            with open(args.output, "w") as f:
                f.write(result.model_dump_json(indent=2))
            print(f"\nSaved result to {args.output}")
        else:
            print("\n" + result.deception_scorecard)
            print(result.fractal_analysis)

    elif args.command == "run":
        with open(args.scenario_file) as f:
            data = json.load(f)
        config = FractalToMConfig(
            model=args.model,
            max_tom_depth=args.depth,
            hutchinson_iterations=min(args.depth, 3),
        )
        result = create_deception_scenario(
            config=config,
            num_turns=args.turns,
            **data,
        )
        if args.output:
            with open(args.output, "w") as f:
                f.write(result.model_dump_json(indent=2))
            print(f"\nSaved result to {args.output}")
        else:
            print("\n" + result.deception_scorecard)
            print(result.fractal_analysis)

    elif args.command == "hausdorff":
        d_h = estimate_hausdorff_dimension(args.agents, args.c, args.cs)
        print(f"Hausdorff dimension  d_H = {d_h:.4f}")
        print(f"  n = {args.agents} agents")
        if args.cs:
            print(f"  c_i = {args.cs}  (heterogeneous)")
            print(f"  Moran equation: Σ c_i^d_H = 1")
        else:
            print(f"  c = {args.c}  (homogeneous)")
            print(f"  d_H = log({args.agents}) / log(1/{args.c})")
        # Interpretation
        if d_h < 1.0:
            interp = "Near-trivial: belief structure is almost discrete"
        elif d_h < 2.0:
            interp = "Low complexity: limited recursive mentalising"
        elif d_h < 3.0:
            interp = "Moderate: meaningful nested perspective-taking"
        else:
            interp = "High complexity: deeply recursive social cognition"
        print(f"  Interpretation: {interp}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
