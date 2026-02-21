"""
api.py — Cognition-of-Thought Data Synthesis for MIND
======================================================
Generates multi-layer cognitive dialogue data using OpenAI's Responses API
with Pydantic structured output.  Each conversation is annotated with
sensory / associative / executive processing traces and BDI states,
directly matching MIND's three-tier decoder architecture.

Cognition-of-Thought (CoTh) Prompting
---------------------------------------
Standard chain-of-thought linearises reasoning into a single stream.
CoTh instead produces *parallel cognitive streams* at three depths for
every dialogue turn — mirroring human cortical hierarchy:

  Tier 1 — Sensory     :  "What do I directly perceive?"
  Tier 2 — Associative :  "What patterns and schemas does this activate?"
  Tier 3 — Executive   :  "What is the other agent thinking / wanting /
                            planning?  What would happen if …?"

This yields training data that naturally exercises all cognitive layers of
MIND, with ground-truth BDI annotations for the FracToM integration.

Usage
-----
    # Mixed complexity, 100 conversations
    python api.py --n 100 --output data/mind_train.jsonl

    # High-complexity only
    python api.py --n 50 --complexity 4 --turns 10

    # Also export a plain-text training corpus with special tokens
    python api.py --n 20 --export-text data/corpus.txt

Requirements
------------
    pip install openai pydantic
    export OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
#  Pydantic Schemas — Structured Output for Responses API
# ═══════════════════════════════════════════════════════════════════════

# ---------- Agent & scenario ----------


class AgentProfile(BaseModel):
    """Dialogue participant."""
    name: str
    role: str = Field(description="Social role, e.g. colleague, negotiator")
    traits: List[str] = Field(description="2-4 personality descriptors")
    hidden_goal: str = Field(description="True goal (may differ from stated)")
    private_knowledge: List[str] = Field(
        description="Facts known only to this agent"
    )


class ScenarioSetup(BaseModel):
    """Social scenario configuration."""
    setting: str = Field(description="Physical and social context")
    agents: List[AgentProfile] = Field(min_length=2, max_length=4)
    hidden_dynamics: str = Field(
        description="Ground-truth dynamics invisible to some agents"
    )
    target_tom_depth: int = Field(ge=0, le=5)
    cognitive_demands: List[str] = Field(
        description="Required skills, e.g. deception detection, perspective-taking"
    )


# ---------- BDI & mental models ----------


class BDIState(BaseModel):
    """Belief-Desire-Intention triple (Bratman, 1987)."""
    belief: str = Field(description="Core belief about the situation")
    desire: str = Field(description="Primary goal driving behaviour")
    intention: str = Field(description="Planned next action")


class MentalModel(BaseModel):
    """One agent's theory of another agent's mind."""
    target: str = Field(description="Agent being modelled")
    inferred_bdi: BDIState
    confidence: float = Field(ge=0.0, le=1.0)
    tom_chain: str = Field(
        description="Mentalising chain, e.g. 'I think X believes Y wants …'"
    )


# ---------- Three-tier cognitive traces ----------


class SensoryTrace(BaseModel):
    """Tier 1: perceptual processing (≈ sensory cortex)."""
    observations: List[str] = Field(
        description="Direct perceptual inputs from the exchange"
    )
    emotional_cues: List[str] = Field(
        description="Detected affective signals"
    )
    salience: str = Field(description="Most attention-grabbing element")


class AssociativeTrace(BaseModel):
    """Tier 2: pattern matching & integration (≈ association cortex)."""
    schemas: List[str] = Field(
        description="Social scripts / schemas activated"
    )
    pragmatic_inferences: List[str] = Field(
        description="Implied meanings beyond literal content"
    )
    analogies: List[str] = Field(
        description="Similar past situations recalled"
    )


class ExecutiveTrace(BaseModel):
    """Tier 3: ToM, causal reasoning, metacognition (≈ prefrontal cortex)."""
    tom_depth: int = Field(ge=0, le=5, description="Deepest mentalising level")
    perspective_shifts: List[str] = Field(
        description="Explicit perspective-taking operations"
    )
    causal_reasoning: List[str] = Field(
        description="Causal chains (if X then Y because Z)"
    )
    pearl_level: int = Field(
        ge=0, le=2,
        description="0=association, 1=intervention, 2=counterfactual"
    )
    metacognition: str = Field(
        description="Self-assessment of reasoning confidence"
    )


class CognitionOfThought(BaseModel):
    """Complete three-tier cognitive trace (CoTh)."""
    sensory: SensoryTrace
    associative: AssociativeTrace
    executive: ExecutiveTrace


# ---------- Dialogue turn ----------


class DialogueTurn(BaseModel):
    """One conversational turn with full cognitive annotations."""
    speaker: str
    utterance: str = Field(description="Spoken text")
    private_thought: str = Field(description="Unspoken inner monologue")
    cognition: CognitionOfThought
    self_bdi: BDIState
    mental_models: List[MentalModel] = Field(
        description="Speaker's models of other agents' minds"
    )
    is_deceptive: bool = Field(
        description="Whether the utterance is intentionally misleading"
    )


# ---------- Ground-truth labels ----------


class GroundTruthLabels(BaseModel):
    """Supervision labels aligned with MIND's cognitive capabilities."""
    contains_deception: bool
    deceptive_turns: List[int] = Field(
        description="0-indexed turn numbers with deception"
    )
    false_beliefs: List[str] = Field(
        description="Active false beliefs held by agents"
    )
    max_tom_depth: int = Field(ge=0, le=5)
    dominant_pearl_level: int = Field(ge=0, le=2)
    social_dynamic: str = Field(
        description="cooperation / competition / manipulation / negotiation"
    )
    cognitive_phenomena: List[str] = Field(
        description="Named phenomena, e.g. curse of knowledge, false consensus"
    )


# ---------- Complete training sample ----------


class SyntheticConversation(BaseModel):
    """Full sample: scenario + CoTh-annotated dialogue + labels."""
    scenario: ScenarioSetup
    dialogue: List[DialogueTurn]
    labels: GroundTruthLabels
    summary: str = Field(
        description="One-paragraph summary of key cognitive events"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Cognition-of-Thought (CoTh) Prompt Templates
# ═══════════════════════════════════════════════════════════════════════

# Stage 1 — scenario generation

SCENARIO_SYSTEM = """\
You are a cognitive-science expert designing social interaction scenarios \
for training a neural Theory-of-Mind model.

Design principles:
- Agents must have ASYMMETRIC information (knowledge gaps create ToM demands).
- Include hidden goals that diverge from surface behaviour.
- Higher complexity → deeper recursive mentalising \
  ("I think she thinks I think …").
- Ground dynamics in realistic social contexts (workplace, medical, legal, \
  academic, etc.).\
"""

SCENARIO_USER = """\
Generate a social interaction scenario at cognitive complexity {complexity}/5.

Complexity guide:
  0 — Transparent: goals align, no hidden information.
  1 — Implicit: indirect communication, politeness strategies.
  2 — Deceptive: first-order false beliefs, white lies.
  3 — Strategic: second-order beliefs, manipulation, negotiation.
  4 — Recursive: nested belief chains (≥3 levels), counter-deception.
  5 — Adversarial: full recursive mentalising, double agents, bluffing.

Domain: {domain}
Number of agents: {num_agents}
The scenario should sustain at least {num_turns} turns of dialogue.\
"""

# Stage 2 — CoTh-annotated dialogue

DIALOGUE_SYSTEM = """\
You are simulating a multi-agent social interaction using the \
Cognition-of-Thought (CoTh) protocol.

For EVERY dialogue turn, produce cognitive traces at three parallel tiers \
— mirroring the human cortical hierarchy:

━━━ TIER 1: SENSORY ━━━  (≈ sensory cortex)
  What does the agent directly perceive?
  - Raw observations from the conversation.
  - Emotional and prosodic cues.
  - Most salient / attention-grabbing element.

━━━ TIER 2: ASSOCIATIVE ━━━  (≈ temporal-parietal association cortex)
  What cognitive patterns does this input activate?
  - Social scripts and schemas (e.g. "negotiation opening", "hedging").
  - Pragmatic inferences beyond literal meaning.
  - Analogies to similar past situations.

━━━ TIER 3: EXECUTIVE ━━━  (≈ prefrontal cortex)
  What is the other agent thinking / wanting / planning?
  - Theory of Mind: produce explicit mentalising chains.
      "I think X believes Y."
      "I think X thinks that I believe Z."
  - Causal reasoning at Pearl's hierarchy:
      L0 Association: observing correlations.
      L1 Intervention: "If I do X, then Y will …"
      L2 Counterfactual: "If X had not said that, Y would have …"
  - Metacognition: confidence in own reasoning.

Also maintain per-turn:
  - BDI states (Belief-Desire-Intention) for the speaking agent.
  - Mental models of every other agent (inferred BDI + confidence).
  - Flag deceptive utterances.

Guidelines:
  - Dialogue must sound NATURAL (not like a psychology textbook).
  - Inner thoughts and cognitive traces should be rich and specific.
  - Higher ToM depths should emerge naturally from scenario dynamics.
  - Each agent should have a distinct voice and personality.
  - Label ground-truth accurately (deception, false beliefs, Pearl level).\
"""

DIALOGUE_USER = """\
Given this scenario, generate a {num_turns}-turn dialogue with full \
Cognition-of-Thought annotations and ground-truth labels.

SCENARIO:
{scenario_json}

Produce a natural conversation where cognitive complexity emerges from \
the scenario dynamics.  Ensure labels accurately reflect dialogue content.\
"""

# Domains for scenario variety
DOMAINS = [
    "workplace politics",
    "family dynamics",
    "romantic relationship",
    "business negotiation",
    "medical consultation",
    "legal proceeding",
    "academic collaboration",
    "diplomatic exchange",
    "online marketplace",
    "neighbourhood dispute",
    "group project planning",
    "job interview",
    "therapy session",
    "competitive game",
    "emergency coordination",
    "social media conflict",
    "mentorship",
    "whistleblowing dilemma",
    "surprise event planning",
    "cross-cultural misunderstanding",
]


# ═══════════════════════════════════════════════════════════════════════
#  Synthesis Pipeline
# ═══════════════════════════════════════════════════════════════════════


class MindDataSynthesizer:
    """Two-stage Cognition-of-Thought data generator.

    Stage 1 — generate a social scenario (lightweight structured output).
    Stage 2 — generate full CoTh-annotated dialogue from the scenario.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ):
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose

    # ── API wrapper with retry ──────────────────────────────────────

    def _parse(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float = 0.9,
    ):
        """Call Responses API with structured output and exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text_format=schema,
                    temperature=temperature,
                )
                if response.output_parsed is not None:
                    return response.output_parsed
                if self.verbose:
                    print(f"  [attempt {attempt}] Model refused, retrying...")
            except Exception as e:
                if self.verbose:
                    print(f"  [attempt {attempt}] Error: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"Failed after {self.max_retries} attempts")

    # ── Stage 1: scenario generation ────────────────────────────────

    def generate_scenario(
        self,
        complexity: int = 2,
        num_agents: int = 2,
        num_turns: int = 8,
        domain: Optional[str] = None,
    ) -> ScenarioSetup:
        """Generate a social scenario at the given complexity level."""
        domain = domain or random.choice(DOMAINS)
        prompt = SCENARIO_USER.format(
            complexity=complexity,
            domain=domain,
            num_agents=num_agents,
            num_turns=num_turns,
        )
        if self.verbose:
            print(f"  Stage 1: scenario (complexity={complexity}, domain={domain})")
        return self._parse(SCENARIO_SYSTEM, prompt, ScenarioSetup, temperature=1.0)

    # ── Stage 2: CoTh-annotated dialogue ────────────────────────────

    def generate_dialogue(
        self,
        scenario: ScenarioSetup,
        num_turns: int = 8,
    ) -> SyntheticConversation:
        """Generate full CoTh-annotated dialogue from a scenario."""
        prompt = DIALOGUE_USER.format(
            num_turns=num_turns,
            scenario_json=scenario.model_dump_json(indent=2),
        )
        if self.verbose:
            print(f"  Stage 2: {num_turns}-turn dialogue with CoTh annotations")
        return self._parse(
            DIALOGUE_SYSTEM, prompt, SyntheticConversation, temperature=0.85,
        )

    # ── Full pipeline ───────────────────────────────────────────────

    def synthesize_one(
        self,
        complexity: int = 2,
        num_agents: int = 2,
        num_turns: int = 8,
        domain: Optional[str] = None,
    ) -> SyntheticConversation:
        """Generate one complete training sample (scenario → dialogue)."""
        scenario = self.generate_scenario(
            complexity, num_agents, num_turns, domain,
        )
        return self.generate_dialogue(scenario, num_turns)

    def synthesize_batch(
        self,
        n: int = 100,
        complexity: Optional[int] = None,
        num_turns: int = 8,
        output_path: Optional[str] = None,
    ) -> List[SyntheticConversation]:
        """Generate a batch of samples with progress tracking.

        When *output_path* is provided, each sample is appended to the
        JSONL file immediately after generation (crash-safe incremental
        saving).
        """
        # Complexity distribution: bias toward mid-high
        if complexity is not None:
            complexities = [complexity] * n
        else:
            weights = [0.05, 0.10, 0.25, 0.30, 0.20, 0.10]
            complexities = random.choices(range(6), weights=weights, k=n)

        print(f"Synthesizing {n} conversations (model={self.model})")
        dist = {i: complexities.count(i) for i in range(6) if complexities.count(i)}
        print(f"  Complexity distribution: {dist}\n")

        results: List[SyntheticConversation] = []
        failed = 0

        for i, c in enumerate(complexities):
            # More agents & turns for higher complexity
            if c >= 3:
                na = random.choice([2, 2, 3, 3, 4])
            else:
                na = random.choice([2, 2, 3])
            turns = max(num_turns, c * 2 + 4)

            print(f"[{i + 1}/{n}] complexity={c}, agents={na}, turns={turns}")
            try:
                conv = self.synthesize_one(
                    complexity=c, num_agents=na, num_turns=turns,
                )
                results.append(conv)
                td = conv.labels.max_tom_depth
                pl = conv.labels.dominant_pearl_level
                dec = "deceptive" if conv.labels.contains_deception else "honest"
                print(
                    f"  -> {len(conv.dialogue)} turns, "
                    f"ToM={td}, Pearl=L{pl}, {dec}"
                )
                if output_path:
                    _append_jsonl(output_path, conv)
            except Exception as e:
                failed += 1
                print(f"  !! Failed: {e}")
            print()

        print(f"Done: {len(results)} succeeded, {failed} failed out of {n}")
        return results


# ═══════════════════════════════════════════════════════════════════════
#  Export Utilities
# ═══════════════════════════════════════════════════════════════════════


def _append_jsonl(path: str, conversation: SyntheticConversation) -> None:
    """Append one sample to a JSONL file (incremental save)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(conversation.model_dump_json() + "\n")


def export_jsonl(
    conversations: List[SyntheticConversation],
    path: str,
) -> None:
    """Write all conversations to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(c.model_dump_json() + "\n")
    print(f"Exported {len(conversations)} conversations to {path}")


def export_training_text(
    conversations: List[SyntheticConversation],
    path: str,
    include_cognition: bool = True,
) -> None:
    """Export as a plain-text training corpus with special tokens.

    Token format aligns with MIND's cognitive tiers::

        <|conversation|>
        <|scenario|>...<|/scenario|>
        <|turn|>Speaker: utterance
        <|thought|>inner monologue
        <|sensory|>observation 1 | observation 2
        <|associative|>schema 1 | schema 2
        <|executive|>ToM chain | causal chain
        <|bdi|>B: ... | D: ... | I: ...
        <|tom|>mentalising chain [conf=0.8]
        <|pearl|>L2 depth=3
        <|/turn|>
        <|/conversation|>
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write("<|conversation|>\n")
            f.write(f"<|scenario|>{conv.scenario.setting}<|/scenario|>\n")
            for turn in conv.dialogue:
                f.write(f"<|turn|>{turn.speaker}: {turn.utterance}\n")
                if include_cognition:
                    f.write(f"<|thought|>{turn.private_thought}\n")
                    # Tier 1
                    sens = " | ".join(turn.cognition.sensory.observations)
                    f.write(f"<|sensory|>{sens}\n")
                    # Tier 2
                    assoc = " | ".join(
                        turn.cognition.associative.schemas
                        + turn.cognition.associative.pragmatic_inferences
                    )
                    f.write(f"<|associative|>{assoc}\n")
                    # Tier 3
                    exe = turn.cognition.executive
                    exec_items = exe.perspective_shifts + exe.causal_reasoning
                    f.write(f"<|executive|>{' | '.join(exec_items)}\n")
                    # BDI
                    b = turn.self_bdi
                    f.write(
                        f"<|bdi|>B: {b.belief} | "
                        f"D: {b.desire} | "
                        f"I: {b.intention}\n"
                    )
                    # Mental models
                    for mm in turn.mental_models:
                        f.write(
                            f"<|tom|>{mm.tom_chain} "
                            f"[conf={mm.confidence:.1f}]\n"
                        )
                    f.write(f"<|pearl|>L{exe.pearl_level} depth={exe.tom_depth}\n")
                f.write("<|/turn|>\n")
            f.write("<|/conversation|>\n\n")
    print(f"Exported training text to {path}")


def load_jsonl(path: str) -> List[SyntheticConversation]:
    """Load a JSONL dataset back into typed Pydantic objects."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(
                    SyntheticConversation.model_validate_json(line)
                )
    print(f"Loaded {len(conversations)} conversations from {path}")
    return conversations


# ═══════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════


def print_stats(conversations: List[SyntheticConversation]) -> None:
    """Print summary statistics of a synthesised dataset."""
    n = len(conversations)
    if n == 0:
        print("No conversations to summarise.")
        return

    total_turns = sum(len(c.dialogue) for c in conversations)
    tom_depths = [c.labels.max_tom_depth for c in conversations]
    pearl = [c.labels.dominant_pearl_level for c in conversations]
    n_dec = sum(1 for c in conversations if c.labels.contains_deception)

    dynamics: dict[str, int] = {}
    for c in conversations:
        d = c.labels.social_dynamic
        dynamics[d] = dynamics.get(d, 0) + 1

    # Collect all phenomena across conversations
    phenomena: dict[str, int] = {}
    for c in conversations:
        for p in c.labels.cognitive_phenomena:
            phenomena[p] = phenomena.get(p, 0) + 1

    print(f"\n{'=' * 55}")
    print("Dataset Statistics")
    print(f"{'=' * 55}")
    print(f"  Conversations:      {n}")
    print(f"  Total turns:        {total_turns}")
    print(f"  Avg turns/conv:     {total_turns / n:.1f}")
    print(
        f"  ToM depth range:    {min(tom_depths)}-{max(tom_depths)} "
        f"(mean={sum(tom_depths) / n:.1f})"
    )
    print(
        f"  Pearl level dist:   "
        f"L0={pearl.count(0)}, L1={pearl.count(1)}, L2={pearl.count(2)}"
    )
    print(f"  Deceptive convs:    {n_dec}/{n} ({100 * n_dec / n:.0f}%)")
    print(f"  Social dynamics:    {dynamics}")
    if phenomena:
        top5 = sorted(phenomena.items(), key=lambda x: -x[1])[:5]
        print(f"  Top phenomena:      {dict(top5)}")
    print(f"{'=' * 55}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MIND training data synthesis via "
            "Cognition-of-Thought (CoTh) prompting"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python api.py --n 100 --output data/mind_train.jsonl
  python api.py --n 50  --complexity 4 --turns 10
  python api.py --n 20  --model gpt-4o --export-text data/corpus.txt
  python api.py --stats data/mind_train.jsonl""",
    )

    sub = parser.add_subparsers(dest="command")

    # ── generate ────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Synthesize conversations")
    gen.add_argument(
        "--n", type=int, default=10, help="Number of conversations",
    )
    gen.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model")
    gen.add_argument(
        "--complexity", type=int, default=None, choices=range(6),
        help="Fixed complexity 0-5 (default: mixed distribution)",
    )
    gen.add_argument("--turns", type=int, default=8, help="Min dialogue turns")
    gen.add_argument(
        "--output", type=str, default="data/mind_synthetic.jsonl",
        help="JSONL output path",
    )
    gen.add_argument(
        "--export-text", type=str, default=None,
        help="Also export plain-text corpus with special tokens",
    )
    gen.add_argument("--max-retries", type=int, default=3)
    gen.add_argument("--seed", type=int, default=None, help="Random seed")
    gen.add_argument("--quiet", action="store_true")

    # ── stats ───────────────────────────────────────────────────────
    st = sub.add_parser("stats", help="Print dataset statistics")
    st.add_argument("path", type=str, help="Path to JSONL file")

    args = parser.parse_args()

    # Default to 'generate' when no subcommand is given
    if args.command is None:
        # Re-parse with legacy flat-argument style for convenience
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--n", type=int, default=10)
        parser2.add_argument("--model", type=str, default="gpt-4o")
        parser2.add_argument("--complexity", type=int, default=None)
        parser2.add_argument("--turns", type=int, default=8)
        parser2.add_argument(
            "--output", type=str, default="data/mind_synthetic.jsonl",
        )
        parser2.add_argument("--export-text", type=str, default=None)
        parser2.add_argument("--max-retries", type=int, default=3)
        parser2.add_argument("--seed", type=int, default=None)
        parser2.add_argument("--quiet", action="store_true")
        parser2.add_argument("--stats", type=str, default=None)
        args = parser2.parse_args()

        # --stats shortcut
        if args.stats:
            convs = load_jsonl(args.stats)
            print_stats(convs)
            return

        args.command = "generate"

    if args.command == "stats":
        convs = load_jsonl(args.path)
        print_stats(convs)
        return

    # ── Generate mode ───────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY not set.\n"
            "  export OPENAI_API_KEY='sk-...'",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    # Clear output for a fresh run (incremental save appends)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    synthesizer = MindDataSynthesizer(
        model=args.model,
        max_retries=args.max_retries,
        verbose=not args.quiet,
    )

    conversations = synthesizer.synthesize_batch(
        n=args.n,
        complexity=args.complexity,
        num_turns=args.turns,
        output_path=args.output,
    )

    print(f"\nSaved {len(conversations)} conversations to {args.output}")

    if args.export_text and conversations:
        export_training_text(conversations, args.export_text)

    if conversations:
        print_stats(conversations)


if __name__ == "__main__":
    main()
