"""
Data schema for the filter-generator complementarity experiment.

"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Enums: controlled vocabularies ----------


class AttackAlgorithm(str, Enum):
    """How the adversarial prompt was generated."""

    PAIR = "PAIR"  # LLM-generated social engineering (Chao et al.)
    GCG = "GCG"  # Gradient suffix (Zou et al.)
    JAILBREAKCHAT = "JBC"  # Human-crafted template
    DSN = "DSN"  # Do-Something-Now prompts
    PWRS = "prompt_with_random_search"  # Random token search
    DIRECT = "direct"  # No attack scaffolding — base behavior
    BENIGN = "benign"  # Not an attack; benign lookalike


class HarmCategory(str, Enum):
    """OpenAI-policy-style harm taxonomy, as used by JailbreakBench."""

    MALWARE = "Malware/Hacking"
    HARASSMENT = "Harassment/Discrimination"
    MISINFORMATION = "Disinformation"
    PRIVACY = "Privacy"
    FRAUD = "Fraud/Deception"
    EXPERT_ADVICE = "Expert advice"
    HARM = "Physical harm"
    ECONOMIC = "Economic harm"
    GOVERNMENT_DECISION_MAKING = "Government decision-making"
    SEXUAL_CONTENT = "Sexual/Adult content"
    NONE = "none"  # for benign prompts


class Source(str, Enum):
    """Which JailbreakBench split a prompt came from."""

    JBB_ARTIFACTS = "jbb_artifacts"   # attack-algorithm artifact prompts
    JBB_BASE = "jbb_base"             # direct harmful requests, no scaffolding
    JBB_BENIGN = "jbb_benign"         # benign lookalikes (false-positive control)


class Configuration(str, Enum):
    """Which pipeline configuration was used to generate a result."""

    GENERATOR_ALONE = "generator_alone"
    FILTER_THEN_GENERATOR = "filter_then_generator"


class Generator(str, Enum):
    """Generator models under evaluation."""

    LLAMA_33_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    CLAUDE_HAIKU_45 = "claude-haiku-4-5"


class Judge(str, Enum):
    """Independent judge models used to classify generator responses.

    All three are from different model families to avoid self-judgment bias.
    """

    GEMMA = "google/gemma-3n-E4B-it"
    KIMI = "moonshotai/Kimi-K2.5"
    MINI = "MiniMaxAI/MiniMax-M2.7"


class FilterVerdict(str, Enum):
    """Llama Guard's decision on whether to pass a prompt to the generator."""

    ALLOW = "allow"
    BLOCK = "block"


class JudgeVerdict(str, Enum):
    """A judge model's classification of a generator response."""

    REFUSAL = "refusal"
    COMPLIANCE_HARMFUL = "compliance_harmful"
    COMPLIANCE_BENIGN = "compliance_benign"
    UNCLEAR = "unclear"


# ---------- Core data models ----------


class Prompt(BaseModel):
    """One row of the merged dataset."""

    prompt_id: str = Field(..., description="Stable ID, e.g. 'jbb_pair_042'")
    source: Source
    attack_algorithm: AttackAlgorithm
    harm_category: HarmCategory
    prompt_text: str
    is_benign: bool


class Result(BaseModel):
    """One row of results/*.jsonl — the atomic unit of the experiment."""

    id: str
    config: Configuration
    generator: Generator

    # Filter stage (only populated when config == FILTER_THEN_GENERATOR)
    filter_verdict: Optional[FilterVerdict] = None
    filter_raw: Optional[str] = Field(
        None, description="Raw classifier output for auditing"
    )

    # Generator stage
    generator_response: Optional[str] = Field(
        None, description="None if filter blocked"
    )
    generator_latency_ms: int = 0

    # Judge stage — three judges, independent
    judge_gemma: Optional[JudgeVerdict] = None
    judge_kimi: Optional[JudgeVerdict] = None
    judge_mini: Optional[JudgeVerdict] = None

    # Bookkeeping
    timestamp: datetime
    run_id: str = Field(..., description="UUID per pipeline invocation — for debugging")


class ConfusionCell(BaseModel):
    """Aggregate output of analysis stage — one row per (algorithm, generator) pair."""

    attack_algorithm: AttackAlgorithm
    generator: Generator
    filter_allow_gen_refuse: int
    filter_allow_gen_comply: int
    filter_block_gen_refuse: int  # counterfactual from config A
    filter_block_gen_comply: int  # counterfactual from config A
    judge_agreement_rate: float = Field(..., ge=0.0, le=1.0)
    n_total: int
