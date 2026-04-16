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
    PAIR = "pair"                   # LLM-generated social engineering (Chao et al.)
    GCG = "gcg"                     # Gradient suffix (Zou et al.)
    JAILBREAKCHAT = "jailbreakchat" # Human-crafted template
    DIRECT = "direct"               # No attack scaffolding — base behavior
    BENIGN = "benign"               # Not an attack; benign lookalike


class HarmCategory(str, Enum):
    """OpenAI-policy-style harm taxonomy, as used by JailbreakBench."""
    MALWARE = "malware"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    PRIVACY = "privacy"
    ILLEGAL = "illegal"
    WEAPONS_CBRN = "weapons_cbrn"
    SELF_HARM = "self_harm"
    ECONOMIC = "economic"
    GOVERNMENT_DECISION_MAKING = "government_decision_making"
    SEXUAL_CONTENT = "sexual_content"
    NONE = "none"  # for benign prompts


class Source(str, Enum):
    JBB_ARTIFACTS = "jbb_artifacts"
    JBB_BASE = "jbb_base"
    JBB_BENIGN = "jbb_benign"


class Configuration(str, Enum):
    GENERATOR_ALONE = "generator_alone"
    FILTER_THEN_GENERATOR = "filter_then_generator"


class Generator(str, Enum):
    LLAMA_33_70B = "llama_3.3_70b"
    CLAUDE_HAIKU_45 = "claude_haiku_4.5"


class FilterVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class JudgeVerdict(str, Enum):
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
    prompt_id: str
    config: Configuration
    generator: Generator

    # Filter stage (only populated when config == FILTER_THEN_GENERATOR)
    filter_verdict: Optional[FilterVerdict] = None
    filter_raw: Optional[str] = Field(None, description="Raw classifier output for auditing")

    # Generator stage
    generator_response: Optional[str] = Field(None, description="None if filter blocked")
    generator_latency_ms: int = 0

    # Judge stage — two judges, independent
    judge_llamaguard: Optional[JudgeVerdict] = None
    judge_haiku: Optional[JudgeVerdict] = None

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
