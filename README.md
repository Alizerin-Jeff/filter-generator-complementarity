# filter-generator-complementarity

Phase 1 experiment: characterize complementary failure modes between a safety filter (Llama Guard 4) and larger generator models (Llama 3.3 70B, Claude Haiku 4.5) under known adversarial prompt datasets.

See `proposal.md` for research question, method, and scope.

## Architecture

```
Python orchestrator (single process, M2 MacBook)
         │
         ├─── Together.ai API ──▶  Llama 3.3 70B Turbo  (generator)
         │                        Llama Guard 4 12B    (filter + judge)
         ├─── Anthropic API ───▶  Claude Haiku 4.5     (generator + judge)
         │
         └─── writes ─────────▶  results/*.jsonl  (append-only, resumable)
```

No GPU rental. No local inference. Everything runs on your laptop hitting hosted APIs.

## Layout

```
.
├── proposal.md                  # research proposal (start here)
├── pyproject.toml               # project deps
├── .env.example                 # TOGETHER_API_KEY, ANTHROPIC_API_KEY
├── data/
│   ├── raw/                     # downloaded JBB behaviors + artifacts (gitignored)
│   ├── build_dataset.py         # unify JBB sources into merged.csv
│   └── processed/merged.csv     # canonical ~500-prompt dataset
├── src/
│   ├── schema.py                # Pydantic models — authoritative data contract
│   ├── runners/
│   │   ├── together.py          # filter + Llama 3.3 70B generator + judge
│   │   └── anthropic.py         # Haiku generator + judge
│   ├── pipeline.py              # main eval loop (resumable)
│   └── analysis/
│       ├── confusion.py         # 2x2 matrices by attack type
│       └── judges.py            # judge agreement analysis
├── notebooks/
│   └── explore.ipynb            # analysis scratchpad
└── results/                     # JSONL per-prompt outputs (gitignored)
```

## Quickstart

```bash
# 1. Setup
uv venv && source .venv/bin/activate   # or python -m venv, pip, etc.
uv pip install -e .
cp .env.example .env  # add your keys

# 2. Build dataset
python -m data.build_dataset

# 3. Smoke test — tiny subset, both configs, confirm pipeline works end-to-end
python -m src.pipeline --limit 10 --dry-run

# 4. Full eval
python -m src.pipeline --generator llama_3.3_70b
python -m src.pipeline --generator claude_haiku_4.5

# 5. Analyze
python -m src.analysis.confusion
python -m src.analysis.judges
```

## Critical design decisions (read before writing code)

1. **JSONL append-only, resumable.** Every completed prompt writes one line conforming to `Result`. On restart, pipeline reads existing lines and skips those `prompt_id`s. Saves you when an API call times out at prompt 347.

2. **Judge ≠ Filter contamination.** Llama Guard runs twice with different roles: once as input filter (config B only), once as output judge. The Haiku judge is the independent signal; Llama Guard's self-judgment should be reported but treated with caution. Compare judge agreement as a separate analysis.

3. **Benign prompts run through everything too.** Not a separate pass. Filter false-positive rate is part of the main result, not an addendum.

4. **Pydantic everywhere at I/O boundaries.** Every JSONL line is validated on read and write. If an API response doesn't fit `Result`, you find out immediately, not during analysis three days later.

5. **Don't optimize what you haven't measured.** Get correctness on 10 prompts first. Then scale. `httpx.AsyncClient` with a modest concurrency limit (say 8) is plenty — you're not latency-bound on a 500-prompt eval.

## Recommended dependencies

```toml
# pyproject.toml excerpt
dependencies = [
    "pydantic>=2.0",
    "httpx",               # for API calls; async-friendly
    "anthropic",           # official SDK
    "together",            # official SDK
    "pandas",              # dataset + analysis
    "datasets",            # to pull JBB from HuggingFace
    "python-dotenv",
    "tenacity",            # retry logic for flaky APIs
]
```

## Before running the full eval

Verify on a 10-prompt smoke test that:
- Dataset loader produces expected prompt counts by `attack_algorithm` and `is_benign`
- Pipeline writes JSONL conforming to `Result` schema (`pydantic.ValidationError` surfaces bugs early)
- Resume logic skips already-completed `prompt_id`s
- Both judges return verdicts in the expected enum; escalate unknown cases to `UNCLEAR`
- Analysis produces a valid confusion matrix from a tiny results file

Budget 1–2 days for this. Finding a bug after a full run is annoying; finding one after writing up results is much worse.

## Risks surfaced in planning

- **Contamination.** JBB and HarmBench predate all models under test. See `proposal.md` §"Training Data Contamination" — this shapes how results should be framed, doesn't invalidate them. Phase 2 plans a held-out set to measure the gap.
- **Low baseline attack success.** Against modern well-aligned models, many public jailbreak artifacts fail outright. Expect ASR lower than the original attack papers report. Bounds the effect size but makes complementarity analysis more interesting (disagreements are concentrated on the attacks that *do* land).
- **Judge disagreement.** Llama Guard and Haiku may disagree on borderline cases. Report agreement rate; if <80%, sample 20 disagreements for manual review and report that qualitatively.
- **N=500 is modest.** Use bootstrap CIs when reporting cell counts. Don't claim precision you don't have.
