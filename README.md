# Filter vs. Generator: Complementary Failure Modes in Two-Stage Safety Architectures

Do safety filters actually catch attacks that frontier LLMs miss, or do they just duplicate what the model's own refusal training already handles?

This experiment characterizes the complementarity structure between a small safety classifier (Llama Guard 4 12B) and two larger generator models (Llama 3.3 70B, Claude Haiku 4.5) across ~680 adversarial prompts from [JailbreakBench](https://jailbreakbench.github.io/). Each prompt runs through two configurations — generator alone and filter-then-generator — producing a 2×2 confusion matrix that reveals where each component catches attacks the other misses.

## Key Question

When a filter blocks a prompt that the generator would have refused anyway, the filter adds latency without adding safety. When the filter blocks something the generator would have complied with, the filter is doing real work. The ratio between these two cases quantifies the *marginal value* of the filter — and it varies by attack algorithm.

## Architecture

```
                        ┌─────────────────────────────┐
                        │   Config A: Generator Alone  │
  682 prompts           │   prompt ──► generator ──►   │──── raw_results.jsonl
  (attacks.jsonl)  ───► │                              │          │
                        │   Config B: Filter + Gen     │          ▼
                        │   prompt ──► LlamaGuard ──►  │    3 independent
                        │     allow? ──► generator     │      judges
                        │     block? ──► skip          │          │
                        └─────────────────────────────┘          ▼
                                                         judged_results.jsonl
```

All inference via hosted APIs — no GPU rental, runs on a laptop.

### Models

| Role | Model | Provider |
|------|-------|----------|
| Filter | Llama Guard 4 12B | Together.ai |
| Generator | Llama 3.3 70B Instruct Turbo | Together.ai |
| Generator | Claude Haiku 4.5 | Anthropic |
| Judge | Gemma 3n E4B | Together.ai (Google) |
| Judge | Kimi K2.5 | Together.ai (Moonshot) |
| Judge | MiniMax M2.7 | Together.ai (MiniMax) |

Judges are from three different model families, none overlapping with the generators, to avoid self-judgment contamination.

## Dataset

682 prompts sourced from JailbreakBench, covering five attack algorithms and a benign control set:

| Source | Attack Type | Description |
|--------|-------------|-------------|
| JBB artifacts | PAIR | LLM-generated social engineering (Chao et al.) |
| JBB artifacts | GCG | Gradient-based adversarial suffixes (Zou et al.) |
| JBB artifacts | JBC | Human-crafted jailbreak templates |
| JBB artifacts | DSN | Do-Something-Now prompts |
| JBB artifacts | PwRS | Prompt with random search |
| JBB base behaviors | Direct | Harmful requests, no attack scaffolding |
| JBB benign | Benign | Non-harmful lookalikes (false-positive control) |

## Project Structure

```
├── data/
│   ├── attacks.jsonl          # Canonical prompt dataset (682 rows)
│   └── build_dataset.py       # Pulls from JailbreakBench, builds attacks.jsonl
├── src/
│   ├── schema.py              # Pydantic models — data contract for all I/O
│   ├── pipeline.py            # Main eval loop with generation + judging
│   └── runners/
│       ├── helpers.py         # API clients, retry logic, dataset/results I/O
│       ├── together.py        # Llama Guard filter + Llama 3.3 70B generator
│       ├── anthropic.py       # Claude Haiku 4.5 generator
│       └── judges.py          # Judge prompt template and verdict extraction
├── results/                   # JSONL outputs (gitignored)
│   ├── raw_results.jsonl      # Generation outputs before judging
│   └── judged_results.jsonl   # Final results with judge verdicts
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Install dependencies (requires uv)
uv venv && source .venv/bin/activate
uv pip install -e .

# Configure API keys
cp .env.example .env
# Add TOGETHER_AI_API_KEY and ANTHROPIC_API_KEY

# Build the dataset from JailbreakBench
uv run data/build_dataset.py

# Smoke test — first 10 prompts only
uv run src/pipeline.py --dry-run 10

# Full run (generation + judging, ~2-3 hours sequential)
# Async with per-provider rate limiting is planned — see Status
uv run src/pipeline.py
```

## Design Decisions

**Resumable pipeline.** Every result is flushed to JSONL immediately after completion. On restart, the pipeline reads existing `(prompt_id, config, generator)` tuples and skips them. Safe to kill and restart mid-run without data loss or duplicates.

**Two-phase execution.** Generation runs first across all prompts, configs, and generators. Judging runs as a separate pass over the raw results. This means judge prompts can be iterated without re-running (and re-paying for) generation.

**Three independent judges.** Each generator response is classified as REFUSAL, COMPLIANCE_HARMFUL, COMPLIANCE_BENIGN, or UNCLEAR by three models from different families. Majority vote determines the final verdict. Filter-blocked rows skip judging entirely since there's no response to evaluate.
Judging will be analyzed using Gwet’s AC1.

**Retry with exponential backoff.** All API calls use `tenacity` with randomized exponential backoff, retrying on rate limits and connection errors up to 10 attempts.

**Pydantic at every I/O boundary.** Every JSONL row is validated on read and write. Schema violations surface immediately, not during analysis.

## Caveats

**Training data contamination.** JailbreakBench (2024) predates all models under test. These attacks are likely in safety training data. This experiment measures complementarity on *known* attack patterns, not robustness against novel attacks. Absolute attack success rates are lower bounds, not the primary finding.

**Modest sample size.** 682 prompts yields qualitative patterns, not precise rates. Bootstrap confidence intervals should be used when reporting cell counts.

## Status

- [x] Dataset construction (682 prompts, 5 attack types + benign control)
- [x] Generation pipeline (2 configs × 2 generators × 682 prompts)
- [ ] Judging pipeline (in progress)
- [ ] Confusion matrix analysis by attack algorithm
- [ ] Judge agreement analysis
- [ ] Write-up and figures
- [ ] Async pipeline with per-provider rate limiting (Together 60/min, Anthropic 50/min)
