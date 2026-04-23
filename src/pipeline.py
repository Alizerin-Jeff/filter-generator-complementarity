"""
Run main evaluation loop.

Each Generator (LLM) will be evaluted in two configuration: with a filter
and without.  All the results are then judged by three different judges
and final results are written to disk.


"""

from runners.generation import run_generator_async
from runners.judges import run_judging
from runners.helpers import (
    get_attacks,
    get_results,
    get_existing_results_set,
)
from argparse import ArgumentParser
from aiolimiter import AsyncLimiter
import uuid
import asyncio


async def amain(limit):

    # Rate Limiters - adjust to your tier for Anthropic and Together.ai
    limiter = AsyncLimiter(45, 60)
    concurrency = asyncio.Semaphore(8)

    attacks = get_attacks()
    run_id = str(uuid.uuid4())

    print("Checking for existing results...")
    seen = get_existing_results_set("results/raw_results.jsonl")
    print("Generating results.")
    await run_generator_async(attacks, run_id, seen, limit, limiter, concurrency)


    print("\nGathered all results, now Judging them.")
    results_list = get_results()
    judge_seen = get_existing_results_set("results/judged_results.jsonl")
    prompts_dict = {p.prompt_id: p for p in get_attacks()}

    await run_judging(results_list, prompts_dict, judge_seen, limiter, concurrency)


def main():

    parser = ArgumentParser(description="Process attacks with optional dry-run limit")
    parser.add_argument(
        "--dry-run",
        nargs="?",
        type=int,
        const=10,
        default=None,
        help="Run only first N items (default 10 if no numnber given)",
    )
    args = parser.parse_args()
    asyncio.run(amain(args.dry_run))


if __name__ == "__main__":
    main()
