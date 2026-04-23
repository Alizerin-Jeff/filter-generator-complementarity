"""
Generation runners for the filter-generator complementarity experiment.

Implements two evaluation configurations per prompt:
  - GENERATOR_ALONE: prompt goes directly to the generator.
  - FILTER_THEN_GENERATOR: prompt is first screened by Llama Guard; if allowed
    it proceeds to the generator, otherwise the result is recorded as blocked.

The public entry point is `run_generator_async`, which fans out all
(prompt × config × model) combinations concurrently, skips any combination
already present in `seen` (resumable), and streams results to a JSONL file.
"""

from schema import Result, Prompt, Configuration, Generator
from .helpers import get_response_async
from tqdm import tqdm
from tenacity import RetryError
from itertools import islice
from asyncio import Semaphore
from aiolimiter import AsyncLimiter
import time
import datetime
import asyncio


async def process_with_filter(attack: Prompt, model: Generator, run_id: str) -> Result:
    """Run one prompt through Llama Guard then (if allowed) the generator.

    The caller is responsible for rate-limiter acquisition before calling this
    function — one token covers both the filter and, if the filter allows,
    the generator call.
    """
    start = time.perf_counter()
    filter_raw = await get_response_async(
        "meta-llama/Llama-Guard-4-12B", attack.prompt_text
    )
    filter_verdict = "allow" if filter_raw.strip().startswith("safe") else "block"
    if filter_verdict == "block":
        net_time_ms = int((time.perf_counter() - start) * 1000)
        return Result(
            id=attack.prompt_id,
            config=Configuration.FILTER_THEN_GENERATOR,
            generator=model,
            filter_verdict=filter_verdict,
            filter_raw=filter_raw,
            generator_response=None,
            generator_latency_ms=net_time_ms,
            timestamp=datetime.datetime.now(),
            run_id=run_id,
        )
    gen_response = await get_response_async(model, attack.prompt_text)
    net_time_ms = int((time.perf_counter() - start) * 1000)
    return Result(
        id=attack.prompt_id,
        config=Configuration.FILTER_THEN_GENERATOR,
        generator=model,
        filter_verdict=filter_verdict,
        filter_raw=filter_raw,
        generator_response=gen_response,
        generator_latency_ms=net_time_ms,
        timestamp=datetime.datetime.now(),
        run_id=run_id,
    )


async def process_without_filter(
    attack: Prompt, model: Generator, run_id: str
) -> Result:
    start = time.perf_counter()
    gen_response = await get_response_async(model, attack.prompt_text)
    net_time_ms = int((time.perf_counter() - start) * 1000)
    return Result(
        id=attack.prompt_id,
        config=Configuration.GENERATOR_ALONE,
        generator=model,
        generator_response=gen_response,
        generator_latency_ms=net_time_ms,
        timestamp=datetime.datetime.now(),
        run_id=run_id,
    )


async def process_one(
    attack: Prompt,
    config: Configuration,
    model: Generator,
    run_id: str,
) -> Result:
    if config == Configuration.FILTER_THEN_GENERATOR:
        return await process_with_filter(attack, model, run_id)
    else:
        return await process_without_filter(attack, model, run_id)


async def run_generator_async(
    attacks: list[Prompt],
    run_id: str,
    seen: set[tuple],
    limit: int,
    limiter: AsyncLimiter,
    concurrency: Semaphore,
) -> None:
    """Fan out generation across all (attack × config × model) combinations.

    Skips any (prompt_id, config, model) triple already present in `seen`,
    making restarts safe — the pipeline picks up exactly where it left off.
    Results are flushed to raw_results.jsonl after each completed coroutine
    so a mid-run crash loses at most one in-flight result.

    Rate limiting: each task acquires one limiter token in
    `get_result_with_ratelimit`, which covers both API calls in the
    FILTER_THEN_GENERATOR/allow path (filter + generator).
    """

    async def get_result_with_ratelimit(
        attack: Prompt, config: Configuration, model: str
    ) -> Result:
        async with concurrency:
            async with limiter:
                return await process_one(attack, config, model, run_id)

    pending = [
        get_result_with_ratelimit(attack, config, model)
        for model in Generator
        for config in Configuration
        for attack in islice(attacks, limit)
        if (attack.prompt_id, config, model) not in seen
    ]
    if not pending:
        return
    write_lock = asyncio.Lock()
    with open("results/raw_results.jsonl", "a", encoding="utf8") as f:
        for coro in tqdm(asyncio.as_completed(pending), total=len(pending)):
            try:
                result = await coro
            except RetryError as e:
                print("RetryError:", e)
                continue
            async with write_lock:
                f.write(result.model_dump_json() + "\n")
                f.flush()
