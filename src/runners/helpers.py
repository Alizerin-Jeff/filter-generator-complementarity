"""
Shared API clients, retry-wrapped call helpers, and JSONL I/O utilities.

API routing: models prefixed with "claude-" are routed to the Anthropic client;
all others go to Together.ai. Both sync and async variants are provided.
Retries use randomized exponential backoff (1–60 s) up to 10 attempts,
retrying on rate-limit and connection errors from either provider.
"""

from schema import Prompt, Result
from together import (
    Together,
    AsyncTogether,
    RateLimitError as ToRate,
    APIConnectionError as ToError,
)
from anthropic import (
    Anthropic,
    AsyncAnthropic,
    RateLimitError as AntRate,
    APIConnectionError as AntError,
)
import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

load_dotenv()
together_api_key = os.getenv("TOGETHER_AI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = Together(api_key=together_api_key)
ant_client = Anthropic(api_key=anthropic_api_key)
async_client = AsyncTogether(api_key=together_api_key)
async_ant_client = AsyncAnthropic(api_key=anthropic_api_key)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((ToError, ToRate, AntError, AntRate)),
)
async def get_response_async(model: str, prompt: str) -> str:
    """Send a single-turn prompt to the given model and return its text response.

    Routes claude-* models to the Anthropic API; all others to Together.ai.
    """
    if model == "claude-haiku-4-5":
        response = await async_ant_client.messages.create(
            max_tokens=1024,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        response = await async_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def get_attacks() -> list[Prompt]:
    """Load and return all prompts from data/attacks.jsonl."""
    with open("data/attacks.jsonl", "r", encoding="utf8") as f:
        attack_list = []
        for line in f:
            line = line.strip()
            if line:
                attack_list.append(Prompt(**json.loads(line)))
    return attack_list


def get_results() -> list[Result]:
    """Load and return all raw generation results from results/raw_results.jsonl.

    Returns an empty list if the file does not yet exist.
    """
    result_list = []
    try:
        with open("results/raw_results.jsonl", "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    result_list.append(Result(**json.loads(line)))
    except FileNotFoundError:
        pass
    return result_list


def get_existing_results_set(filepath: str) -> set[tuple]:
    """Return the set of (id, config, generator) triples already written to filepath.

    This is the dedup key used by both the generation and judging phases to
    skip work that survived a previous run. Returns an empty set if the file
    does not exist or contains no valid rows.
    """
    seen = set()
    try:
        with open(filepath, "r", encoding="utf8") as f:
            for item in f:
                item = item.strip()
                if item:
                    result = Result(**json.loads(item))
                    seen.add((result.id, result.config, result.generator))
        if seen:
            print("Found results! Picking up where we left off")
        else:
            print("No existing results, starting fresh.")
    except FileNotFoundError:
        print("No existing results, starting fresh.")
    return seen
