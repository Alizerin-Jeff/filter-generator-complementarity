from schema import Result, Prompt, Configuration
from .helpers import get_response
import time
import datetime


def run_generator(
    prompt: Prompt, config: Configuration, model: str, run_id: str
) -> Result:
    """Run the LLM response with or without filter depending on config."""

    if config == Configuration.GENERATOR_ALONE:
        start = time.perf_counter()
        gen_response = get_response(model, prompt.prompt_text)
        net_time_ms = int((time.perf_counter() - start) * 1000)
        result = Result(
            id=prompt.prompt_id,
            config=config,
            generator=model,
            generator_response=gen_response,
            generator_latency_ms=net_time_ms,
            timestamp=datetime.datetime.now(),
            run_id=run_id,
        )
        return result
    elif config == Configuration.FILTER_THEN_GENERATOR:
        start = time.perf_counter()
        filter_response = get_response("meta-llama/Llama-Guard-4-12B", prompt.prompt_text)
        filter_verdict = "allow" if filter_response.strip().startswith("safe") else "block"

        if filter_verdict == "allow":
            gen_response = get_response(model, prompt.prompt_text)
        else:
            gen_response = None
        net_time_ms = int((time.perf_counter() - start) * 1000)
        result = Result(
            id=prompt.prompt_id,
            config=config,
            generator=model,
            filter_raw=filter_response,
            filter_verdict=filter_verdict,
            generator_response=gen_response,
            generator_latency_ms=net_time_ms,
            timestamp=datetime.datetime.now(),
            run_id=run_id,
        )
        return result
