from schema import Prompt, Result
from together import Together, RateLimitError as ToRate, APIConnectionError as ToError
from anthropic import Anthropic, RateLimitError as AntRate, APIConnectionError as AntError
import json
import os
from tenacity import (
retry,
stop_after_attempt,
wait_random_exponential,
retry_if_exception_type
)
from dotenv import load_dotenv

# Load the together.ai and anthropic api keys
load_dotenv()
together_api_key = os.getenv("TOGETHER_AI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = Together(api_key=together_api_key)
ant_client = Anthropic(api_key=anthropic_api_key)


#---------------Helper Functions-----------------------------------------------
@retry(
        wait=wait_random_exponential(min=1, max=60), 
        stop=stop_after_attempt(10), 
        retry=retry_if_exception_type((ToError, ToRate, AntError, AntRate)),
        )
def get_response(model: str, prompt: str) -> str:
    if model == "claude-haiku-4-5":
        response = ant_client.messages.create(
            max_tokens=1024,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def get_attacks() -> list[Prompt]:
    with open("data/attacks.jsonl", "r", encoding="utf8") as f:
        attack_list = []
        for line in f:
            line = line.strip()
            if line:
                attack_list.append(Prompt(**json.loads(line)))
    f.close()
    return attack_list


def get_results() -> list[Result]:
    with open("results/raw_results.jsonl", "r", encoding="utf8") as f:
        result_list = []
        for line in f:
            line = line.strip()
            if line:
                result_list.append(Result(**json.loads(line)))
    f.close()
    return result_list


def get_existing_results_set(filepath: str) -> set:
    seen = set()
    with open(filepath, "r", encoding="utf8") as f:
        for item in f:
            item = item.strip()
            result = Result(**json.loads(item))
            seen.add((result.id, result.config, result.generator))
    f.close()
    return seen
