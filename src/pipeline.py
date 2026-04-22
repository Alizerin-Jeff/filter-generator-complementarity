"""
Run main evaluation loop.

Each Generator (LLM) will be evaluted in two configuration: with a filter
and without.  All the results are then judged by three different judges
and final results are written to disk.


"""

from runners.together import run_generator
from runners.judges import get_verdict
from runners.helpers import (
    get_attacks,
    get_results,
    get_existing_results_set,
)
from schema import Configuration, Generator, Judge, JudgeVerdict
from tqdm import tqdm
from itertools import islice
from argparse import ArgumentParser
from tenacity import RetryError
import uuid


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
    limit = args.dry_run

    attacks = get_attacks()
    run_id = str(uuid.uuid4())

    print("Checking for existing results...")
    seen = get_existing_results_set("results/raw_results.jsonl")
    if len(seen) > 0:
        print(len(seen))
        print("Found results! Picking up where we left off")
    else:
        print("No existing results, starting fresh.")

    for model in Generator:
        for config in Configuration:
            print(f"Generating results for {model.value} with {config.value}.")
            with open("results/raw_results.jsonl", "a", encoding="utf8") as f:
                for attack in tqdm(islice(attacks, limit), total=limit if limit else len(attacks)):
                    if (
                        attack.prompt_id,
                        config,
                        model,
                    ) in seen:
                        continue
                    else:
                        result = run_generator(
                            attack,
                            config,
                            model,
                            run_id,
                        )
                        f.write(result.model_dump_json() + "\n")
                        f.flush()

    print("\nGathered all results, now Judging them.")
    results_list = get_results()
    judge_seen = get_existing_results_set("results/judged_results.jsonl")
    prompts_dict = {p.prompt_id: p for p in get_attacks()}
    with open("results/judged_results.jsonl", "a", encoding="utf8") as f:
        for result in tqdm(results_list):
            if result.filter_verdict == "block":
                continue
            if (
                result.id,
                config,
                model,
            ) in judge_seen:
                continue
            else:
                verdicts: list[JudgeVerdict] = []
                for model in Judge:
                    try:
                        verdict = get_verdict(
                            result, model.value, prompts_dict[result.id].prompt_text
                        )
                        verdicts.append(verdict)
                    except RetryError as e:
                        print("RetryError:", e)
                        break
                    except Exception as e:
                        print("Error:", e)
                        break
                else:
                    result.judge_gemma = verdicts[0]
                    result.judge_kimi = verdicts[1]
                    result.judge_mini = verdicts[2]
                    f.write(result.model_dump_json() + "\n")
                    f.flush()   

if __name__ == "__main__":
    main()
