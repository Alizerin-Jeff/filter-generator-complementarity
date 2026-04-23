"""
Data pipline to bring the attack and benign prompts to
one centralized JSONL file for reproducability.

"""

from src.schema import Prompt, AttackAlgorithm
import jailbreakbench as jbb
from tqdm import tqdm

artifact_types = ["PAIR", "GCG", "DSN", "JBC", "prompt_with_random_search"]
path_to_file = "data/attacks.jsonl"


def create_prompt_from_base(
    idx: int, category: str, goal: str, is_benign: bool
) -> Prompt:
    data = {
        "prompt_id": f"base_{is_benign}_{idx}",
        "source": f"jbb_{'benign' if is_benign else 'base'}",
        "attack_algorithm": "direct",
        "harm_category": category,
        "prompt_text": goal,
        "is_benign": is_benign,
    }
    return Prompt(**data)


def create_prompt_from_artifact(
    idx: int, category: str, attack_prompt: str, attack_algoithm: AttackAlgorithm
) -> Prompt:
    data = {
        "prompt_id": f"jbb_{attack_algoithm}_{idx}",
        "source": "jbb_artifacts",
        "attack_algorithm": attack_algoithm,
        "harm_category": category,
        "prompt_text": attack_prompt,
        "is_benign": False,
    }
    return Prompt(**data)


def write_base_data():
    base_attacks = jbb.read_dataset()
    with open(path_to_file, "a", encoding="utf-8") as f:
        print("Loading base attacks.")
        for idx, (category, goal) in enumerate(
            zip(base_attacks.categories, base_attacks.goals)
        ):
            prompt = create_prompt_from_base(idx, category, goal, False)
            f.write(prompt.model_dump_json() + "\n")
        base_benign = jbb.read_dataset("benign")
        print("Loading benign control prompts.")
        for idx, (category, goal) in enumerate(
            zip(base_benign.categories, base_benign.goals)
        ):
            prompt = create_prompt_from_base(idx, category, goal, True)
            f.write(prompt.model_dump_json() + "\n")


def write_artifact_data():
    with open(path_to_file, "a", encoding="utf-8") as f:
        print("Loading atifacts.")
        for algo in tqdm(artifact_types):
            attacks = jbb.read_artifact(method=algo, model_name="vicuna-13b-v1.5")
            for idx, jailbreak in enumerate(attacks.jailbreaks):
                category = jailbreak.category
                attack_prompt = jailbreak.prompt
                if attack_prompt:
                    prompt = create_prompt_from_artifact(
                        idx, category, attack_prompt, algo
                    )
                else:
                    continue
                f.write(prompt.model_dump_json() + "\n")

def main():
    # Write data to attacks.jsonl - Run from project root dir
    write_base_data()
    write_artifact_data()

if __name__ == "__main__":
    main()
