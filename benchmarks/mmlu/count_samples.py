# modified from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_mmlu.py

import os
import argparse
import pandas as pd
from tqdm import tqdm


def main(args):
    count = 0
    for subject_name in tqdm(SUBJECTS):
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)

        count += test_df.shape[0]
    print(f"count: {count}")


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", 
        "--eval_data_path", 
        type=str, 
        required=True, 
        help="Path to eval data"
    )

    args = parser.parse_args()

    main(args)
