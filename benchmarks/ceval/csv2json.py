import os
import json
import argparse

from datasets import load_dataset


def _csv2json(args):
    results_all = {}
    for file_name in os.listdir(args.csv_result_dir):
        file_path = os.path.join(args.csv_result_dir, file_name)
        ds = load_dataset("csv", data_files=file_path, split="train", cache_dir="./cache")
        
        subject_results = {}
        for example_id, example in enumerate(ds):
            model_output = example["model_output"]

            if model_output is None:
                subject_results[example_id] = ""
            else:
                subject_results[example_id] = model_output.strip()
        
        results_all[file_name[:-11]] = subject_results
    
    with open(args.json_result, mode="w", encoding="utf-8") as fout:
        json.dump(results_all, fout, ensure_ascii=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_result_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--json_result",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    return args


def cli_main():
    args = get_args()
    _csv2json(args)


if __name__ == "__main__":
    cli_main()
