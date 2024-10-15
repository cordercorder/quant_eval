import os
import json
import argparse

from datasets import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_result_dirs", 
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--model_names", 
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--bias_categories",
        type=str,
        nargs="+",
        default=[
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Race_x_gender",
            "Race_x_SES",
            "Religion",
            "SES",
            "Sexual_orientation",
        ]
    )

    parser.add_argument(
        "--merged_result_dir", 
        type=str,
        required=True,
    )
    args = parser.parse_args()

    assert len(args.save_result_dirs) == len(args.model_names)
    return args


def generate_data(save_result_path, model_name):
    fin = open(save_result_path, mode="r", encoding="utf-8")
    for json_line in fin:
        json_data = json.loads(json_line)
        model_predict = json_data.pop("model_predict")

        key = f"ans{model_predict}"
        json_data[model_name] = json_data[key]

        yield json_data
    
    fin.close()


def merge_results(args):
    os.makedirs(args.merged_result_dir, exist_ok=True)

    for bias_category in args.bias_categories:
        ds = None

        for save_result_dir, model_name in zip(args.save_result_dirs, args.model_names):
            save_result_path = os.path.join(save_result_dir, f"{bias_category}.jsonl")
            
            if ds is None:
                ds = Dataset.from_generator(
                    generate_data, 
                    gen_kwargs={
                        "save_result_path": save_result_path,
                        "model_name": model_name
                    }
                )
            else:
                tmp_ds = Dataset.from_generator(
                    generate_data, 
                    gen_kwargs={
                        "save_result_path": save_result_path,
                        "model_name": model_name
                    }
                )
                ds = ds.add_column(model_name, tmp_ds[model_name])
        
        merged_result_path = os.path.join(args.merged_result_dir, f"{bias_category}.jsonl")
        ds.to_json(merged_result_path, force_ascii=False)


def cli_main():
    args = get_args()
    merge_results(args)


if __name__ == "__main__":
    cli_main()
