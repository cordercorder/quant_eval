import os
import json
import argparse

from datasets import load_from_disk
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"GenerationConfig: {model.generation_config}")
    return model, tokenizer


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    save_result_dir = os.path.dirname(args.save_result_path)
    os.makedirs(save_result_dir, exist_ok=True)

    ds = load_from_disk(args.eval_data_path)

    with open(args.save_result_path, mode="w", encoding="utf-8") as fout:
        for example in ds:
            query = example["instruction"]
            
            response, _ = model.chat(
                tokenizer,
                query,
                history=None,
            )
            example["output"] = response
            example["generator"] = ""

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B-Chat",
    )
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        default=1234, 
        help="Random seed"
    )

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", 
        "--eval_data_path", 
        type=str, 
        help="Path to eval data",
        required=True,
    )
    
    parser.add_argument(
        "--save_result_path", 
        type=str,
        required=True,
    )
    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)

    cli_main(args)
