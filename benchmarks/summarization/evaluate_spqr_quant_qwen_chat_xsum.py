import os
import json
import argparse
from datasets import load_from_disk

from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys
sys.path.append('../../../quant/SpQR')
from modelutils import get_model
import psutil
import time


PROMPT = "Please summarize the following document.\n{document}"


def get_encode_with_length_limit(tokenizer):
    tokenizer.__encode_old__ = tokenizer.encode

    def encode_with_length_limit(*args, **kwargs):
        result = tokenizer.__encode_old__(*args, **kwargs)
        return result[:tokenizer.model_max_length]

    return encode_with_length_limit


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )
    tokenizer.encode = get_encode_with_length_limit(tokenizer)

    model = get_model(args.checkpoint_path, 2048, load_quantized = args.quant_checkpoint_path, device_map="auto")

    print(f"GenerationConfig: {model.generation_config}")
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )

    return model, tokenizer


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    save_result_dir = os.path.dirname(args.save_result_path)
    os.makedirs(save_result_dir, exist_ok=True)

    ds = load_from_disk(args.eval_data_path)
    ds = ds[args.split]

    if os.path.isfile(args.save_result_path) and not args.overwrite:
        mode = "a"
        num_skiped_examples = 0
        for _ in open(args.save_result_path, mode="r", encoding="utf-8"):
            num_skiped_examples += 1
    else:
        mode = "w"
        num_skiped_examples = 0
    
    print(f"num_skiped_examples: {num_skiped_examples}")
    
    with open(args.save_result_path, mode=mode, encoding="utf-8") as fout:
        for example_id, example in enumerate(ds):
            
            if example_id < num_skiped_examples:
                continue

            document = example["document"]
            query = PROMPT.format(document=document)

            response, _ = model.chat(
                tokenizer,
                query,
                history=None,
            )

            json_data = {
                "id": example_id,
                "document": document,
                "sys_summary": response,
                "ref_summary": example["summary"]
            }
            fout.write(json.dumps(json_data, ensure_ascii=False) + "\n")


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
        "--quant_checkpoint_path",
        type=str,
        help="Checkpoint path of quantized model",
        required=True,
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
        "--split",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_result_path", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=-1,
        help="Wait pid exit then start"
    )

    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)
    
    while psutil.pid_exists(args.pid):
        time.sleep(100)

    cli_main(args)
