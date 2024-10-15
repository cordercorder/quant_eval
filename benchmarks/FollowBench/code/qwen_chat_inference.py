import os
import json
import torch
import argparse

from utils import convert_to_api_input
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )

    torch_dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map[args.torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"GenerationConfig: {model.generation_config}")
    return model, tokenizer


@torch.inference_mode()
def inference(args):
    # Load model
    model, tokenizer = load_models_tokenizer(args)

    for constraint_type in args.constraint_types:

        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        for i in range(len(data)):
            msg = data[i]['prompt_new']            
            response, _ = model.chat(
                tokenizer,
                msg,
                history=None,
            )
            data[i]['choices'] = [{'message': {'content': ""}}]
            data[i]['choices'][0]['message']['content'] = response

        # save file
        with open(os.path.join(args.api_output_path, f"{args.model_name}_{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen-7B-Chat",
    )
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
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["fp16", "bf16"],
        required=True
    )
    
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output")
    
    args = parser.parse_args()
    print(f"Args: {args}")

    set_seed(args.seed)

    if not os.path.exists(args.api_input_path):
        os.makedirs(args.api_input_path)

    if not os.path.exists(args.api_output_path):
        os.makedirs(args.api_output_path)
    
    ### convert data to api_input
    for constraint_type in args.constraint_types:
        convert_to_api_input(
            data_path=args.data_path, 
            api_input_path=args.api_input_path, 
            constraint_type=constraint_type
        )

    ### model inference
    inference(args)
