import os
import json
import torch
import argparse
from datasets import load_from_disk

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
)
from transformers.trainer_utils import set_seed


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )
    # left padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id

    config = AutoConfig.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )
    config.use_flash_attn = args.use_flash_attn

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        config=config,
        device_map="auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return model, tokenizer


def get_max_memory_allocated(devices):
    return sum(
        torch.cuda.max_memory_allocated(device) for device in devices
    )


def get_max_memory_reserved(devices):
    return sum(
        torch.cuda.max_memory_reserved(device) for device in devices
    )


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    save_result_dir = os.path.dirname(args.save_result_path)
    os.makedirs(save_result_dir, exist_ok=True)

    ds = load_from_disk(args.eval_data_path)

    if args.split is not None:
        ds = ds[args.split]

    device = model.device

    all_devices = list(range(torch.cuda.device_count()))

    samples = ds[: args.batch_size][args.filed_name]
    inputs = tokenizer(
        samples,
        truncation=True,
        padding="max_length",
        max_length=args.input_length,
        return_token_type_ids=False,
        return_tensors="pt"
    )

    result_data = {
        "use_flash_attn": args.use_flash_attn,
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "num_new_tokens": args.num_new_tokens
    }

    max_memory_allocated_after_load_model = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_allocated_after_load_model"] = f"{max_memory_allocated_after_load_model} GB"

    max_memory_reserved_after_load_model = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_reserved_after_load_model"] = f"{max_memory_reserved_after_load_model} GB"
    
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)
    
    max_memory_allocated_after_input_to_cuda = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_allocated_after_input_to_cuda"] = f"{max_memory_allocated_after_input_to_cuda} GB"

    max_memory_reserved_after_input_to_cuda = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_reserved_after_input_to_cuda"] = f"{max_memory_reserved_after_input_to_cuda} GB"

    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs.input_ids,
                do_sample=False,
                top_k=None,
                top_p=1.0,
                max_new_tokens=args.num_new_tokens,
                min_new_tokens=args.num_new_tokens,
                attention_mask=inputs.attention_mask,
                return_dict_in_generate=False,
            )

            max_memory_allocated_after_generate = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
            result_data["max_memory_allocated_after_generate"] = f"{max_memory_allocated_after_generate} GB"

            max_memory_reserved_after_generate = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
            result_data["max_memory_reserved_after_generate"] = f"{max_memory_reserved_after_generate} GB"

            assert outputs.shape[1] == args.input_length + args.num_new_tokens, \
                f"outputs.shape: {outputs.shape}"
            del outputs
        except Exception as e:
            result_data["exception_message"] = str(e)

    max_memory_allocated_final = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_allocated_final"] = f"{max_memory_allocated_final} GB"

    max_memory_reserved_final = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
    result_data["max_memory_reserved_final"] = f"{max_memory_reserved_final} GB"

    with open(args.save_result_path, mode="a", encoding="utf-8") as fout:
        fout.write(json.dumps(result_data, ensure_ascii=False) + "\n")


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

    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
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
        "--filed_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_result_path", 
        type=str,
        required=True,
    )

    # Args for generation
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)

    cli_main(args)
