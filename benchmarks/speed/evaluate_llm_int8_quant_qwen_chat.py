import torch
import argparse

from datasets import load_from_disk
import torch.utils.benchmark as benchmark

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


def func(model, input_ids, attention_mask, num_new_tokens):
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            do_sample=False,
            top_k=None,
            top_p=1.0,
            max_new_tokens=num_new_tokens,
            min_new_tokens=num_new_tokens,
            attention_mask=attention_mask,
            return_dict_in_generate=False,
        )

        assert outputs.shape[1] == input_ids.shape[1] + num_new_tokens


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    ds = load_from_disk(args.eval_data_path)

    if args.split is not None:
        ds = ds[args.split]

    device = model.device

    samples = ds[: args.batch_size][args.filed_name]
    inputs = tokenizer(
        samples,
        truncation=True,
        padding="max_length",
        max_length=args.input_length,
        return_token_type_ids=False,
        return_tensors="pt"
    )

    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    t = benchmark.Timer(
        stmt='func(model, input_ids, attention_mask, num_new_tokens)', 
        setup='from __main__ import func',
        globals={
            'model': model,
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'num_new_tokens': args.num_new_tokens,
        }
    )

    results = t.timeit(args.num_times)
    print(f"Results: {results}")


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
    
    parser.add_argument(
        "--num_times",
        type=int,
        default=3
    )
    
    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)

    cli_main(args)
