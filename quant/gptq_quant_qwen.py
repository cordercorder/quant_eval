import time
import torch
import argparse
import numpy as np

from auto_gptq import (
    AutoGPTQForCausalLM, 
    BaseQuantizeConfig
)
from transformers import (
    AutoTokenizer,
)
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--fast_tokenizer", 
        action="store_true", 
        help="whether use fast tokenizer"
    )

    parser.add_argument(
        "--quantized_model_dir", 
        type=str, 
        required=True,
    )

    # args for BaseQuantizeConfig
    parser.add_argument(
        "--bits", 
        type=int, 
        default=4, 
        choices=[2, 3, 4, 8]
    )
    parser.add_argument(
        "--group_size", 
        type=int,
        default=128, 
    )
    parser.add_argument(
        "--damp_percent", 
        type=float,
        default=0.01, 
    )
    parser.add_argument(
        "--no_desc_act", 
        action="store_false",
        dest="desc_act",
    )
    parser.add_argument(
        "--static_groups", 
        action="store_true",
    )
    parser.add_argument(
        "--no_sym", 
        action="store_false",
        dest="sym",
    )
    parser.add_argument(
        "--no_true_sequential", 
        action="store_false",
        dest="true_sequential",
    )

    # args for quantization
    parser.add_argument(
        "--quant_batch_size", 
        type=int, 
        default=1, 
        help="examples batch size for quantization"
    )
    parser.add_argument(
        "--use_triton", 
        action="store_true", 
        help="whether use triton to speedup at inference"
    )
    parser.add_argument(
        "--per_gpu_max_memory", 
        type=int, 
        default=None, 
        help="max memory used to load model per gpu"
    )
    parser.add_argument(
        "--cpu_max_memory", 
        type=int, 
        default=None, 
        help="max memory used to offload model to cpu"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=128, 
        help="how many samples will be used to quantize model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True
    )

    args = parser.parse_args()
    return args


def preprocess(
    sources,
    tokenizer,
    system_message: str = "You are a helpful assistant."
):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    # Apply prompt templates
    input_ids = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id = []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id

        input_id = torch.as_tensor(input_id, dtype=torch.long)
        input_id.unsqueeze_(0)
        attention_mask = input_id.new_ones([input_id.shape[0], input_id.shape[1]], dtype=torch.bool)
        input_ids.append(
            {
                "input_ids": torch.as_tensor(input_id, dtype=torch.long),
                "attention_mask": attention_mask
            }
        )
    
    return input_ids


def gptq_quant_qwen(args):
    print(f"Args: {args}")

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        use_fast=args.fast_tokenizer,
        trust_remote_code=True
    )

    data = load_dataset("json", data_files=args.data_path, split="train")
    assert len(data) >= args.num_samples

    rng = np.random.default_rng(args.seed)
    sampled_index = rng.choice(len(data), size=args.num_samples, replace=False)
    sampled_data = data.select(sampled_index)
    del data

    sources = []
    for sample in sampled_data:
        sources.append(
            [
                {
                    "from": "user",
                    "value": sample["instruction"] if len(sample["input"]) == 0 else sample["instruction"] + "\n\n" + sample["input"]
                },
                {
                    "from": "assistant",
                    "value": sample["output"]
                }
            ]
        )
    
    del sampled_data

    input_ids = preprocess(sources, tokenizer)

    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        quantize_config=BaseQuantizeConfig(
            bits=args.bits, 
            group_size=args.group_size, 
            damp_percent=args.damp_percent,
            desc_act=args.desc_act,
            static_groups=args.static_groups,
            sym=args.sym,
            true_sequential=args.true_sequential,
        ),
        max_memory=max_memory,
        trust_remote_code=True
    )

    start = time.time()
    model.quantize(
        input_ids,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    model.save_quantized(args.quantized_model_dir)


def cli_main():
    args = get_args()
    gptq_quant_qwen(args)


if __name__ == "__main__":
    cli_main()
