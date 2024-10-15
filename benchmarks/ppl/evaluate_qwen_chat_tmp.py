import json
import time
import torch
import psutil
import argparse

from datasets import load_from_disk
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        torch_dtype=torch_dtype, 
        trust_remote_code=True, 
        device_map="auto"
    )
    return model, tokenizer


def cli_main(args):
    model, tokenizer = load_models_tokenizer(args)

    device = model.device

    for param in model.parameters():
        print(param.dtype)

    results = {}

    for data_path, filed, num_samples in zip(args.data_paths, args.fileds, args.num_samples):
        testdata = load_from_disk(data_path)

        if num_samples > 0:
            testdata = testdata[: num_samples]
        
        if "wikitext" in data_path:
            testenc = tokenizer.encode("\n\n".join(testdata[filed]), return_tensors="pt")
        else:
            testenc = tokenizer.encode(" ".join(testdata[filed]), return_tensors="pt")

        num_segments = testenc.numel() // args.seqlen

        nll_loss_collection = []
        for i in range(num_segments):
            input_ids = testenc[:, i * args.seqlen: (i + 1) * args.seqlen]
            input_ids = input_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids).logits
            
            print(f"logits.dtype: {logits.dtype}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Checkpoint path",
        default="Qwen-7B-Chat",
        required=True
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

    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "--seqlen",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--data_paths", 
        type=str, 
        help="Path to alpaca data path",
        nargs="+",
        required=True
    )
    group.add_argument(
        "--fileds", 
        type=str, 
        help="filed names of text",
        nargs="+",
        required=True
    )
    group.add_argument(
        "--num_samples", 
        type=int, 
        help="number of used samples, negative number denote to use all samples",
        nargs="+",
        required=True
    )
    
    parser.add_argument(
        "--pid",
        type=int,
        default=-1,
        help="Wait pid exit then start"
    )

    args = parser.parse_args()

    print(f"Args: {args}")

    assert len(args.data_paths) == len(args.fileds) == len(args.num_samples)

    set_seed(args.seed)

    while psutil.pid_exists(args.pid):
        time.sleep(100)

    cli_main(args)
