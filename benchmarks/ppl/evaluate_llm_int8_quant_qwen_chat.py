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

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return model, tokenizer


def cli_main(args):
    model, tokenizer = load_models_tokenizer(args)

    device = model.device

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
            
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:]

            nll_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(), 
                shift_labels.view(-1),
            )

            nll_loss_collection.append(nll_loss)

        ppl = torch.exp(torch.stack(nll_loss_collection).mean()).item()
        print("#" * 16)
        print(f"data_path: {data_path}, ppl: {ppl}")
        print("#" * 16)

        results[data_path] = ppl
    
    with open(args.save_result_path, "w") as fout:
        json.dump(results, fout, ensure_ascii=False)
    

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
        "--save_result_path",
        type=str,
        required=True,
        help="save_path"
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
