import os
import json
import torch
import argparse
import transformers
import numpy as np

from datasets import load_from_disk
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from transformers.trainer_utils import set_seed
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
):
    """mask all tokens in the prompt"""

    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(system)
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == '<|im_start|>assistant':
                _target = [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [IGNORE_TOKEN_ID] * 2
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, 
        trust_remote_code=True
    )
    model = AutoGPTQForCausalLM.from_quantized(
        args.quant_checkpoint_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = model.model
    return model, tokenizer


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    save_result_dir = os.path.dirname(args.save_result_path)
    os.makedirs(save_result_dir, exist_ok=True)

    ds = load_from_disk(args.eval_data_path)["validation"]

    num_total = 0
    num_correct = 0

    device = model.device

    with open(args.save_result_path, mode="w", encoding="utf-8") as fout:
        for example_id, example in enumerate(ds):

            question = example["question"]
            mc1_targets = example["mc1_targets"]

            choices = mc1_targets["choices"]
            labels = mc1_targets["labels"]

            loss_collection = []
            for choice in choices:
                conversations = [
                    [
                        {
                            "from": "user",
                            "value": question,
                        },
                        {
                            "from": "assistant",
                            "value": choice,
                        },
                    ]
                ]

                inputs = preprocess(conversations, tokenizer, args.max_len)

                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                
                with torch.no_grad():
                    loss = model(**inputs).loss.item()
                
                loss_collection.append(loss)
            
            loss_collection = np.asarray(loss_collection, dtype=np.float32)

            model_predict = int(np.argmin(loss_collection))
            num_correct += int(model_predict == labels.index(1))
            num_total += 1

            fout.write(
                json.dumps(
                    {
                        "id": example_id,
                        "model_predict": model_predict,
                        "labels": labels,
                    }, 
                    ensure_ascii=False
                ) + "\n"
            )

    print("="*16)
    print(f"num_total: {num_total}")
    print(f"num_correct: {num_correct}")
    print(f"Acc: {num_correct / num_total:.3f}")
    print("="*16)


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

    parser.add_argument(
        "--max_len",
        type=int,
        default=8192,
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
