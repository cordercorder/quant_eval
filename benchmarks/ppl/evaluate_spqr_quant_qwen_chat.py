import argparse
from contextlib import contextmanager
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import set_seed
import evaluate
import sys
import os
import psutil
import time

sys.path.append('../../../quant/SpQR')
sys.path.append('../../quant/SpQR')
from modelutils import get_model


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


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

    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = tokenizer.eod_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eod_id

    # model = get_model(args.checkpoint_path, 2048, load_quantized = args.quant_checkpoint_path)

    return model, tokenizer

def preprocess_str(
    sources,
    system_message: str = "You are a helpful assistant."
):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    nl_tokens_str = '\n'
    _system_str = 'system' + nl_tokens_str
    # Apply prompt templates
    input_strs = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_str = ''
        system_str = '<|im_start|>' + _system_str + system_message + '<|im_end|>' + nl_tokens_str

        input_str += system_str

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_str = role + nl_tokens_str + sentence["value"] + '<|im_end|>' + nl_tokens_str
            input_str += _input_str

        input_strs.append(input_str)
    
    return input_strs


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
        input_str = ''
        system_str = '<|im_start|>' + 'system' + system_message + '<|im_end|>' + '\n'
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system

        input_str += system_str

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_str = role + '\n' + sentence["value"] + '<|im_end|>' + '\n'
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            input_str += _input_str

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


def alpaca_ppl(data_path, model, tokenizer):
    dataset = load_dataset('json', data_files=data_path)['train']
    sampled_data = dataset
    if args.num_sample_alpaca:
        sampled_data = dataset.train_test_split(test_size=args.num_sample_alpaca)["test"]
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

    input_texts = preprocess_str(sources)

    # encodings = preprocess(sources, tokenizer)

    perplexity = evaluate.load("../perplexity_hf.py", module_type="metric")

    input_texts = [s for s in input_texts if s!='']
    with suspend_nn_inits():
        results = perplexity.compute(model_id=args.checkpoint_path,
                                     model=model, tokenizer=tokenizer,
                                    batch_size=args.batch_size,
                                    predictions=input_texts, add_start_token=False)
    print('alpaca ppl: ')
    print(f'{results["mean_perplexity"]=}')
    print(f'{results["all_exp_ppl"]=}')


def cli_main(args):
    model, tokenizer = load_models_tokenizer(args)

    perplexity = evaluate.load("../perplexity_gptq.py", module_type="metric")

    ret = {}

    if args.wikitext:
        input_texts = load_from_disk(args.wikitext_data_path)["text"]
        print(len(input_texts))
        input_texts = [s for s in input_texts if s!='']
        with suspend_nn_inits():
            results = perplexity.compute(model_id=args.checkpoint_path,
                                        batch_size=args.wikitext_batch_size,
                                        join_seq='\n\n',
                                        model=model,
                                        tokenizer=tokenizer,
                                        predictions=input_texts)
        
        ret["wikitext"] = {
            "all_exp_ppl": results["all_exp_ppl"]
        }
        torch.cuda.empty_cache()

    if args.c4:
        input_texts = load_from_disk(args.c4_data_path)["text"]
        input_texts = input_texts[:1100]
        print(len(input_texts))
        input_texts = [s for s in input_texts if s!='']
        with suspend_nn_inits():
            results = perplexity.compute(model_id=args.checkpoint_path,
                                        batch_size=args.c4_batch_size,
                                        model=model,
                                        join_seq=' ',
                                        tokenizer=tokenizer,
                                        predictions=input_texts)
        ret["c4"] = {
            "all_exp_ppl": results["all_exp_ppl"]
        }
        torch.cuda.empty_cache()

    if args.ptb:
        input_texts = load_from_disk(args.ptb_data_path)["sentence"]
        print(len(input_texts))
        input_texts = [s for s in input_texts if s!='']
        with suspend_nn_inits():
            results = perplexity.compute(model_id=args.checkpoint_path,
                                        batch_size=args.ptb_batch_size,
                                        model=model,
                                        join_seq=' ',
                                        tokenizer=tokenizer,
                                        predictions=input_texts)
        ret["ptb"] = {
            "all_exp_ppl": results["all_exp_ppl"]
        }
        torch.cuda.empty_cache()
    
    with open(args.save_result_path, 'w') as f:
        json.dump(ret, f, indent=4)


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
        "--save_result_path",
        type=str,
        required=True,
        help="save_path"
    )

    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "--c4_data_path", 
        type=str, 
        help="Path to alpaca data path",
        required=True
    )
    group.add_argument(
        "--ptb_data_path", 
        type=str, 
        help="Path to alpaca data path",
        required=True
    )
    group.add_argument(
        "--wikitext_data_path", 
        type=str, 
        help="Path to alpaca data path",
        required=True
    )
    group.add_argument(
        "--num_sample_alpaca", 
        type=int,
        default=None,
        help="Sample nums to calc ppl",
        required=False
    )
    group.add_argument(
        "--wikitext_batch_size",
        type=int,
        default=1,
        required=False
    )
    group.add_argument(
        "--c4_batch_size",
        type=int,
        default=1,
        required=False
    )
    group.add_argument(
        "--ptb_batch_size",
        type=int,
        default=1,
        required=False
    )
    group.add_argument(
        "--wikitext",
        action='store_true',
        help="Compute ppl on wikitext"
    )
    group.add_argument(
        "--c4",
        action='store_true',
        help="Compute ppl on c4"
    )
    group.add_argument(
        "--ptb",
        action='store_true',
        help="Compute ppl on ptb"
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