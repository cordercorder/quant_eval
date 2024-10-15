import os
import time
import json
import argparse

import torch
import shortuuid
from tqdm import tqdm

from fastchat.llm_judge.common import temperature_config
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(question_file: str):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model(args):
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
    model, tokenizer = load_model(args)
    print("model loaded")

    model_id = os.path.basename(args.checkpoint_path)

    questions = load_questions(args.question_file)

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    with open(os.path.expanduser(args.answer_file), mode="w", encoding="utf-8") as fout:
        for question in tqdm(questions):
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

            choices = []

            for i in range(args.num_choices):
                torch.manual_seed(i)

                turns = []
                history = []

                for j in range(len(question["turns"])):
                    question_turn_j = question["turns"][j]

                    if temperature < 1e-4:
                        do_sample = False
                    else:
                        do_sample = True
                    
                    response, history = model.chat(
                        tokenizer, 
                        question_turn_j, 
                        history=history,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=args.max_new_token,
                    )

                    turns.append(response)
                
                choices.append({"index": i, "turns": turns})

            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        default=1234, 
        help="Random seed"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B-Chat",
    )

    parser.add_argument(
        "--question_file", 
        type=str,
        required=True,
    )

    parser.add_argument(
        "--answer_file", 
        type=str, 
        required=True,
        help="The output answer file."
    )

    parser.add_argument(
        "--num_choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )

    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)

    cli_main(args)
