import os
import torch
import argparse
import numpy as np
import pandas as pd

from typing import Tuple, List
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nThe answer is:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(subject):
    prompt = "The following is a multiple-choice question about{}. Please select the most appropriate answer from options A, B, C, and D for this question. Provide the answer directly, without detailing the intermediate steps or reasoning used in the problem-solving process.\n\n".format(format_subject(subject))
    return prompt


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


@torch.no_grad()
def evaluate(subject, model, tokenizer, choices_ids, test_df):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(subject)
        prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        _, context_tokens = make_context(tokenizer=tokenizer, query=prompt, chat_format="raw")
        input_ids = torch.as_tensor([context_tokens], device="cuda")

        outputs = model.generate(
            input_ids,
            return_dict_in_generate=False,
        )

        print(f"prompt: {prompt}")
        print(f"outputs: {tokenizer.decode(outputs[0])}")


def get_choices_ids(tokenizer):
    choices_ids = []

    for choice in choices:
        choice_id = tokenizer.encode(
            choice,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        assert len(choice_id) == 1

        # [1, ]
        choice_id = torch.as_tensor(choice_id, dtype=torch.long, device="cuda")
        choices_ids.append(choice_id)
    
    # [1, 4]
    choices_ids = torch.stack(choices_ids, dim=0)
    choices_ids = choices_ids.unsqueeze(0)
    return choices_ids


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        trust_remote_code=True,
    )
    model = model.eval()

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    model_name = os.path.basename(args.pretrained_model_name_or_path)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(model_name))):
        os.mkdir(os.path.join(args.save_dir, "results_{}".format(model_name)))

    print(f"subjects: {subjects}")
    print(f"args: {args}")

    print(f"model_name: {model_name}")

    choices_ids = get_choices_ids(tokenizer)

    all_cors = []

    for subject in subjects:
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        evaluate(subject, model, tokenizer, choices_ids, test_df)
    #     print("Average accuracy {:.3f} - {}".format(acc, subject))

    #     all_cors.append(cors)

    #     test_df["{}_correct".format(model_name)] = cors
    #     for j in range(probs.shape[1]):
    #         choice = choices[j]
    #         test_df["{}_choice{}_probs".format(model_name, choice)] = probs[:, j]
    #     test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(model_name), "{}.csv".format(subject)), index=None)

    # weighted_acc = np.mean(np.concatenate(all_cors))
    # print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
