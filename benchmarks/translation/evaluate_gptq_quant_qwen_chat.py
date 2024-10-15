import os
import argparse

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from transformers.trainer_utils import set_seed
from transformers.generation import GenerationConfig


PROMPT = "Please translate the following {source_lang} text into {target_lang}.\n{source_lang} text: {text}"


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    model = AutoGPTQForCausalLM.from_quantized(
        args.quant_checkpoint_path,
        device="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = model.model
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print(f"GenerationConfig: {model.generation_config}")
    return model, tokenizer


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)
    print("model loaded")

    save_result_dir = os.path.dirname(args.save_result_path)
    os.makedirs(save_result_dir, exist_ok=True)

    with open(args.save_result_path, mode="w", encoding="utf-8") as fout, \
        open(args.eval_data_path, mode="r", encoding="utf-8") as fin:

        for src_line in fin:
            src_line = src_line.rstrip()
            query = PROMPT.format(
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                text=src_line,
            )
            response, _ = model.chat(
                tokenizer,
                query,
                history=None,
            )

            response = response.replace("\n", " ")
            fout.write(response + "\n")


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
        "--source_lang", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target_lang", 
        type=str,
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
