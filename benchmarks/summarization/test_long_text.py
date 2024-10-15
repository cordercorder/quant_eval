import sys
sys.path.append("/home/jinrenren/pretrained_models/Qwen-72B-Chat")

from datasets import load_from_disk
from transformers import AutoTokenizer

from qwen_generation_utils import (
    make_context,
)


PROMPT = "Please summarize the following document.\n{document}"


def test():
    ds = load_from_disk("/home/jinrenren/datasets/xsum")["test"]
    num_skiped_examples = 8018
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/jinrenren/pretrained_models/Qwen-72B-Chat",
        trust_remote_code=True
    )

    def get_encode_new(tokenizer):
        tokenizer.__encode_old__ = tokenizer.encode
        
        def encode_new(*args, **kwargs):
            result = tokenizer.__encode_old__(*args, **kwargs)
            return result[:8192]

        return encode_new

    tokenizer.encode = get_encode_new(tokenizer)

    for example_id, example in enumerate(ds):
            
        if example_id < num_skiped_examples:
            continue

        document = example["document"]
        query = PROMPT.format(document=document)

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=None,
            system="You are a helpful assistant.",
            max_window_size=6144,
            chat_format="chatml",
        )

        print(f"context_tokens_length: {len(context_tokens)}")
        break


if __name__ == "__main__":
    test()
