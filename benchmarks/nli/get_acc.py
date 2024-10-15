import json
import argparse
import unicodedata


ANSWER_MAP = {
    0: "yes",
    1: "maybe",
    2: "no",
}


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def get_acc(args):
    for input in args.input:
        fin = open(input, mode="r", encoding="utf-8")

        num_total = 0
        num_correct = 0

        for json_line in fin:
            json_data = json.loads(json_line)
            sys_response = json_data["sys_response"]
            label = json_data["label"]

            if label not in ANSWER_MAP:
                continue

            sys_response = sys_response.lower()
            answer = ANSWER_MAP[label]
            num_total += 1

            left_idx = sys_response.find(answer)
            if left_idx == -1:
                continue
            
            if left_idx == 0:
                left_char = " "
            else:
                left_char = sys_response[left_idx - 1]
            
            right_idx = left_idx + len(answer)
            if right_idx == len(sys_response):
                right_char = " "
            else:
                right_char = sys_response[right_idx]
            
            if (_is_whitespace(left_char) or _is_punctuation(left_char)) and \
                (_is_whitespace(right_char) or _is_punctuation(right_char)):
                num_correct += 1

        fin.close()

        print("="*16)
        print(input)
        print(f"Acc: {num_correct / num_total:.3f}")
        print("="*16)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()
    return args


def cli_main():
    args = get_args()
    get_acc(args)


if __name__ == "__main__":
    cli_main()
