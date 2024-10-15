import re
import math
import json
import argparse


INVALID_ANS = "[invalid]"


def extract_answer(completion):
    try:
        answer = completion.split("####")[-1].strip()
        answer = answer.replace(",", "")
        return answer
    except Exception:
        return INVALID_ANS


def extract_model_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        digits = [m.group().replace(",", "").replace("+", "").strip() for m in match]
    else:
        digits = None
        print(f"No digits found in {s!r}", flush=True)
    return digits


def is_correct(sys_answer, ref_answer):
    gold = extract_answer(ref_answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            for digit in pred:
                if math.isclose(float(answer), float(digit), rel_tol=0, abs_tol=1e-4):
                    return True
            return False
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_model_answer(sys_answer))


def get_acc(args):
    for input in args.input:
        fin = open(input, mode="r", encoding="utf-8")

        num_total = 0
        num_correct = 0

        for json_line in fin:
            json_data = json.loads(json_line)
            ref_answer = json_data["ref_answer"]
            sys_answer = json_data["sys_answer"]

            num_total += 1

            if is_correct(sys_answer, ref_answer):
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
