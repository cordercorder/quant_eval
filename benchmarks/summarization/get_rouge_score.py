import json
import argparse
import evaluate


def get_rouge_score(args):
    rouge = evaluate.load("rouge")

    for input in args.input:
        fin = open(input, mode="r", encoding="utf-8")

        predictions = []
        references = []

        prev_example_id = -1
        for json_line in fin:
            json_data = json.loads(json_line)

            example_id = json_data["id"]
            sys_summary = json_data["sys_summary"]
            ref_summary = json_data["ref_summary"]

            assert prev_example_id + 1 == example_id

            prev_example_id = example_id

            predictions.append(sys_summary)
            references.append(ref_summary)
        
        results = rouge.compute(predictions=predictions, references=references)

        print("="*16)
        print(f"input: {input}")
        print(results)
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
    get_rouge_score(args)


if __name__ == "__main__":
    cli_main()
