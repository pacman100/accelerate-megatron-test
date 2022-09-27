import argparse
from datasets import load_dataset


def tokenizer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="huggingface-course/codeparrot-ds-train")
    parser.add_argument("--output_filepath", type=str)
    return parser


if __name__ == "__main__":
    parser = tokenizer_args()
    args = parser.parse_args()
    train_data = load_dataset(args.dataset, split="train")
    train_data.to_json(args.output_filepath, lines=True)
