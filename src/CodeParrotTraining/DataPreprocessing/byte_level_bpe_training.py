import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


def tokenizer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", type=str, default="gpt2")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--is_iter_dataset", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default="code-parrot-minimal")
    parser.add_argument("--dataset", type=str, default="huggingface-course/codeparrot-ds-train")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--text_column", type=str, default="content")
    parser.add_argument("--num_samples", type=int, default=200_000)
    return parser


# Iterator for Training
def batch_iterator(dataset, text_column, batch_size, num_samples, is_iter_dataset):
    if is_iter_dataset:
        dataset = iter(dataset)
        for _ in tqdm(range(0, num_samples, batch_size)):
            yield [next(dataset)[text_column] for _ in range(batch_size)]
    else:
        for start_idx in tqdm(range(0, len(dataset), batch_size)):
            samples = dataset[start_idx : start_idx + batch_size]
            yield samples[text_column]


if __name__ == "__main__":
    parser = tokenizer_args()
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    base_vocab = list(bytes_to_unicode().values())

    # Load dataset
    dataset = load_dataset(args.dataset, split="train", streaming=args.is_iter_dataset)

    # Training and saving
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(dataset, args.text_column, args.batch_size, args.num_samples, args.is_iter_dataset),
        vocab_size=50257,
        initial_alphabet=base_vocab,
    )
    new_tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=True, private=True)
