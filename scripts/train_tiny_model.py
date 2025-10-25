#!/usr/bin/env python
"""Train a tiny language model on Hugging Face's book_cleaned.txt file.

This script downloads a text file from the Hugging Face Hub and fine-tunes a
small GPT-2 style model on a short subset of the data so it can run quickly on
CPUs.  The default configuration only consumes the first few million
characters, but the limits can be adjusted with command-line flags.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="bookcorpusopen",
        help="Dataset repository on the Hugging Face Hub that hosts book_cleaned.txt.",
    )
    parser.add_argument(
        "--filename",
        default="book_cleaned.txt",
        help="Filename inside the dataset repository to download.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000,
        help="Limit the amount of text (in characters) used for training. Set to 0 to use the entire file.",
    )
    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="Base causal language model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tiny-book-model"),
        help="Directory where the trained model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Maximum sequence length for training examples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency (in steps) at which to log training metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Frequency (in steps) at which checkpoints are saved. Set to 0 to disable checkpoints.",
    )
    return parser.parse_args()


def download_corpus(repo_id: str, filename: str) -> Path:
    LOGGER.info("Downloading %s from dataset repo %s", filename, repo_id)
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    LOGGER.info("Cached dataset at %s", path)
    return Path(path)


def load_text(path: Path, max_chars: int) -> str:
    LOGGER.info("Loading text from %s", path)
    text = path.read_text(encoding="utf-8")
    if max_chars > 0:
        text = text[:max_chars]
        LOGGER.info("Trimmed dataset to %d characters", len(text))
    else:
        LOGGER.info("Using full dataset (%d characters)", len(text))
    return text


def tokenize_corpus(text: str, tokenizer, block_size: int) -> Dataset:
    LOGGER.info("Tokenizing corpus")
    dataset = Dataset.from_dict({"text": [text]})

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        from itertools import chain

        concatenated = list(chain.from_iterable(batch["input_ids"]))
        total_length = len(concatenated)
        total_length = (total_length // block_size) * block_size
        input_ids = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": input_ids}

    chunked = tokenized.map(group_texts, batched=True)
    if len(chunked) == 0:
        raise ValueError(
            "No training sequences were created. Increase --max-chars or decrease --block-size."
        )
    LOGGER.info("Created %d training sequences", len(chunked))
    return chunked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = parse_args()

    corpus_path = download_corpus(args.repo_id, args.filename)
    text = load_text(corpus_path, args.max_chars)

    LOGGER.info("Loading tokenizer and model from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = tokenize_corpus(text, tokenizer, args.block_size)
    if len(dataset) < 2:
        LOGGER.warning(
            "Dataset only produced %d sequences; using the same data for training and evaluation.",
            len(dataset),
        )
        split = {"train": dataset, "test": dataset}
    else:
        split = dataset.train_test_split(test_size=0.1, seed=42)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_strategy = "steps" if args.save_steps > 0 else "no"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch" if len(dataset) >= 2 else "no",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=1,
        push_to_hub=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )

    LOGGER.info("Starting training")
    trainer.train()

    LOGGER.info("Saving model to %s", args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
