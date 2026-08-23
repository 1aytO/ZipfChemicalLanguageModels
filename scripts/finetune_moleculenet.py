#!/usr/bin/env python3
"""Fine-tune a pretrained molecular BERT model on a labeled CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification

try:
    from scripts.train_bpe import encode_sequence, load_tokenizer
except ModuleNotFoundError:  # Direct execution: python scripts/finetune_moleculenet.py
    from train_bpe import encode_sequence, load_tokenizer


def load_labeled_csv(
    path: Path, sequence_column: str, label_column: str
) -> tuple[list[str], list[float]]:
    """Load nonempty molecular strings and numeric labels from a CSV."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = [
            column
            for column in (sequence_column, label_column)
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(f"missing column(s): {', '.join(missing)}")

        sequences: list[str] = []
        labels: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            sequence = (row[sequence_column] or "").strip()
            if not sequence:
                raise ValueError(f"blank sequence at CSV row {row_number}")
            try:
                label = float(row[label_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"nonnumeric label at CSV row {row_number}") from exc
            sequences.append(sequence)
            labels.append(label)

    if len(sequences) < 2:
        raise ValueError("at least two labeled rows are required")
    return sequences, labels


class LabeledSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        labels: list[float],
        representation: str,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        max_length: int,
        task: str,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        pad_id = vocab["[PAD]"]
        label_dtype = torch.float if task == "regression" else torch.long
        for sequence, label in zip(sequences, labels, strict=True):
            ids = encode_sequence(
                sequence,
                representation,
                vocab,
                merges,
                max_length=max_length,
            )
            attention = [1] * len(ids)
            padding = max_length - len(ids)
            ids.extend([pad_id] * padding)
            attention.extend([0] * padding)
            value = label if task == "regression" else int(label)
            self.examples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                    "labels": torch.tensor(value, dtype=label_dtype),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("validation size must be between 0 and 1")

    set_seed(args.seed)
    sequences, labels = load_labeled_csv(
        args.csv, args.sequence_column, args.label_column
    )
    stratify = None
    if args.task == "classification":
        class_labels = [int(label) for label in labels]
        if any(float(integer) != label or integer not in (0, 1) for integer, label in zip(class_labels, labels, strict=True)):
            raise ValueError("classification labels must be 0 or 1")
        if len(set(class_labels)) != 2:
            raise ValueError("classification data must contain both classes")
        labels = [float(label) for label in class_labels]
        stratify = class_labels

    train_sequences, valid_sequences, train_labels, valid_labels = train_test_split(
        sequences,
        labels,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=stratify,
    )
    vocab, merges = load_tokenizer(args.vocab, args.merges)
    train_dataset = LabeledSequenceDataset(
        train_sequences,
        train_labels,
        args.representation,
        vocab,
        merges,
        args.max_length,
        args.task,
    )
    valid_dataset = LabeledSequenceDataset(
        valid_sequences,
        valid_labels,
        args.representation,
        vocab,
        merges,
        args.max_length,
        args.task,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    config = BertConfig.from_pretrained(args.model)
    config.num_labels = 1 if args.task == "regression" else 2
    config.problem_type = (
        "regression" if args.task == "regression" else "single_label_classification"
    )
    model = BertForSequenceClassification.from_pretrained(args.model, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("fine-tuning loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()

    model.eval()
    observed: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for batch in valid_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.cpu()
            observed.extend(batch["labels"].tolist())
            if args.task == "regression":
                predictions.extend(logits.squeeze(-1).tolist())
            else:
                predictions.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())

    if args.task == "regression":
        metrics: dict[str, float | int | str] = {
            "task": args.task,
            "n_train": len(train_dataset),
            "n_validation": len(valid_dataset),
            "rmse": float(mean_squared_error(observed, predictions) ** 0.5),
            "r2": float(r2_score(observed, predictions)),
        }
    else:
        predicted_classes = [int(value >= 0.5) for value in predictions]
        metrics = {
            "task": args.task,
            "n_train": len(train_dataset),
            "n_validation": len(valid_dataset),
            "roc_auc": float(roc_auc_score(observed, predictions)),
            "accuracy": float(accuracy_score(observed, predicted_classes)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sequence-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument(
        "--task", choices=("regression", "classification"), required=True
    )
    parser.add_argument(
        "--representation", choices=("smiles", "selfies"), required=True
    )
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
