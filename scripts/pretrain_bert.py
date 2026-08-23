#!/usr/bin/env python3
"""Pretrain a compact BERT masked-language model on molecular sequences."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM

try:
    from scripts.train_bpe import encode_sequence, load_tokenizer
except ModuleNotFoundError:  # Direct execution: python scripts/pretrain_bert.py
    from train_bpe import encode_sequence, load_tokenizer


class SequenceDataset(Dataset):
    """Fixed-length token IDs and attention masks for molecular strings."""

    def __init__(
        self,
        sequences: list[str],
        representation: str,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        max_length: int,
    ) -> None:
        if not sequences:
            raise ValueError("at least one sequence is required")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.examples: list[dict[str, torch.Tensor]] = []
        pad_id = vocab["[PAD]"]
        for sequence in sequences:
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
            self.examples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def mask_tokens(
    input_ids: torch.Tensor,
    special_token_ids: set[int],
    mask_token_id: int,
    vocab_size: int,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT's 80/10/10 masking policy without masking special tokens."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("mask probability must be between 0 and 1")

    masked_input = input_ids.clone()
    labels = input_ids.clone()
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        eligible &= input_ids.ne(token_id)

    probabilities = torch.full(input_ids.shape, probability, dtype=torch.float)
    selected = torch.bernoulli(probabilities, generator=generator).bool() & eligible
    if probability > 0 and not selected.any() and eligible.any():
        first = eligible.nonzero(as_tuple=False)[0]
        selected[tuple(first.tolist())] = True

    labels[~selected] = -100
    replaced = (
        torch.bernoulli(
            torch.full(input_ids.shape, 0.8, dtype=torch.float),
            generator=generator,
        ).bool()
        & selected
    )
    masked_input[replaced] = mask_token_id

    random_replacements = (
        torch.bernoulli(
            torch.full(input_ids.shape, 0.5, dtype=torch.float),
            generator=generator,
        ).bool()
        & selected
        & ~replaced
    )
    random_tokens = torch.randint(
        vocab_size, input_ids.shape, generator=generator, dtype=torch.long
    )
    masked_input[random_replacements] = random_tokens[random_replacements]
    return masked_input, labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.hidden_size % args.heads != 0:
        raise ValueError("hidden size must be divisible by the number of heads")

    set_seed(args.seed)
    vocab, merges = load_tokenizer(args.vocab, args.merges)
    sequences = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = SequenceDataset(
        sequences,
        args.representation,
        vocab,
        merges,
        args.max_length,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_length,
        pad_token_id=vocab["[PAD]"],
    )
    model = BertForMaskedLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    masking_generator = torch.Generator().manual_seed(args.seed)
    special_ids = {vocab[token] for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]")}

    epoch_losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch in loader:
            masked_ids, labels = mask_tokens(
                batch["input_ids"],
                special_ids,
                vocab["[MASK]"],
                len(vocab),
                args.mask_probability,
                masking_generator,
            )
            outputs = model(
                input_ids=masked_ids.to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels.to(device),
            )
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("masked-language-model loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()
            batch_losses.append(float(outputs.loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    metrics: dict[str, object] = {
        "device": str(device),
        "n_sequences": len(dataset),
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1],
        "model": {
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "intermediate_size": args.intermediate_size,
            "dropout": args.dropout,
            "mask_probability": args.mask_probability,
        },
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("smiles", "selfies"), required=True
    )
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
