#!/usr/bin/env python3
"""Train a small, deterministic BPE tokenizer for molecular sequences."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

SMILES_TOKEN_RE = re.compile(
    r"\[[^\[\]]+\]|"
    r"Br|Cl|Si|Se|Te|As|Sb|Bi|Sn|Pb|Ge|Ga|In|Tl|"
    r"Na|Mg|Al|Ca|Cr|Mn|Fe|Co|Ni|Cu|Zn|Mo|Li|Ag|Au|Pt|Pd|Hg|"
    r"B|C|N|O|P|S|F|I|H|K|"
    r"b|c|n|o|p|s|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9]"
)
SELFIES_TOKEN_RE = re.compile(r"\[[^\]]+\]")


def initial_tokenize(sequence: str, representation: str) -> list[str]:
    """Split one SMILES or SELFIES string into its initial BPE units."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a nonempty string")
    if any(char.isspace() for char in sequence):
        raise ValueError("sequence must not contain whitespace")

    patterns = {"smiles": SMILES_TOKEN_RE, "selfies": SELFIES_TOKEN_RE}
    try:
        pattern = patterns[representation.lower()]
    except KeyError as exc:
        raise ValueError("representation must be 'smiles' or 'selfies'") from exc

    tokens = pattern.findall(sequence)
    if not tokens or "".join(tokens) != sequence:
        raise ValueError(
            f"sequence is not fully covered by {representation.upper()} tokens: "
            f"{sequence}"
        )
    return tokens


def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def train_bpe(
    sequences: list[str], representation: str, vocab_size: int
) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, int]]:
    """Train BPE and return the vocabulary, ordered merges, and final counts."""
    if not sequences:
        raise ValueError("at least one sequence is required")

    corpus = [initial_tokenize(item, representation) for item in sequences]
    initial_tokens = sorted({token for row in corpus for token in row})
    minimum_size = len(SPECIAL_TOKENS) + len(initial_tokens)
    if vocab_size < minimum_size:
        raise ValueError(
            f"vocab_size must be at least {minimum_size} for this corpus"
        )

    ordered_tokens = SPECIAL_TOKENS + initial_tokens
    known_tokens = set(ordered_tokens)
    merges: list[tuple[str, str]] = []

    while len(ordered_tokens) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for row in corpus:
            pair_counts.update(zip(row, row[1:]))
        if not pair_counts:
            break

        highest_frequency = max(pair_counts.values())
        pair = min(
            candidate
            for candidate, count in pair_counts.items()
            if count == highest_frequency
        )
        merged_token = pair[0] + pair[1]
        corpus = [_merge_pair(row, pair) for row in corpus]
        merges.append(pair)

        if merged_token not in known_tokens:
            known_tokens.add(merged_token)
            ordered_tokens.append(merged_token)

    counts: Counter[str] = Counter(token for row in corpus for token in row)
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    return vocab, merges, dict(counts)


def encode_sequence(
    sequence: str,
    representation: str,
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    max_length: int | None = None,
) -> list[int]:
    """Apply ordered BPE merges and return IDs surrounded by CLS and SEP."""
    missing_specials = [token for token in SPECIAL_TOKENS if token not in vocab]
    if missing_specials:
        raise ValueError(f"vocabulary is missing special tokens: {missing_specials}")
    if max_length is not None and max_length < 2:
        raise ValueError("max_length must be at least 2")

    tokens = initial_tokenize(sequence, representation)
    for pair in merges:
        tokens = _merge_pair(tokens, pair)
    if max_length is not None:
        tokens = tokens[: max_length - 2]

    unknown = vocab["[UNK]"]
    return [vocab["[CLS]"]] + [vocab.get(token, unknown) for token in tokens] + [
        vocab["[SEP]"]
    ]


def save_tokenizer(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Write the compact tokenizer artifacts used by the other examples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "merges.txt").open("w", encoding="utf-8") as stream:
        for left, right in merges:
            stream.write(f"{left}\t{right}\n")
    with (output_dir / "token_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["token", "count"])
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([token, count])


def load_tokenizer(
    vocab_path: Path, merges_path: Path
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Load artifacts created by :func:`save_tokenizer`."""
    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = {str(token): int(index) for token, index in vocab_data.items()}
    merges: list[tuple[str, str]] = []
    for line in merges_path.read_text(encoding="utf-8").splitlines():
        if line:
            left, right = line.split("\t", maxsplit=1)
            merges.append((left, right))
    return vocab, merges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("smiles", "selfies"), required=True
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vocab, merges, counts = train_bpe(
        sequences, args.representation, args.vocab_size
    )
    save_tokenizer(vocab, merges, counts, args.output_dir)
    print(
        f"Saved {len(vocab)} tokens and {len(merges)} merges to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
