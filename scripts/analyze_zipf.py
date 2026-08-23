#!/usr/bin/env python3
"""Fit a Zipf-like rank-frequency line to molecular token counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fit_zipf(
    counts: list[int], min_rank: int = 1, max_rank: int | None = None
) -> dict[str, float | int]:
    """Fit log10(frequency) against log10(rank) with ordinary least squares."""
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("token counts must be positive")
    if min_rank < 1:
        raise ValueError("min_rank must be at least 1")

    ordered = np.asarray(sorted(counts, reverse=True), dtype=float)
    final_rank = len(ordered) if max_rank is None else max_rank
    if final_rank < min_rank or final_rank > len(ordered):
        raise ValueError("rank range is outside the available token counts")

    selected = ordered[min_rank - 1 : final_rank]
    if len(selected) < 3:
        raise ValueError("at least three rank-frequency points are required")

    ranks = np.arange(min_rank, final_rank + 1, dtype=float)
    log_rank = np.log10(ranks)
    log_frequency = np.log10(selected)
    slope, intercept = np.polyfit(log_rank, log_frequency, 1)
    predicted = slope * log_rank + intercept
    residual_sum = float(np.sum((log_frequency - predicted) ** 2))
    total_sum = float(np.sum((log_frequency - log_frequency.mean()) ** 2))
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n_points": int(len(selected)),
        "min_rank": int(min_rank),
        "max_rank": int(final_rank),
    }


def load_counts(path: Path) -> list[int]:
    """Load the `count` column written by train_bpe.py."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "count" not in reader.fieldnames:
            raise ValueError("counts CSV must contain a 'count' column")
        try:
            return [int(row["count"]) for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError("count values must be integers") from exc


def save_plot(counts: list[int], result: dict[str, float | int], path: Path) -> None:
    """Save a compact log-log rank-frequency plot and fitted line."""
    import matplotlib.pyplot as plt

    ordered = np.asarray(sorted(counts, reverse=True), dtype=float)
    min_rank = int(result["min_rank"])
    max_rank = int(result["max_rank"])
    ranks = np.arange(min_rank, max_rank + 1, dtype=float)
    frequencies = ordered[min_rank - 1 : max_rank]
    fitted = 10 ** (
        float(result["intercept"]) + float(result["slope"]) * np.log10(ranks)
    )

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    axis.scatter(ranks, frequencies, s=18, color="#1868B2", label="Tokens")
    axis.plot(
        ranks,
        fitted,
        color="black",
        linewidth=1.2,
        label=(
            f"OLS slope = {float(result['slope']):.3f}, "
            f"$R^2$ = {float(result['r_squared']):.3f}"
        ),
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Token rank")
    axis.set_ylabel("Token frequency")
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = load_counts(args.counts)
    result = fit_zipf(counts, args.min_rank, args.max_rank)
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report, encoding="utf-8")
    if args.plot:
        save_plot(counts, result, args.plot)


if __name__ == "__main__":
    main()
