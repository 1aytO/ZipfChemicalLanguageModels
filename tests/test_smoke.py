import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.analyze_zipf import fit_zipf
from scripts.finetune_moleculenet import load_labeled_csv
from scripts.pretrain_bert import mask_tokens
from scripts.train_bpe import (
    SPECIAL_TOKENS,
    encode_sequence,
    initial_tokenize,
    load_tokenizer,
    save_tokenizer,
    train_bpe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "train_bpe.py",
    "analyze_zipf.py",
    "pretrain_bert.py",
    "finetune_moleculenet.py",
]


def test_initial_tokenization_supports_smiles_and_selfies():
    assert initial_tokenize("ClC(=O)N", "smiles") == [
        "Cl",
        "C",
        "(",
        "=",
        "O",
        ")",
        "N",
    ]
    assert initial_tokenize("[C][=O][O]", "selfies") == [
        "[C]",
        "[=O]",
        "[O]",
    ]


def test_bpe_is_deterministic_and_serializable(tmp_path: Path):
    sequences = ["CCO", "CCN", "CCCl", "CCO"]
    vocab, merges, counts = train_bpe(
        sequences, "smiles", vocab_size=14
    )

    assert list(vocab)[:5] == SPECIAL_TOKENS
    assert merges[0] == ("C", "C")
    assert sum(counts.values()) > 0

    save_tokenizer(vocab, merges, counts, tmp_path)
    loaded_vocab, loaded_merges = load_tokenizer(
        tmp_path / "vocab.json", tmp_path / "merges.txt"
    )

    assert loaded_vocab == vocab
    assert loaded_merges == merges
    assert encode_sequence("CCO", "smiles", vocab, merges)[0] == vocab[
        "[CLS]"
    ]


def test_fit_zipf_recovers_inverse_rank_slope():
    counts = [round(10000 / rank) for rank in range(1, 101)]
    result = fit_zipf(counts)

    assert math.isclose(result["slope"], -1.0, abs_tol=0.03)
    assert result["r_squared"] > 0.99
    assert result["n_points"] == 100


def test_fit_zipf_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match="positive"):
        fit_zipf([10, 0, 2])


def test_mask_tokens_never_predicts_special_tokens():
    inputs = torch.tensor([[2, 6, 7, 3, 0, 0]])
    generator = torch.Generator().manual_seed(7)

    masked, labels = mask_tokens(
        inputs,
        special_token_ids={0, 2, 3},
        mask_token_id=4,
        vocab_size=12,
        probability=1.0,
        generator=generator,
    )

    assert labels[0, 0].item() == -100
    assert labels[0, 3].item() == -100
    assert labels[0, 4].item() == -100
    assert labels[0, 1].item() == 6
    assert labels[0, 2].item() == 7
    assert masked.shape == inputs.shape


def test_load_labeled_csv_validates_columns(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text(
        "smiles,target\nCCO,-0.3\nCCN,-0.5\n", encoding="utf-8"
    )

    sequences, labels = load_labeled_csv(path, "smiles", "target")

    assert sequences == ["CCO", "CCN"]
    assert labels == [-0.3, -0.5]
    with pytest.raises(ValueError, match="missing column"):
        load_labeled_csv(path, "sequence", "target")


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_command_line_help(script_name: str):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_tracked_files_exclude_private_paths_secrets_and_large_artifacts():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    paths = [
        REPO_ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]
    forbidden = (
        "/" + "lay/",
        "/home/" + "ps/",
        "github" + "_pat_",
        "gh" + "p_",
        "-----BEGIN " + "PRIVATE KEY-----",
    )

    for path in paths:
        assert path.stat().st_size <= 5 * 1024 * 1024, path
        if path.suffix.lower() in {".py", ".md", ".txt", ".csv", ".json"}:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in text, f"{marker!r} found in {path}"
