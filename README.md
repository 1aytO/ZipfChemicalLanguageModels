# Zipf-like Statistical Regularities in Molecular Sequence Representations

Reference code for:

> Anyu Liu, Chao Fang, Yuntao Li, Zongguo Wang, Tao Qi, and Guoping Hu.
> **Zipf-like Statistical Regularities in Molecular Sequence Representations
> for Chemical Language Models.** *The Journal of Physical Chemistry Letters*
> (2026). [ACS article](https://pubs.acs.org/jpclcd/article/doi/10.1021/acs.jpclett.6c02089/5259571/Zipf-like-Statistical-Regularities-in-Molecular) ·
> [DOI: 10.1021/acs.jpclett.6c02089](https://doi.org/10.1021/acs.jpclett.6c02089)

This repository contains a compact implementation of four operations used in
the study:

1. train a chemistry-aware BPE tokenizer for SMILES or SELFIES;
2. fit a Zipf-like token rank-frequency slope;
3. pretrain a BERT encoder with masked-language modeling;
4. fine-tune the encoder on a MoleculeNet-style regression or binary
   classification CSV.


## Files

```text
scripts/train_bpe.py              Molecular BPE training and tokenizer I/O
scripts/analyze_zipf.py           Rank-frequency fitting and plotting
scripts/pretrain_bert.py          BERT masked-language pretraining
scripts/finetune_moleculenet.py   Regression or classification fine-tuning
data/example_sequences.txt        Small SMILES example corpus
data/example_delaney.csv          20-row Delaney/ESOL format example
tests/test_smoke.py               Unit and command-line smoke tests
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

### 1. Train a molecular BPE tokenizer

```bash
python scripts/train_bpe.py \
  --input data/example_sequences.txt \
  --representation smiles \
  --vocab-size 32 \
  --output-dir outputs/tokenizer
```

The command writes `vocab.json`, `merges.txt`, and `token_counts.csv`.

For SELFIES input, store one SELFIES string per line and pass
`--representation selfies`.

### 2. Fit the Zipf-like rank-frequency slope

```bash
python scripts/analyze_zipf.py \
  --counts outputs/tokenizer/token_counts.csv \
  --json-out outputs/zipf_fit.json \
  --plot outputs/zipf_plot.png
```

The script fits ordinary least squares to
`log10(token frequency) ~ log10(token rank)` and reports the slope, intercept,
number of fitted points, and coefficient of determination.

### 3. Pretrain BERT with masked-language modeling

```bash
python scripts/pretrain_bert.py \
  --input data/example_sequences.txt \
  --representation smiles \
  --vocab outputs/tokenizer/vocab.json \
  --merges outputs/tokenizer/merges.txt \
  --output-dir outputs/pretrained_bert \
  --max-length 64 \
  --epochs 1
```

The default model uses 2 encoder layers, 4 attention heads, a hidden size of
128, and an intermediate size of 256. These dimensions keep the example small.

The paper used a standard encoder-only BERT with 12 layers, 12 attention heads,
a hidden size of 768, an intermediate size of 3072, dropout 0.1, and masking
probability 0.15. Supply the corresponding settings when you have the full
corpus and suitable hardware:

```bash
python scripts/pretrain_bert.py \
  --input PATH_TO_CORPUS.txt \
  --representation smiles \
  --vocab PATH_TO_TOKENIZER/vocab.json \
  --merges PATH_TO_TOKENIZER/merges.txt \
  --output-dir outputs/paper_scale_bert \
  --hidden-size 768 \
  --layers 12 \
  --heads 12 \
  --intermediate-size 3072 \
  --dropout 0.1 \
  --mask-probability 0.15
```

### 4. Fine-tune on a MoleculeNet-style CSV

```bash
python scripts/finetune_moleculenet.py \
  --csv data/example_delaney.csv \
  --sequence-column smiles \
  --label-column measured_log_solubility_in_mols_per_litre \
  --task regression \
  --representation smiles \
  --vocab outputs/tokenizer/vocab.json \
  --merges outputs/tokenizer/merges.txt \
  --model outputs/pretrained_bert \
  --output-dir outputs/delaney_model \
  --max-length 64 \
  --epochs 1
```

Regression runs report RMSE and R². Binary classification runs require labels
encoded as 0 and 1 and report ROC-AUC and accuracy.

The paper evaluated two regression tasks, Lipophilicity and Delaney-processed,
and four classification tasks, BACE, MUV, Tox21, and ToxCast. All 24 pretrained
models covered SMILES and SELFIES at vocabulary sizes 600, 700, 800, 900, 1000,
1500, 2000, 2500, 3000, 4000, 5000, and 8000.

## Input formats

`train_bpe.py` and `pretrain_bert.py` expect one molecular sequence per line.
`finetune_moleculenet.py` accepts a CSV and lets you name the sequence and label
columns. The scripts reject incomplete SMILES/SELFIES tokenization, empty
corpora, missing CSV columns, and nonnumeric labels.

The Delaney file in `data/` contains 20 unchanged sequence-label pairs from the
public ESOL data. Its size supports an execution check, not model assessment.

## Data sources

The paper analyzed molecular data from:

- [ChEMBL 36](https://www.ebi.ac.uk/chembl/)
- [ZINC-22](https://cartblanche22.docking.org/)
- [COCONUT](https://coconut.naturalproducts.net/)
- [GDB](https://gdb.unibe.ch/downloads/)
- [MoleculeNet](https://moleculenet.org/)

Download and prepare those datasets under their source terms. This repository
does not redistribute them.

## Tests

```bash
python -m pytest -v
```

## Citation

```bibtex
@article{liu2026zipf,
  title   = {Zipf-like Statistical Regularities in Molecular Sequence Representations for Chemical Language Models},
  author  = {Liu, Anyu and Fang, Chao and Li, Yuntao and Wang, Zongguo and Qi, Tao and Hu, Guoping},
  journal = {The Journal of Physical Chemistry Letters},
  year    = {2026},
  doi     = {10.1021/acs.jpclett.6c02089}
}
```

## License

The code is available under the [MIT License](LICENSE).
