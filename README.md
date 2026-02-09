# The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations

This repository contains code to reproduce the analysis reported in the paper **“The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations”**.

Core tasks include:
- Building a data-driven molecular vocabulary from SELFIES sequences using **Byte-Pair Encoding (BPE)**
- Tracking vocabulary growth statistics across merge steps
- Verifying **Zipf-like scaling** in molecular token rank–frequency distributions
- Exporting intermediate logs and final vocabularies for downstream analysis and figure generation

## Repository Structure

```
.
├── data/
│   └── chembl_selfies.txt              # Input corpus (SELFIES, one sequence per line)
├── outputs/
│   └── bpe_stats_sharded/              # All generated logs, stats, vocab files
├── notebooks/
│   └── *.ipynb                         # Optional: exploratory / figure notebooks
├── src/
│   └── *.py                            # Core scripts (if you split code from notebooks)
└── README.md
```

## Quick Start

### 1) Prepare input data
Place your SELFIES corpus at:
- `data/chembl_selfies.txt`

Format:
- one SELFIES string per line

### 2) Run
Use your main script / notebook entry to run BPE and statistics logging.  
All outputs are written to:
- `outputs/bpe_stats_sharded/`

## Configuration: Use Relative Paths (Recommended)

Do **not** hard-code absolute paths like:
- `/home/ps/lay/code/zipf/data/chembl_selfies.txt`
- `/home/ps/lay/code/zipf/bpe_stats_sharded`

Instead, set paths relative to the repository root.

### Option A: in a Python script (`.py`)
```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
DATA_PATH = REPO_ROOT / "data" / "chembl_selfies.txt"
OUT_DIR   = REPO_ROOT / "outputs" / "bpe_stats_sharded"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Option B: in a notebook (`.ipynb`)
If you start Jupyter at the repository root:
```python
from pathlib import Path

REPO_ROOT = Path.cwd()
DATA_PATH = REPO_ROOT / "data" / "chembl_selfies.txt"
OUT_DIR   = REPO_ROOT / "outputs" / "bpe_stats_sharded"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

If you start Jupyter inside `notebooks/`:
```python
from pathlib import Path

REPO_ROOT = Path.cwd().parents[0]  # notebooks -> repo root
DATA_PATH = REPO_ROOT / "data" / "chembl_selfies.txt"
OUT_DIR   = REPO_ROOT / "outputs" / "bpe_stats_sharded"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

## Outputs
Typical outputs (depending on your pipeline) include:
- `step_log.csv` (merge-step statistics)
- final vocabulary files (e.g., JSON/TSV)
- rank–frequency tables
- figure-ready CSV exports

## Reproducibility Notes
- Fix random seeds if your implementation samples data or uses nondeterministic sharding.
- Record corpus filtering rules and the exact merge-step grid used for statistics.

## Citation
If you use this repository in academic work, please cite the corresponding paper:
> The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations.

## License
Add a license before public release (e.g., MIT, Apache-2.0, or CC BY-NC for research code).
