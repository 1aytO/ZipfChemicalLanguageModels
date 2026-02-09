# The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations

This repository contains code to reproduce the analysis reported in the paper **“The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations”**.

Core tasks include:
- Building a data-driven molecular vocabulary from SELFIES sequences using **Byte-Pair Encoding (BPE)**
- Tracking vocabulary growth statistics across merge steps
- Verifying **Zipf-like scaling** in molecular token rank–frequency distributions
- Exporting intermediate logs and final vocabularies for downstream analysis and figure generation


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


## Reproducibility Notes
- Fix random seeds if your implementation samples data or uses nondeterministic sharding.
- Record corpus filtering rules and the exact merge-step grid used for statistics.

## Citation
If you use this repository in academic work, please cite the corresponding paper:
> The Linguistic Basis of Chemical Language Models: Zipf’s Law in Molecular Sequence Representations.

