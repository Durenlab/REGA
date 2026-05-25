# REGA

> Reference **R**egulatory **E**lement-Guided **G**ene Expression **A**nalysis
> for Mechanistic Inference of Gene Regulatory Networks

<!-- Badges: fill after GitHub + PyPI release -->
<!-- ![PyPI version](TODO) ![License](TODO) ![Tests](TODO) -->

## Overview

REGA is an **interpretable hierarchical network representation learning framework**
for reference regulatory element–guided gene expression analysis.

## Key Features

- **End-to-end pipeline** — raw counts + peaks → regulatory networks in one command
- **Modular CLI** — each step independently runnable (`rega peaks`, `rega motif`, …)
- **One-shot mode** — full pipeline from a single YAML config file
- **AnnData-based result** — integrates with scanpy, muon, and the broader ecosystem
- **Supported data types** — bulk RNA-seq, scRNA-seq/snRNA-seq pseudobulk, spatial RNA-seq, Perturb-seq
- **Built-in downstream analysis** — module assignment, GRN extraction, disease module identification, driver TF discovery
- **HPC-ready** — GPU training, SLURM examples, long-job guidance

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.9 | Recommended via conda |
| **PyTorch** (GPU build strongly recommended) | Required for model training |
| **CUDA-capable GPU** | CPU training is 10–100× slower |
| bedtools ≥ 2.30 | For RE-TG pair building |
| HOMER (with genome package) | For motif scanning |

See [docs/installation.md](docs/installation.md) for full setup instructions.

## Installation

REGA is currently distributed via GitHub. Install in 4 steps:

```bash
# 1. Clone the repository
git clone https://github.com/Lixin017/REGA.git
cd REGA

# 2. Create conda environment
conda env create -f environment.yml          # GPU/CPU-agnostic base environment
# or for a CPU-only named environment:
# conda env create -f environment-cpu.yml

# 3. Activate environment
conda activate rega

# 4. Install PyTorch — choose the build that matches your hardware
# Visit https://pytorch.org/get-started/locally/ for the exact command.
# Examples:
#   GPU (CUDA 12.1):  pip install torch --index-url https://download.pytorch.org/whl/cu121
#   GPU (CUDA 12.4):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   CPU-only:         pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install REGA
pip install .
```

### Install HOMER (separately)

HOMER cannot be installed via conda. Follow the official guide:
http://homer.ucsd.edu/homer/introduction/install.html

After installation, install the genome package:
```bash
configureHomer.pl -install hg38
```

### Verify
```bash
rega --version
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

For HPC tips and troubleshooting, see [docs/installation.md](docs/installation.md).

## Quick Start

```bash
# Option 1: One-shot pipeline (recommended)
# First decompress the peaks BED (required — cannot read .gz)
gunzip -c examples/data/naiveCD4T_ref_peaks.bed.gz > /tmp/naiveCD4T_ref_peaks.bed

# Edit examples/pipeline_config.yaml to set peak_bed and motif_file paths,
# then run:
rega pipeline --config examples/pipeline_config.yaml

# Option 2: Step-by-step
rega peaks   /tmp/naiveCD4T_ref_peaks.bed  output/naiveCD4T_named.bed
rega motif   output/naiveCD4T_named.bed \
             --genome hg38 \
             --motif-db reference_data/all_motif_rmdup.txt \
             --output-dir output/ --prefix naiveCD4T
rega preprocess examples/data/naiveCD4T_pseudobulk_counts.csv.gz \
             --output output/naiveCD4T_gex.csv \
             --meta   examples/data/Metadata.csv \
             --input-type counts
rega retg    output/naiveCD4T_named.bed \
             --tss    reference_data/TSS_hg38.txt \
             --output output/naiveCD4T_re_tg.txt
rega build-input \
             --gex        output/naiveCD4T_gex.csv \
             --re-tg      output/naiveCD4T_re_tg.txt \
             --peak-motif output/naiveCD4T_peak_motif.txt \
             --output     output/naiveCD4T_rega_input.h5ad \
             --motif-tf   reference_data/Motif_TF_human.txt
rega train   output/naiveCD4T_rega_input.h5ad \
             --output-dir output/model/ --K 20
rega assemble \
             --input-h5ad output/naiveCD4T_rega_input.h5ad \
             --result     output/model/final.pt \
             --output     output/naiveCD4T_rega_result.h5ad
```

## Downstream Analysis

```bash
RESULT=output/naiveCD4T_rega_result.h5ad

# Assign entities to regulatory modules
rega analysis modules $RESULT --output-dir analysis/modules/

# Extract GRN tables (RE→gene, TF→gene)
rega analysis grn $RESULT --output-dir analysis/grn/

# RE importance scores
rega analysis re-score $RESULT --output analysis/RE_importance.tsv

# Disease-associated modules (requires label CSV)
rega analysis disease $RESULT --label disease_labels.csv --output-dir analysis/disease/

# Driver TF identification
rega analysis driver-tf $RESULT --label disease_labels.csv --output-dir analysis/driver_tf/
```

## Documentation

- [Installation guide](docs/installation.md)
- [Step-by-step tutorial](docs/tutorial.md)
- API reference: `help(rega.<module>)` or docstrings in `src/rega/`

## Citation

If you use REGA in your research, please cite:

```bibtex
@article{rega2026,
  title   = {TODO: fill after publication},
  author  = {Ren, Lixin and ...},
  journal = {TODO},
  year    = {2026},
  doi     = {TODO},
}
```

## Author

**Lixin Ren**
Duren Lab, Indiana University School of Medicine
Email: rlxmath017@gmail.com

## License

MIT — see [LICENSE](LICENSE)
