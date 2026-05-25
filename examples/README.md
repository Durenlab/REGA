# REGA Examples — naiveCD4T Dataset

This directory contains a complete, real-world example using
**naive CD4+ T-cell pseudobulk RNA-seq** data from the OneK1K cohort,
with reference ATAC-seq peaks derived from the 10x Genomics PBMC Multiome
public dataset.

## Data

| File | Size | Description |
|------|------|-------------|
| `data/naiveCD4T_pseudobulk_counts.csv.gz` | ~9.2 MB | Gene × sample raw count matrix (12,000+ genes × 981 pseudobulk samples). Each sample is aggregated UMI counts from one donor's naive CD4+ T cells. |
| `data/naiveCD4T_ref_peaks.bed.gz` | ~1.3 MB | Reference ATAC-seq peak set (100,350 peaks, hg38). Peaks were called from the 10x Genomics PBMC Multiome public dataset and used as reference regulatory elements. |
| `data/Metadata.csv` | 27 KB | Sample metadata: donor ID, age, sex. Used for covariate correction in `rega preprocess`. |

### Data sources

- **RNA-seq (expression matrix)**:
  OneK1K cohort naive CD4+ T-cell pseudobulk counts.
  Pseudobulk aggregation: UMI counts summed per donor per cell type.

  > Yazar S, Alquicira-Hernandez J, Wing K, et al.
  > Single-cell eQTL mapping identifies cell type–specific genetic control
  > of autoimmune disease.
  > *Science*, 2022, 376(6589): eabf3041.
  > https://doi.org/10.1126/science.abf3041

- **ATAC-seq peaks (reference regulatory elements)**:
  Peak set derived from the 10x Genomics PBMC Multiome public dataset (hg38).
  Peaks were called and merged across samples to form a consensus reference
  peak set covering regulatory elements active in PBMCs.

  > 10x Genomics. PBMC from a Healthy Donor — Granulocytes Removed Through
  > Cell Sorting (10k) — Multiome ATAC + Gene Expression.
  > https://www.10xgenomics.com/datasets/

### Reference data (not included here)

The pipeline also requires reference files found in `reference_data/`:

| File | Description |
|------|-------------|
| `reference_data/TSS_hg38.txt` | Transcription start sites for hg38 (4-col TSV) |
| `reference_data/all_motif_rmdup.txt` | HOMER motif database (deduplicated) |
| `reference_data/Motif_TF_human.txt` | Motif-to-TF mapping for human |

## Usage

### Option 1: One-shot pipeline (recommended)

```bash
# Step 0: decompress the peaks BED (required — rega peaks cannot read .gz)
gunzip -c examples/data/naiveCD4T_ref_peaks.bed.gz > /tmp/naiveCD4T_ref_peaks.bed

# Edit examples/pipeline_config.yaml: set peak_bed to /tmp/naiveCD4T_ref_peaks.bed
#   and set reference.motif_file to your HOMER motif database path.

# Run full pipeline
rega pipeline --config examples/pipeline_config.yaml
```

### Option 2: Step-by-step

See [docs/tutorial.md](../docs/tutorial.md) for a complete step-by-step walkthrough.

## Expected outputs

After a full run, `examples/run_naiveCD4T/` will contain:

```
run_naiveCD4T/
├── naiveCD4T_named.bed          # peaks with chr_start_end names
├── naiveCD4T_peak_motif.txt     # peak × motif binding table
├── naiveCD4T_gex.csv            # preprocessed expression matrix
├── naiveCD4T_re_tg.txt          # RE-TG pair table
├── naiveCD4T_rega_input.h5ad    # assembled input AnnData
├── model/
│   └── final.pt                 # trained model checkpoint
└── naiveCD4T_rega_result.h5ad   # final result AnnData (1-2 GB)
```

The `rega_result.h5ad` file is the primary output used by all
`rega analysis` subcommands.
