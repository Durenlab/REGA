# REGA Tutorial

This tutorial walks through a complete REGA analysis using the
naiveCD4T example data (`examples/data/`).

---

## 1. Overview

REGA is an **interpretable hierarchical network representation learning
framework** for reference regulatory element–guided gene expression analysis.

Given a gene expression matrix and a reference peak set, REGA:
1. Links peaks to nearby genes via TSS-window overlap (RE-TG pairs)
2. Annotates peaks with transcription factor binding motifs (HOMER)
3. Learns regulatory modules connecting TFs, regulatory elements, and target genes
4. Extracts interpretable regulatory networks: RE→gene, TF→gene


### Supported data types

REGA accepts gene expression matrices from the following experimental designs.
The expression unit must be one of: `counts`, `cpm`, `logcpm`, `tpm`, or `logtpm`
(set via `--input-type`).

| Data type | Matrix dimensions | Recommended unit |
|-----------|-------------------|------------------|
| **Bulk RNA-seq** | gene × donor/sample | TPM or log2(TPM+1) |
| **scRNA-seq / snRNA-seq** | gene × cell-type-specific pseudobulk sample | counts, CPM, or log2(CPM+1) |
| **Spatial RNA-seq** | gene × spatial spot | counts, CPM, or log2(CPM+1) |
| **Perturb-seq** | gene × perturbation-target pseudobulk | counts, CPM, or log2(CPM+1) |

### Pipeline structure

| Step | CLI command | Module | Input → Output |
|------|-------------|--------|----------------|
| 1 | `rega peaks` | `rega.peaks` | raw BED → named BED |
| 2 | `rega motif` | `rega.motif` | named BED → peak_motif.txt |
| 3 | `rega preprocess` | `rega.preprocessing` | count matrix → log2(CPM+1) CSV |
| 4 | `rega retg` | `rega.retg` | named BED + TSS → RE-TG TSV |
| 5+6 | `rega build-input` | `rega.matrices` + `rega.anndata_builder` | RE-TG + GEX + motif → h5ad |
| 7+8 | `rega train` | `rega.blocks` + `rega.model` | h5ad → final.pt |
| 9 | `rega assemble` | `rega.results` | h5ad + final.pt → result h5ad |

---

## 2. Data preparation

### 2.1 Gene expression matrix

**Format**: CSV with gene symbols as row index, sample IDs as column headers.

```
         sample_1   sample_2   sample_3
GENE_A   100        250        0
GENE_B   5          8          12
...
```

The example file `examples/data/naiveCD4T_pseudobulk_counts.csv.gz` contains
raw pseudobulk UMI counts (981 GTEx donors, 12,000+ genes). REGA reads `.gz`
files natively for the expression matrix.

### 2.2 Peak BED file

**Format**: 3-column BED (chrom, start, end). Additional columns are ignored.

```
chr1    10000    10500
chr1    20100    20800
...
```

The example file `examples/data/naiveCD4T_ref_peaks.bed.gz` contains 100,350
ATAC-seq peaks (hg38).

> **Important**: `rega peaks` reads the BED file with plain text I/O and
> **does not support `.gz` files**. Decompress before use:
> ```bash
> gunzip -c examples/data/naiveCD4T_ref_peaks.bed.gz > /tmp/naiveCD4T_ref_peaks.bed
> ```
> Similarly, `bedtools` (used by `rega retg`) requires uncompressed input.

### 2.3 Sample metadata (optional)

**Format**: CSV where the first column is sample ID matching expression columns.
Remaining columns are treated as covariates for regression.

The example file `examples/data/Metadata.csv` contains donor age, sex,
and race/ethnicity for the 981 GTEx samples.

### 2.4 Reference data

| File | Location | Auto-detected? |
|------|----------|----------------|
| TSS coordinates | `reference_data/TSS_hg38.txt` | Yes, via `ReferenceData("hg38")` |
| HOMER motif database | `reference_data/all_motif_rmdup.txt` | No — must provide |
| Motif-TF mapping | `reference_data/Motif_TF_human.txt` | Optional |

---

## 3. Step-by-step usage

All examples use absolute paths for clarity. Adjust to your environment.

```bash
# Set a working directory for outputs
WORKDIR=/path/to/my_rega_run
mkdir -p $WORKDIR

# Decompress peaks BED once
gunzip -c examples/data/naiveCD4T_ref_peaks.bed.gz > $WORKDIR/ref_peaks.bed
```

### 3.1 `rega peaks` — Peak renaming

Adds a `chr_start_end` name column to a BED file so peaks can be
referenced by name in downstream steps.

**CLI**:

```bash
rega peaks $WORKDIR/ref_peaks.bed \
           $WORKDIR/naiveCD4T_named.bed
```

**Python API**:

```python
from rega.peaks import rename_peaks
rename_peaks("ref_peaks.bed", "naiveCD4T_named.bed")
```

**Output**: 4-column BED — `chrom  start  end  chr_start_end`

---

### 3.2 `rega motif` — HOMER motif scan

Scans peaks for TF binding motifs using HOMER `findMotifsGenome.pl`.
Produces a 2-column TSV: `peak_name  motif_name`.

> This step can take **hours** for 100k+ peaks. Use `--skip-motif` in
> `rega pipeline` to reuse an existing result.

**CLI**:

```bash
rega motif $WORKDIR/naiveCD4T_named.bed \
           --genome hg38 \
           --motif-db reference_data/all_motif_rmdup.txt \
           --output-dir $WORKDIR \
           --prefix naiveCD4T \
           --threads 8
```

**Python API**:

```python
from rega.motif import run_motif_scan
run_motif_scan(
    named_bed="naiveCD4T_named.bed",
    genome="hg38",
    motif_file="reference_data/all_motif_rmdup.txt",
    output_dir="output/",
    output_prefix="naiveCD4T",
    threads=8,
)
```

**Output**: `{output_dir}/{prefix}_peak_motif.txt` — tab-separated, no header

---

### 3.3 `rega preprocess` — Expression preprocessing

Normalizes and filters the raw expression matrix. Steps:
1. Convert to logCPM (or pass through if already log-scaled)
2. Remove low-expression genes (expressed in < 5% of samples by default)
3. Remove mitochondrial genes (`MT-` prefix)
4. Optionally regress out covariates (age, sex, etc.)

**CLI**:

```bash
rega preprocess examples/data/naiveCD4T_pseudobulk_counts.csv.gz \
    --output $WORKDIR/naiveCD4T_gex.csv \
    --meta examples/data/Metadata.csv \
    --input-type counts
```

**Python API**:

```python
from rega.preprocessing import preprocess_expression
df = preprocess_expression(
    expr_file="examples/data/naiveCD4T_pseudobulk_counts.csv.gz",
    meta_file="examples/data/Metadata.csv",
    input_type="counts",
)
df.to_csv("naiveCD4T_gex.csv")
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--input-type` | `counts` | `counts|cpm|logcpm|tpm|logtpm` |
| `--meta` | none | omit to skip covariate correction |
| `--min-expr-threshold` | auto | depends on `--input-type` |
| `--no-remove-mt` | false | keep mitochondrial genes |

**Output**: gene × sample CSV in logCPM (or logTPM) scale

---

### 3.4 `rega retg` — RE-TG pair building

For each gene, finds all peaks within a cis-regulatory window (default ±300 kb)
around the TSS using bedtools intersect.

**CLI**:

```bash
rega retg $WORKDIR/naiveCD4T_named.bed \
    --tss reference_data/TSS_hg38.txt \
    --output $WORKDIR/naiveCD4T_re_tg.txt \
    --window 300000
```

**Python API**:

```python
from rega.retg import build_re_tg
build_re_tg(
    tss_file="reference_data/TSS_hg38.txt",
    peaks_bed="naiveCD4T_named.bed",
    output_tsv="naiveCD4T_re_tg.txt",
    window=300_000,
)
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--window` | 300000 | bp each side of TSS (600 kb total) |
| `--chrom` | all | restrict to specific chromosomes |

**Output**: 10-column TSV with header — gene, TSS, peak coordinates, peak name

---

### 3.5 `rega build-input` — Input AnnData assembly

Combines the preprocessed expression, RE-TG pairs, and peak-motif table
into a single AnnData file (`rega_input.h5ad`) ready for training.

Internally: builds sparse W (gene × peak) and B (peak × motif) matrices,
then assembles the AnnData. Intermediate matrix files are written to a
temporary directory and cleaned up automatically.

**CLI**:

```bash
rega build-input \
    --gex       $WORKDIR/naiveCD4T_gex.csv \
    --re-tg     $WORKDIR/naiveCD4T_re_tg.txt \
    --peak-motif $WORKDIR/naiveCD4T_peak_motif.txt \
    --output    $WORKDIR/naiveCD4T_rega_input.h5ad \
    --motif-tf  reference_data/Motif_TF_human.txt
```

**Python API**:

```python
import tempfile, shutil
from rega.matrices import build_rega_input
from rega.anndata_builder import build_rega_anndata

with tempfile.TemporaryDirectory() as tmpdir:
    mat = build_rega_input(
        gex_csv="naiveCD4T_gex.csv",
        re_tg_file="naiveCD4T_re_tg.txt",
        peak_motif_file="naiveCD4T_peak_motif.txt",
        output_dir=tmpdir,
    )
    build_rega_anndata(
        re_tg_h5=mat["re_tg_h5"],
        gex_csv=mat["gex_csv"],
        binding_npz=mat["binding_npz"],
        binding_peaks_txt=mat["binding_peaks_txt"],
        binding_motifs_txt=mat["binding_motifs_txt"],
        annotation_file="naiveCD4T_re_tg.txt",
        output_h5ad="naiveCD4T_rega_input.h5ad",
        motif_tf_file="reference_data/Motif_TF_human.txt",
    )
```

**Output**: `rega_input.h5ad` — AnnData with X (W matrix), obs (genes),
var (peaks), obsm (GEX), varm (B matrix), uns (motif/TF names)

---

### 3.6 `rega train` — Model training

> ⚠️ **GPU strongly recommended**
>
> REGA model training is computationally intensive and **strongly
> recommended to run on a GPU**. CPU training is functional but
> can be **10–100× slower** for real-world datasets (typically
> hours on GPU vs. days on CPU).
>
> Before training, verify GPU availability:
> ```bash
> python -c "import torch; print('CUDA:', torch.cuda.is_available())"
> ```
> Expected output: `CUDA: True`
>
> If running on HPC, allocate a GPU node before invoking `rega train`:
> ```bash
> # SLURM example
> srun -p gpu --gres=gpu:1 --time=8:00:00 --pty bash
> # then activate your environment and run training
> conda activate rega
> rega train naiveCD4T_rega_input.h5ad --output-dir model/
> ```

Runs two-phase multiplicative-update training on the prepared AnnData.
Per-chromosome blocks are split internally and cleaned up after training
(use `--keep-blocks` to retain them).

**CLI**:

```bash
rega train $WORKDIR/naiveCD4T_rega_input.h5ad \
    --output-dir $WORKDIR/model/ \
    --K 20
```

**With custom hyperparameters**:

```bash
rega train $WORKDIR/naiveCD4T_rega_input.h5ad \
    --output-dir $WORKDIR/model/ \
    --K 20 \
    --phase2-min-outer 5000 \
    --patience 100
```

**Python API**:

```python
from rega.blocks import load_and_check, split_and_save
from rega.model import load_blocks, train_rega, TrainingConfig
import tempfile, shutil

config = TrainingConfig(K=20)

with tempfile.TemporaryDirectory() as blocks_dir:
    adata = load_and_check("naiveCD4T_rega_input.h5ad")
    split_and_save(adata, output_dir=blocks_dir)
    block_data = load_blocks(blocks_dir)
    train_rega(block_data, output_dir="model/", config=config)
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--K` | 20 | Number of regulatory modules |
| `--device` | auto | `cuda` or `cpu` |
| `--keep-blocks` | false | retain per-chrom .npz in `output-dir/blocks/` |
| `--phase2-min-outer` | 3000 | minimum Phase 2 iterations before early stopping |
| `--patience` | 50 | outer iterations without improvement to trigger stop |

**Output**: `output-dir/final.pt` (PyTorch checkpoint with W, B, V, H tensors)

---

### 3.7 `rega assemble` — Result h5ad collection

Loads the training checkpoint, assembles learned matrices, computes
downstream regulatory quantities, and writes `rega_result.h5ad`.

**CLI**:

```bash
rega assemble \
    --input-h5ad $WORKDIR/naiveCD4T_rega_input.h5ad \
    --result     $WORKDIR/model/final.pt \
    --output     $WORKDIR/naiveCD4T_rega_result.h5ad
```

**Python API**:

```python
from rega.results import collect_results
collect_results(
    input_h5ad="naiveCD4T_rega_input.h5ad",
    rega_result_path="model/final.pt",
    out_h5ad="naiveCD4T_rega_result.h5ad",
)
```

**Output**: `rega_result.h5ad` — complete result AnnData (see Section 6).

---

## 4. Downstream analysis

All analysis subcommands take `rega_result.h5ad` as input.

```bash
RESULT=$WORKDIR/naiveCD4T_rega_result.h5ad
```

### 4.1 `rega analysis modules` — Module assignment

Assigns genes, peaks, motifs, and samples to regulatory modules.

```bash
rega analysis modules $RESULT \
    --output-dir $WORKDIR/analysis/modules/ \
    --entity-type all \
    --assignment-type hard
```

**Options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--entity-type` | `all` | `gene|peak|motif|sample|all` |
| `--assignment-type` | `hard` | `hard` = unique module; `soft` = multi-module; `both` = run both |
| `--topk-frac` | 0.1 | top-K fraction per module for initial selection |
| `--p-abs` | 0.25 | absolute score threshold for label assignment |

**Python API**:

```python
import anndata as ad
from rega.analysis import assign_modules, load_entity_matrices, write_module_assignments

adata = ad.read_h5ad("naiveCD4T_rega_result.h5ad")
matrices = load_entity_matrices(adata)
result = assign_modules(matrices["gene"], assignment_type="hard")
write_module_assignments(result, prefix="gene", out_dir="modules/gene/hard/")
```

**Outputs** (per entity type and assignment type):
- `{prefix}_module_assignments.tsv` — long format (entity_name, module_id)
- `{prefix}_module_sets.tsv` — GMT-like format (module → members)
- `{prefix}_module_counts.tsv` — entity count per module

---

### 4.2 `rega analysis grn` — GRN extraction

Extracts long-format RE→gene and TF→gene regulatory network tables.

```bash
rega analysis grn $RESULT \
    --output-dir $WORKDIR/analysis/grn/
```

**Python API**:

```python
from rega.analysis import extract_grn
extract_grn(adata, out_dir="analysis/grn/")
```

**Outputs**:
- `RE_TG.tsv` — columns: target_gene, regulatory_element, score
- `TF_TG.tsv` — columns: target_gene, TF, score

---

### 4.3 `rega analysis re-score` — RE importance

Exports peak-level regulatory importance scores.

```bash
rega analysis re-score $RESULT \
    --output $WORKDIR/analysis/RE_importance.tsv
```

**Python API**:

```python
from rega.analysis import write_re_importance
write_re_importance(adata, out_tsv="analysis/RE_importance.tsv")
```

**Output**: 2-column TSV (regulatory_element, RE_importance), sorted descending.

---

### 4.4 `rega analysis disease` — Disease-associated modules

Identifies modules whose sample-level activity correlates with a binary phenotype label
using Spearman correlation + BH-FDR correction.

```bash
rega analysis disease $RESULT \
    --label $WORKDIR/disease_labels.csv \
    --output-dir $WORKDIR/analysis/disease/
```

The label CSV must have columns `sample_id` (matching `GEX_sample_names`)
and `disease_label` (0 = control, 1 = case):

```
sample_id,disease_label
GTEX-1117F,0
GTEX-111CU,1
...
```

**Python API**:

```python
import pandas as pd
from rega.analysis import identify_disease_modules, write_disease_module_results

label_df = pd.read_csv("disease_labels.csv")
result = identify_disease_modules(adata, label_df, fdr_cutoff=0.05)
write_disease_module_results(result, out_dir="analysis/disease/")
```

**Outputs**:
- `disease_modules_all.tsv` — all modules with Spearman ρ, p-value, FDR
- `disease_modules_significant.tsv` — FDR-significant modules only

---

### 4.5 `rega analysis driver-tf` — Disease driver TFs

Identifies TFs whose activity correlates with the phenotype label.

```bash
rega analysis driver-tf $RESULT \
    --label $WORKDIR/disease_labels.csv \
    --output-dir $WORKDIR/analysis/driver_tf/
```

**Python API**:

```python
from rega.analysis import identify_driver_tfs, write_driver_tf_results

result = identify_driver_tfs(adata, label_df, pcc_cutoff=0.1, fdr_cutoff=0.05)
write_driver_tf_results(result, out_dir="analysis/driver_tf/")
```

**Outputs**:
- `driver_tfs_all.tsv` — all TFs with Spearman ρ, p-value, FDR
- `driver_tfs_significant.tsv` — FDR-significant driver TFs only

---

## 5. One-shot pipeline

Run the complete 7-step pipeline from a single YAML config file.

### 5.1 Config structure

See `examples/pipeline_config.yaml` for a fully annotated example with all
default values documented.

Key sections:

```yaml
data:
  expression_csv: "path/to/expr.csv.gz"    # gzip supported
  peak_bed:       "path/to/peaks.bed"      # must be uncompressed
  meta_csv:       "path/to/metadata.csv"   # optional
  input_type:     "counts"

reference:
  genome:     "hg38"
  motif_file: "path/to/homer.motif"        # required unless --skip-motif

output:
  base_dir: "run_output/"
  prefix:   "my_sample"

train:
  K:      20
  device: null    # null = auto-detect GPU
```

### 5.2 Run

```bash
rega pipeline --config examples/pipeline_config.yaml
```

### 5.3 Skip and resume options

```bash
# Skip HOMER scan (use existing peak_motif.txt)
rega pipeline --config config.yaml --skip-motif

# Skip training (use existing final.pt)
rega pipeline --config config.yaml --skip-train

# Resume from a specific step (all prior outputs must exist)
rega pipeline --config config.yaml --resume-from train

# Keep per-chromosome blocks and build-input intermediates
rega pipeline --config config.yaml --keep-intermediate
```

The `--resume-from` option accepts:
`peaks | motif | preprocess | retg | build-input | train | assemble`

### 5.4 Example

```bash
# First run (full pipeline)
rega pipeline --config examples/pipeline_config.yaml

# Re-run only training + assembly (motif scan took 3 hours)
rega pipeline --config examples/pipeline_config.yaml \
    --resume-from train
```

---

## 6. Output structure (`rega_result.h5ad`)

The result AnnData encodes the complete REGA model output.

**Dimensions**: `n_obs` = genes, `n_vars` = peaks, `K` = modules (default 20)

| Slot | Key | Shape | Description |
|------|-----|-------|-------------|
| `obsm` | `GEX` | gene × sample | Preprocessed expression matrix |
| `obsm` | `WBV` | gene × K | Gene regulatory module scores |
| `obsm` | `WBV_norm` | gene × K | L1-normalized gene scores |
| `obsm` | `Net_TF_TG` | gene × TF | TF→gene regulatory network (expression-weighted) |
| `varm` | `BV` | peak × K | Peak regulatory module scores |
| `varm` | `BV_norm` | peak × K | L1-normalized peak scores |
| `varm` | `B_learned` | peak × motif | Learned motif binding weights |
| `varm` | `Peak_Motif` | peak × motif | Initial binary binding matrix (from HOMER) |
| `varm` | `Peak_Activity_BVH` | peak × sample | Peak activity across samples |
| `uns` | `H` | K × sample | Module activity matrix |
| `uns` | `H_T` | sample × K | Transposed module activity |
| `uns` | `H_T_norm` | sample × K | L1-normalized sample module scores |
| `uns` | `V` | motif × K | Learned motif module embeddings |
| `uns` | `V_norm` | motif × K | L1-normalized motif scores |
| `uns` | `Motif_Activity_VH` | motif × sample | Motif activity across samples |
| `uns` | `Motif_TF` | motif × TF | Binary motif-TF association |
| `uns` | `Motif_TF_PCC` | motif × TF | Pearson correlation (motif activity vs TF expression) |
| `uns` | `Module_names` | K | Module labels (Module_1 … Module_K) |
| `uns` | `motif_names` | M | Motif identifiers |
| `uns` | `TF_names` | T | TF gene symbols |
| `uns` | `GEX_sample_names` | N | Sample identifiers |
| `layers` | `W0` | gene × peak | Initial (distance-decay) W matrix |
| `layers` | `Net_RE_TG` | gene × peak | Learned RE→gene regulatory network |
| `var` | `RE_importance` | peak | Peak-level importance score |
| `var` | `peak_chrom`, `peak_start`, `peak_end` | peak | Genomic coordinates |
| `obs` | (index) | gene | Gene symbols |
