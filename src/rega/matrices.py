from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix

from rega._version import __version__

__all__ = [
    "build_rega_input",
    "filter_common_features",
    "build_re_tg_matrix",
    "build_peak_motif_matrix",
    "sort_genes_by_coord",
    "sort_peaks_by_coord",
]


def _chrom_sort_key(chrom: str) -> tuple:
    """Chromosome sort key: chr1..22 < chrX < chrY < chrM/MT < other."""
    c = str(chrom).replace("chr", "")
    if c.isdigit():
        return (0, int(c), "")
    if c == "X":
        return (1, 0, "")
    if c == "Y":
        return (2, 0, "")
    if c in ("M", "MT"):
        return (3, 0, "")
    return (4, 0, c)


def sort_genes_by_coord(
    genes: list[str] | set[str],
    re_tg: pd.DataFrame,
) -> list[str]:
    """Sort gene names by (chromosome, TSS, name) using coordinates from re_tg.

    Genes absent from re_tg are sorted last (after all known chromosomes).
    When a gene appears multiple times in re_tg, the first occurrence is used.
    """
    gene_coord = (
        re_tg[["gene_name", "chrom", "TSS"]]
        .drop_duplicates(subset="gene_name", keep="first")
        .set_index("gene_name")
    )

    def _key(g: str) -> tuple:
        if g in gene_coord.index:
            row = gene_coord.loc[g]
            return (_chrom_sort_key(row["chrom"]), int(row["TSS"]), g)
        return ((9, 0, ""), 0, g)

    return sorted(genes, key=_key)


def sort_peaks_by_coord(
    peaks: list[str] | set[str],
) -> list[str]:
    """Sort peak names by (chromosome, start, name).

    Assumes peak name format 'chr_start_end' as produced by B1 rename_peaks().
    """

    def _key(p: str) -> tuple:
        parts = p.split("_")
        if len(parts) < 3:
            return ((9, 0, ""), 0, p)
        try:
            return (_chrom_sort_key(parts[0]), int(parts[1]), p)
        except ValueError:
            return ((9, 0, ""), 0, p)

    return sorted(peaks, key=_key)


def filter_common_features(
    expr: pd.DataFrame,
    re_tg: pd.DataFrame,
    peak_motif: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame, pd.DataFrame]:
    """Three-source intersection filter + genomic coordinate sort.

    Returns: (genes_sorted, peaks_sorted, re_tg_filtered, peak_motif_filtered).

    Filter order:
    1. genes:  GEX index ∩ RE-TG gene_name
    2. peaks:  RE-TG peak_name ∩ motif peak column
    3. genes re-filter: genes that lose all peaks after step 2 are excluded
    All returned lists are sorted by genomic coordinate.
    """
    genes_in_expr = set(expr.index)

    genes_valid = genes_in_expr & set(re_tg["gene_name"].unique())
    re_tg = re_tg[re_tg["gene_name"].isin(genes_valid)].copy()

    peaks_common = set(re_tg["peak_name"].unique()) & set(peak_motif["peak"].unique())
    re_tg = re_tg[re_tg["peak_name"].isin(peaks_common)].copy()
    peak_motif = peak_motif[peak_motif["peak"].isin(peaks_common)].copy()

    # Re-filter: some genes may have lost all their peaks after peak filtering.
    genes_valid = genes_in_expr & set(re_tg["gene_name"].unique())
    re_tg = re_tg[re_tg["gene_name"].isin(genes_valid)].copy()

    genes_sorted = sort_genes_by_coord(genes_valid, re_tg)
    peaks_sorted = sort_peaks_by_coord(peaks_common)

    print(
        f"[filter_common_features] genes={len(genes_sorted)}, "
        f"peaks={len(peaks_sorted)}, re_tg_rows={len(re_tg)}"
    )
    return genes_sorted, peaks_sorted, re_tg.reset_index(drop=True), peak_motif.reset_index(drop=True)


def build_re_tg_matrix(
    re_tg: pd.DataFrame,
    genes: list[str],
    peaks: list[str],
    tau: float = 1e5,
) -> csr_matrix:
    """Build gene×peak sparse CSR weight matrix W.

    Weight = exp(-|peak_midpoint - TSS| / tau).
    Duplicate (gene, peak) pairs are resolved by taking the max weight (defensive;
    B3 output does not produce duplicates under normal conditions).
    """
    peak_mid = (re_tg["peak_start"].values + re_tg["peak_end"].values) / 2.0
    dist = np.abs(peak_mid - re_tg["TSS"].values.astype(float))

    df = re_tg[["gene_name", "peak_name"]].copy()
    df["weight"] = np.exp(-dist / tau)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    df["i"] = df["gene_name"].map(gene_to_idx)
    df["j"] = df["peak_name"].map(peak_to_idx)
    df = df.dropna(subset=["i", "j"])
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    df = df.groupby(["i", "j"], as_index=False)["weight"].max()

    W = coo_matrix(
        (df["weight"].values, (df["i"].values, df["j"].values)),
        shape=(len(genes), len(peaks)),
    ).tocsr()

    print(f"[build_re_tg_matrix] shape={W.shape}, nnz={W.nnz}")
    return W


def build_peak_motif_matrix(
    peak_motif: pd.DataFrame,
    peaks: list[str],
) -> tuple[csr_matrix, list[str]]:
    """Build peak×motif sparse CSR binary matrix B (int8 0/1).

    Returns (matrix, motif_list) where motif_list is sorted(unique motif names).
    """
    motif_list = sorted(peak_motif["motif"].unique())
    peak_to_idx = {p: i for i, p in enumerate(peaks)}
    motif_to_idx = {m: i for i, m in enumerate(motif_list)}

    sub = peak_motif[
        peak_motif["peak"].isin(peak_to_idx) & peak_motif["motif"].isin(motif_to_idx)
    ]
    pairs = sub[["peak", "motif"]].drop_duplicates()
    i_idx = pairs["peak"].map(peak_to_idx).values.astype(int)
    j_idx = pairs["motif"].map(motif_to_idx).values.astype(int)

    B = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (i_idx, j_idx)),
        shape=(len(peaks), len(motif_list)),
    ).tocsr()

    print(f"[build_peak_motif_matrix] shape={B.shape}, nnz={B.nnz}")
    return B, motif_list


def build_rega_input(
    gex_csv: Path,
    re_tg_file: Path,
    peak_motif_file: Path,
    output_dir: Path,
    tau: float = 1e5,
) -> dict[str, Path]:
    """Build REGA input matrices from expression, RE-TG, and peak-motif data.

    Outputs written to output_dir (caller must ensure it exists):
      re_tg_matrix.h5          sparse gene×peak weight matrix (W); HDF5 with
                                0-based COO arrays; attrs: indexing="0-based"
      input_GEX.csv             filtered expression, rows in genomic coord order
      input_binding.npz         sparse peak×motif binary matrix (B); scipy npz
      input_binding_peaks.txt   one peak name per line  (row labels for B)
      input_binding_motifs.txt  one motif name per line (col labels for B)

    The three binding files must remain co-located.  B5 reconstruction:
      B      = scipy.sparse.load_npz("input_binding.npz")
      peaks  = Path("input_binding_peaks.txt").read_text().splitlines()
      motifs = Path("input_binding_motifs.txt").read_text().splitlines()

    peak_motif_file: 2-col TSV, no header — peak_name TAB motif_name.
    Format will be standardized in B7 (motif.py).

    Returns dict with keys: "re_tg_h5", "gex_csv", "binding_npz",
    "binding_peaks_txt", "binding_motifs_txt".
    """
    print(
        f"[build_rega_input] gex={gex_csv.name}, re_tg={re_tg_file.name}, "
        f"motif={peak_motif_file.name}, tau={tau:.0f}"
    )

    expr = pd.read_csv(gex_csv, index_col=0)
    re_tg = pd.read_csv(re_tg_file, sep="\t", header=0)
    re_tg.columns = [c.strip() for c in re_tg.columns]
    peak_motif = pd.read_csv(
        peak_motif_file, sep="\t", header=None, names=["peak", "motif"]
    )
    print(
        f"[build_rega_input] loaded: expr={expr.shape}, "
        f"re_tg={len(re_tg)}, motif_rows={len(peak_motif)}"
    )

    genes, peaks, re_tg_f, pm_f = filter_common_features(expr, re_tg, peak_motif)
    W = build_re_tg_matrix(re_tg_f, genes, peaks, tau=tau)
    B, motif_list = build_peak_motif_matrix(pm_f, peaks)

    re_tg_h5 = output_dir / "re_tg_matrix.h5"
    gex_out = output_dir / "input_GEX.csv"
    binding_npz = output_dir / "input_binding.npz"
    binding_peaks_txt = output_dir / "input_binding_peaks.txt"
    binding_motifs_txt = output_dir / "input_binding_motifs.txt"

    coo = W.tocoo()
    with h5py.File(re_tg_h5, "w") as h5:
        h5.attrs["indexing"] = "0-based"
        h5.attrs["created_by"] = f"rega.matrices {__version__}"
        h5.create_dataset("i",        data=coo.row.astype(np.int32))
        h5.create_dataset("j",        data=coo.col.astype(np.int32))
        h5.create_dataset("x",        data=coo.data.astype(np.float64))
        h5.create_dataset("dims",     data=np.array(W.shape, dtype=np.int32))
        h5.create_dataset("rownames", data=np.array(genes, dtype="S"))
        h5.create_dataset("colnames", data=np.array(peaks, dtype="S"))
    print(f"[build_rega_input] written: {re_tg_h5}")

    gex_filtered = expr.loc[expr.index.isin(set(genes))].reindex(genes)
    gex_filtered.to_csv(gex_out)
    print(f"[build_rega_input] written: {gex_out}")

    scipy.sparse.save_npz(str(binding_npz), B)
    binding_peaks_txt.write_text("\n".join(peaks) + "\n")
    binding_motifs_txt.write_text("\n".join(motif_list) + "\n")
    print(f"[build_rega_input] written: {binding_npz}")

    return {
        "re_tg_h5":           re_tg_h5,
        "gex_csv":            gex_out,
        "binding_npz":        binding_npz,
        "binding_peaks_txt":  binding_peaks_txt,
        "binding_motifs_txt": binding_motifs_txt,
    }
