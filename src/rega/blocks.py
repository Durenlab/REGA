from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

__all__ = ["load_and_check", "get_common_chroms", "split_and_save"]


def load_and_check(h5ad_path: Path) -> ad.AnnData:
    """Load REGA AnnData and validate required slots and dimension consistency.

    Raises ValueError if required slots are missing or dimensions are inconsistent.
    """
    adata = ad.read_h5ad(h5ad_path)

    if not sparse.issparse(adata.X):
        raise ValueError(f"adata.X must be sparse, got {type(adata.X)}")

    if "GEX" not in adata.obsm:
        raise ValueError("adata.obsm['GEX'] missing.")
    if "Peak_Motif" not in adata.varm:
        raise ValueError("adata.varm['Peak_Motif'] missing.")
    if "GEX_sample_names" not in adata.uns:
        raise ValueError("adata.uns['GEX_sample_names'] missing.")
    if "motif_names" not in adata.uns:
        raise ValueError("adata.uns['motif_names'] missing.")

    if "chrom" not in adata.obs.columns:
        raise ValueError("adata.obs missing 'chrom' column.")
    if "peak_chrom" not in adata.var.columns:
        raise ValueError("adata.var missing 'peak_chrom' column.")

    if adata.X.shape[0] != adata.obsm["GEX"].shape[0]:
        raise ValueError(
            f"X rows ({adata.X.shape[0]}) != obsm['GEX'] rows "
            f"({adata.obsm['GEX'].shape[0]})"
        )
    if adata.X.shape[1] != adata.varm["Peak_Motif"].shape[0]:
        raise ValueError(
            f"X cols ({adata.X.shape[1]}) != varm['Peak_Motif'] rows "
            f"({adata.varm['Peak_Motif'].shape[0]})"
        )
    if adata.obsm["GEX"].shape[1] != len(adata.uns["GEX_sample_names"]):
        raise ValueError(
            f"obsm['GEX'] cols ({adata.obsm['GEX'].shape[1]}) "
            f"!= uns['GEX_sample_names'] ({len(adata.uns['GEX_sample_names'])})"
        )
    if adata.varm["Peak_Motif"].shape[1] != len(adata.uns["motif_names"]):
        raise ValueError(
            f"varm['Peak_Motif'] cols ({adata.varm['Peak_Motif'].shape[1]}) "
            f"!= uns['motif_names'] ({len(adata.uns['motif_names'])})"
        )

    print(
        f"[load_and_check] {adata.X.shape[0]} genes, {adata.X.shape[1]} peaks, "
        f"{adata.obsm['GEX'].shape[1]} samples, {adata.varm['Peak_Motif'].shape[1]} motifs"
    )
    return adata


def get_common_chroms(adata: ad.AnnData) -> list[str]:
    """Return sorted intersection of gene and peak chromosomes.

    Sort order: chr1–chr22, chrX, chrY, chrM/chrMT, other.
    """
    chrs_g = set(adata.obs["chrom"].astype(str).unique())
    chrs_r = set(adata.var["peak_chrom"].astype(str).unique())
    common = chrs_g & chrs_r

    def chrom_key(c: str) -> tuple:
        s = c.replace("chr", "")
        if s.isdigit():
            return (0, int(s), "")
        if s == "X":
            return (1, 0, "")
        if s == "Y":
            return (2, 0, "")
        if s in ("M", "MT"):
            return (3, 0, "")
        return (4, 0, s)

    return sorted(common, key=chrom_key)


def split_and_save(
    adata: ad.AnnData,
    output_dir: Path,
    chroms: list[str] | None = None,
) -> int:
    """Split AnnData into per-chromosome .npz blocks and write to output_dir.

    Args:
        adata: validated REGA AnnData (from load_and_check).
        output_dir: directory for {chrom}.npz files; caller must ensure it exists.
        chroms: whitelist of chromosomes to write; None means all common chromosomes.

    Returns:
        Number of block files written.

    Raises:
        ValueError if no blocks are written (likely a chromosome name mismatch).
    """
    common_chrs = get_common_chroms(adata)
    process_chrs = [c for c in common_chrs if c in chroms] if chroms is not None else common_chrs

    genes_all = np.asarray(adata.obs_names, dtype=np.str_)
    peaks_all = np.asarray(adata.var_names, dtype=np.str_)
    samples_all = np.asarray(adata.uns["GEX_sample_names"], dtype=np.str_)
    motifs_all = np.asarray(adata.uns["motif_names"], dtype=np.str_)

    gene_chrom_all = np.asarray(adata.obs["chrom"].astype(str).values, dtype=np.str_)
    peak_chrom_all = np.asarray(adata.var["peak_chrom"].astype(str).values, dtype=np.str_)

    gex_all = np.asarray(adata.obsm["GEX"])
    binding_all = np.asarray(adata.varm["Peak_Motif"])

    W_sparse = adata.X.tocsr() if not sparse.isspmatrix_csr(adata.X) else adata.X

    written = 0
    skipped = 0

    for chr_name in process_chrs:
        gmask = gene_chrom_all == chr_name
        rmask = peak_chrom_all == chr_name

        if not gmask.any() or not rmask.any():
            skipped += 1
            continue

        X_blk = gex_all[gmask, :].astype(np.float32)
        W_blk = W_sparse[gmask, :][:, rmask].toarray().astype(np.float32)
        B_blk = binding_all[rmask, :].astype(np.float32)

        if X_blk.size == 0 or W_blk.size == 0 or B_blk.size == 0:
            skipped += 1
            continue

        np.savez_compressed(
            str(output_dir / f"{chr_name}.npz"),
            X=X_blk,
            W=W_blk,
            B=B_blk,
            genes=np.asarray(genes_all[gmask], dtype=np.str_),
            RE=np.asarray(peaks_all[rmask], dtype=np.str_),
            samples=samples_all,
            motifs=motifs_all,
            gene_chr=np.asarray(gene_chrom_all[gmask], dtype=np.str_),
            re_chr=np.asarray(peak_chrom_all[rmask], dtype=np.str_),
        )
        print(f"[blocks] {chr_name}: X{X_blk.shape} W{W_blk.shape} B{B_blk.shape}")
        written += 1

    if skipped > 0:
        print(f"[blocks] WARNING: skipped {skipped} empty chromosomes")

    if written == 0:
        obs_sample = sorted(set(adata.obs["chrom"].astype(str).unique()))[:5]
        var_sample = sorted(set(adata.var["peak_chrom"].astype(str).unique()))[:5]
        raise ValueError(
            "No chromosome blocks written. Possible causes:\n"
            "  - Chromosome name mismatch between obs['chrom'] and var['peak_chrom']\n"
            "  - Empty intersection of chromosomes across genes/peaks\n"
            f"  Sample obs chromosomes: {obs_sample}\n"
            f"  Sample var chromosomes: {var_sample}"
        )

    print(f"[blocks] wrote {written} chromosome blocks to {output_dir}")
    return written
