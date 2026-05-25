from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix

__all__ = [
    "build_rega_anndata",
    "load_re_tg_matrix",
    "load_binding_sparse",
    "load_gene_peak_meta",
    "build_motif_tf_matrix",
]


def _decode(s) -> str:
    if isinstance(s, (bytes, np.bytes_)):
        return s.decode()
    return str(s)


def load_re_tg_matrix(
    h5_path: Path,
) -> tuple[csr_matrix, list[str], list[str]]:
    """Load W matrix from B4 HDF5; validate 0-based indexing; return (W, genes, peaks).

    Raises ValueError if attrs['indexing'] != '0-based' or row/col names contain duplicates.
    """
    with h5py.File(h5_path, "r") as h5:
        indexing = h5.attrs.get("indexing")
        if indexing != "0-based":
            raise ValueError(
                f"Expected HDF5 attrs['indexing']='0-based', got {indexing!r}. "
                "Re-generate re_tg_matrix.h5 with rega.matrices.build_rega_input."
            )
        i = h5["i"][:].astype(np.int64)
        j = h5["j"][:].astype(np.int64)
        x = h5["x"][:].astype(np.float64)
        dims = tuple(h5["dims"][:].astype(np.int64))
        genes = [_decode(s) for s in h5["rownames"][:]]
        peaks = [_decode(s) for s in h5["colnames"][:]]

    if len(set(genes)) != len(genes):
        n_dup = len(genes) - len(set(genes))
        raise ValueError(f"re_tg_matrix.h5 rownames contain {n_dup} duplicate gene names.")
    if len(set(peaks)) != len(peaks):
        n_dup = len(peaks) - len(set(peaks))
        raise ValueError(f"re_tg_matrix.h5 colnames contain {n_dup} duplicate peak names.")

    W = coo_matrix((x, (i, j)), shape=dims).tocsr()
    print(f"[load_re_tg_matrix] shape={W.shape}, nnz={W.nnz}")
    return W, genes, peaks


def load_binding_sparse(
    npz_path: Path,
    peaks_txt: Path,
    motifs_txt: Path,
) -> tuple[csr_matrix, list[str], list[str]]:
    """Load sparse B matrix from npz + companion label files; return (B, peak_names, motif_names).

    Raises ValueError if B.shape != (len(peak_names), len(motif_names)).
    The three files must be co-located (written together by build_rega_input).
    """
    B = scipy.sparse.load_npz(str(npz_path))
    b_peaks = peaks_txt.read_text().splitlines()
    motif_names = motifs_txt.read_text().splitlines()
    if B.shape != (len(b_peaks), len(motif_names)):
        raise ValueError(
            f"Binding matrix shape {B.shape} does not match label files: "
            f"({len(b_peaks)} peaks, {len(motif_names)} motifs). "
            "Ensure npz and companion .txt files are from the same build_rega_input run."
        )
    print(f"[load_binding_sparse] shape={B.shape}")
    return B, b_peaks, motif_names


def load_gene_peak_meta(
    annotation_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract gene and peak coordinate metadata from the B3 RE-TG annotation table.

    Returns (gene_meta, peak_meta) indexed by gene_name and peak_name respectively.
    Numeric coordinate columns are cast to Int64.
    """
    ann = pd.read_csv(annotation_file, sep="\t", header=0)
    ann.columns = [c.strip() for c in ann.columns]

    gene_cols = ["gene_name", "chrom", "region_start", "region_end", "TSS", "strand"]
    missing = [c for c in gene_cols if c not in ann.columns]
    if missing:
        raise ValueError(f"Annotation file missing gene columns: {missing}")

    peak_cols = ["peak_name", "peak_chrom", "peak_start", "peak_end"]
    missing = [c for c in peak_cols if c not in ann.columns]
    if missing:
        raise ValueError(f"Annotation file missing peak columns: {missing}")

    gene_meta = (
        ann[gene_cols]
        .drop_duplicates(subset="gene_name", keep="first")
        .set_index("gene_name")
    )
    peak_meta = (
        ann[peak_cols]
        .drop_duplicates(subset="peak_name", keep="first")
        .set_index("peak_name")
    )

    for col in ["region_start", "region_end", "TSS"]:
        gene_meta[col] = pd.to_numeric(gene_meta[col], errors="coerce").astype("Int64")
    for col in ["peak_start", "peak_end"]:
        peak_meta[col] = pd.to_numeric(peak_meta[col], errors="coerce").astype("Int64")

    print(f"[load_gene_peak_meta] genes={len(gene_meta)}, peaks={len(peak_meta)}")
    return gene_meta, peak_meta


def build_motif_tf_matrix(
    motif_tf_file: Path,
    motif_names: list[str],
    gene_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build motif×TF int8 binary matrix filtered to known motifs and genes.

    Returns (matrix, tf_names). Rows follow motif_names order exactly (missing rows are 0).
    Raises ValueError if no pairs survive filtering.
    """
    mt = pd.read_csv(motif_tf_file, sep="\t", header=None, names=["motif", "TF"])
    mt["motif"] = mt["motif"].astype(str).str.strip()
    mt["TF"] = mt["TF"].astype(str).str.strip()

    motif_set = set(motif_names)
    gene_set = set(gene_names)
    mt = mt[mt["motif"].isin(motif_set) & mt["TF"].isin(gene_set)]
    mt = mt.drop_duplicates(subset=["motif", "TF"])

    if len(mt) == 0:
        raise ValueError(
            "No motif-TF pairs survived filtering. "
            "Check that motif IDs match motif_names and TF IDs match gene_names."
        )

    tf_names = sorted(set(mt["TF"]))
    motif_to_idx = {m: i for i, m in enumerate(motif_names)}
    tf_to_idx = {t: i for i, t in enumerate(tf_names)}

    i_idx = mt["motif"].map(motif_to_idx).values.astype(int)
    j_idx = mt["TF"].map(tf_to_idx).values.astype(int)

    matrix = np.zeros((len(motif_names), len(tf_names)), dtype=np.int8)
    matrix[i_idx, j_idx] = 1

    print(f"[build_motif_tf_matrix] shape={matrix.shape}, nonzero={matrix.sum()}")
    return matrix, tf_names


def build_rega_anndata(
    re_tg_h5: Path,
    gex_csv: Path,
    binding_npz: Path,
    binding_peaks_txt: Path,
    binding_motifs_txt: Path,
    annotation_file: Path,
    output_h5ad: Path,
    motif_tf_file: Path | None = None,
) -> ad.AnnData:
    """Assemble a REGA AnnData from B4 outputs and the B3 annotation table.

    Inputs:
      re_tg_h5, gex_csv                    — B4 matrix outputs
      binding_npz, binding_peaks_txt,
      binding_motifs_txt                   — B4 sparse binding outputs (three co-located files)
      annotation_file                      — B3 RE-TG table (*_re_tg.txt); provides obs/var metadata
      motif_tf_file                        — optional 2-col TSV (no header): motif TAB TF

    AnnData structure:
      X                        sparse CSR gene×peak (W)
      obs                      gene_name index; chrom, region_start, region_end, TSS, strand
      var                      peak_name index; peak_chrom, peak_start, peak_end
      obsm["GEX"]              float64 dense  genes×samples
      varm["Peak_Motif"]       int8 dense     peaks×motifs
      uns["GEX_sample_names"]  sample name array
      uns["motif_names"]       motif name array
      uns["Motif_TF"]          int8 dense motifs×TFs  (only if motif_tf_file provided)
      uns["TF_names"]          TF name array          (only if motif_tf_file provided)

    Raises ValueError on: wrong HDF5 indexing scheme, label-count mismatch,
    any gene/peak present in W but absent from annotation/GEX/binding.
    Caller must ensure output_h5ad.parent exists.
    """
    print(f"[build_rega_anndata] re_tg_h5={re_tg_h5.name}, gex={gex_csv.name}")

    W, genes, peaks = load_re_tg_matrix(re_tg_h5)
    B, b_peaks, motif_names = load_binding_sparse(binding_npz, binding_peaks_txt, binding_motifs_txt)
    gene_meta, peak_meta = load_gene_peak_meta(annotation_file)
    gex = pd.read_csv(gex_csv, index_col=0)

    b_peak_set = set(b_peaks)
    missing_g = [g for g in genes if g not in gene_meta.index or g not in gex.index]
    if missing_g:
        raise ValueError(
            f"{len(missing_g)} gene(s) in W not found in annotation or GEX: {missing_g[:5]}"
        )
    missing_p = [p for p in peaks if p not in peak_meta.index or p not in b_peak_set]
    if missing_p:
        raise ValueError(
            f"{len(missing_p)} peak(s) in W not found in annotation or binding: {missing_p[:5]}"
        )

    gex_a = gex.loc[genes]
    b_peak_idx = {p: i for i, p in enumerate(b_peaks)}
    row_order = np.array([b_peak_idx[p] for p in peaks])
    B_a = B[row_order, :]

    obs = gene_meta.loc[genes]
    var = peak_meta.loc[peaks]

    adata = ad.AnnData(X=W, obs=obs, var=var)
    adata.obsm["GEX"] = gex_a.values.astype(np.float64)
    adata.varm["Peak_Motif"] = B_a.toarray().astype(np.int8)
    adata.uns["GEX_sample_names"] = np.array(list(gex_a.columns))
    adata.uns["motif_names"] = np.array(motif_names)

    if motif_tf_file is not None:
        mtx, tf_names = build_motif_tf_matrix(motif_tf_file, motif_names, genes)
        adata.uns["Motif_TF"] = mtx
        adata.uns["TF_names"] = np.array(tf_names)
        adata.uns["Motif_TF_description"] = (
            "Binary motif-by-TF match matrix; rows follow motif_names, columns follow TF_names"
        )

    print(f"[build_rega_anndata] X={adata.X.shape}, GEX={adata.obsm['GEX'].shape}, "
          f"Peak_Motif={adata.varm['Peak_Motif'].shape}")
    adata.write_h5ad(output_h5ad)
    print(f"[build_rega_anndata] written: {output_h5ad}")
    return adata
