from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from rega.anndata_builder import (
    build_rega_anndata,
    build_motif_tf_matrix,
    load_binding_sparse,
    load_gene_peak_meta,
    load_re_tg_matrix,
)
from rega.matrices import build_rega_input

MATRICES_FIXTURES = Path(__file__).parent / "fixtures" / "matrices"
ANNDATA_FIXTURES = Path(__file__).parent / "fixtures" / "anndata_builder"
MINI_ANNOTATION = MATRICES_FIXTURES / "mini_re_tg.txt"
MINI_MOTIF_TF = ANNDATA_FIXTURES / "motif_tf_mini.txt"

TAU = 1000.0


@pytest.fixture(scope="module")
def b4_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("b4_out")
    return build_rega_input(
        MATRICES_FIXTURES / "mini_gex.csv",
        MATRICES_FIXTURES / "mini_re_tg.txt",
        MATRICES_FIXTURES / "mini_peak_motif.txt",
        out,
        tau=TAU,
    )


# ── load_re_tg_matrix ─────────────────────────────────────────────────────────

def test_load_re_tg_matrix_0based_required(b4_out, tmp_path):
    """HDF5 without attrs['indexing']='0-based' raises ValueError."""
    bad_h5 = tmp_path / "bad.h5"
    with h5py.File(bad_h5, "w") as h5:
        h5.attrs["indexing"] = "1-based"
        h5.create_dataset("i", data=np.array([0], dtype=np.int32))
        h5.create_dataset("j", data=np.array([0], dtype=np.int32))
        h5.create_dataset("x", data=np.array([1.0], dtype=np.float64))
        h5.create_dataset("dims", data=np.array([1, 1], dtype=np.int32))
        h5.create_dataset("rownames", data=np.array(["G"], dtype="S"))
        h5.create_dataset("colnames", data=np.array(["P"], dtype="S"))
    with pytest.raises(ValueError, match="0-based"):
        load_re_tg_matrix(bad_h5)


def test_load_re_tg_matrix_shape_and_names(b4_out):
    """W has expected shape and gene/peak name lists."""
    W, genes, peaks = load_re_tg_matrix(b4_out["re_tg_h5"])
    assert W.shape == (3, 3)
    assert set(genes) == {"GENE_A", "GENE_B", "GENE_C"}
    assert set(peaks) == {"chr1_1100_1200", "chr1_3000_3200", "chr2_900_1100"}


def test_load_re_tg_matrix_weight_nonzero(b4_out):
    """W has nonzero entries with weights in (0, 1]."""
    W, _, _ = load_re_tg_matrix(b4_out["re_tg_h5"])
    assert W.nnz > 0
    assert float(W.max()) <= 1.0 + 1e-9
    assert float(W.min()) >= 0.0


# ── load_binding_sparse ───────────────────────────────────────────────────────

def test_load_binding_sparse_shape_matches_labels(b4_out):
    """B.shape matches the label file line counts."""
    B, b_peaks, motifs = load_binding_sparse(
        b4_out["binding_npz"], b4_out["binding_peaks_txt"], b4_out["binding_motifs_txt"]
    )
    assert B.shape == (len(b_peaks), len(motifs))


def test_load_binding_sparse_labels_correct(b4_out):
    """Peak and motif names match expected values from mini fixture."""
    _, b_peaks, motifs = load_binding_sparse(
        b4_out["binding_npz"], b4_out["binding_peaks_txt"], b4_out["binding_motifs_txt"]
    )
    assert set(b_peaks) == {"chr1_1100_1200", "chr1_3000_3200", "chr2_900_1100"}
    assert set(motifs) == {"MOTIF1", "MOTIF2", "MOTIF3"}


def test_load_binding_sparse_raises_on_shape_mismatch(b4_out, tmp_path):
    """ValueError when peaks_txt row count does not match B.shape[0]."""
    bad_peaks_txt = tmp_path / "bad_peaks.txt"
    bad_peaks_txt.write_text("chr1_1100_1200\nextra_peak\n")
    with pytest.raises(ValueError, match="does not match"):
        load_binding_sparse(b4_out["binding_npz"], bad_peaks_txt, b4_out["binding_motifs_txt"])


# ── build_rega_anndata success ────────────────────────────────────────────────

@pytest.fixture
def built_adata(b4_out, tmp_path):
    out = tmp_path / "rega.h5ad"
    return build_rega_anndata(
        b4_out["re_tg_h5"],
        b4_out["gex_csv"],
        b4_out["binding_npz"],
        b4_out["binding_peaks_txt"],
        b4_out["binding_motifs_txt"],
        MINI_ANNOTATION,
        out,
    )


def test_anndata_slots_exist(built_adata):
    """adata has required X, obsm['GEX'], varm['Peak_Motif'], uns keys."""
    adata = built_adata
    assert adata.X is not None
    assert "GEX" in adata.obsm
    assert "Peak_Motif" in adata.varm
    assert "GEX_sample_names" in adata.uns
    assert "motif_names" in adata.uns


def test_anndata_shapes_consistent(built_adata):
    """X, GEX, Peak_Motif shapes are mutually consistent."""
    adata = built_adata
    n_genes, n_peaks = adata.X.shape
    assert adata.obsm["GEX"].shape[0] == n_genes
    assert adata.varm["Peak_Motif"].shape[0] == n_peaks
    assert adata.varm["Peak_Motif"].shape[1] == len(adata.uns["motif_names"])


def test_anndata_dtypes(built_adata):
    """X is sparse CSR, GEX is float64, Peak_Motif is int8."""
    adata = built_adata
    assert isinstance(adata.X, scipy.sparse.csr_matrix)
    assert adata.obsm["GEX"].dtype == np.float64
    assert adata.varm["Peak_Motif"].dtype == np.int8


def test_anndata_obs_var_index(built_adata):
    """obs_names are the valid gene list; var_names are the valid peak list."""
    adata = built_adata
    assert set(adata.obs_names) == {"GENE_A", "GENE_B", "GENE_C"}
    assert set(adata.var_names) == {"chr1_1100_1200", "chr1_3000_3200", "chr2_900_1100"}


def test_anndata_writes_readable_h5ad(b4_out, tmp_path):
    """Output h5ad exists and can be read back with correct shape."""
    out = tmp_path / "rega.h5ad"
    build_rega_anndata(
        b4_out["re_tg_h5"],
        b4_out["gex_csv"],
        b4_out["binding_npz"],
        b4_out["binding_peaks_txt"],
        b4_out["binding_motifs_txt"],
        MINI_ANNOTATION,
        out,
    )
    assert out.exists()
    adata2 = ad.read_h5ad(out)
    assert adata2.X.shape == (3, 3)


# ── build_rega_anndata failure ────────────────────────────────────────────────

def test_anndata_missing_gene_in_annotation_raises(b4_out, tmp_path):
    """ValueError when annotation file lacks a gene present in W."""
    ann = pd.read_csv(MINI_ANNOTATION, sep="\t")
    ann_reduced = ann[ann["gene_name"] != "GENE_A"]
    bad_ann = tmp_path / "bad_ann.txt"
    ann_reduced.to_csv(bad_ann, sep="\t", index=False)
    out = tmp_path / "rega.h5ad"
    with pytest.raises(ValueError, match="GENE_A"):
        build_rega_anndata(
            b4_out["re_tg_h5"],
            b4_out["gex_csv"],
            b4_out["binding_npz"],
            b4_out["binding_peaks_txt"],
            b4_out["binding_motifs_txt"],
            bad_ann,
            out,
        )


def test_anndata_missing_peak_in_binding_raises(b4_out, tmp_path):
    """ValueError when binding peaks_txt lacks a peak present in W."""
    bad_peaks_txt = tmp_path / "bad_peaks.txt"
    bad_peaks_txt.write_text("chr1_3000_3200\nchr2_900_1100\n")
    out = tmp_path / "rega.h5ad"
    with pytest.raises(ValueError):
        build_rega_anndata(
            b4_out["re_tg_h5"],
            b4_out["gex_csv"],
            b4_out["binding_npz"],
            bad_peaks_txt,
            b4_out["binding_motifs_txt"],
            MINI_ANNOTATION,
            out,
        )


# ── build_motif_tf_matrix ─────────────────────────────────────────────────────

def test_motif_tf_shape_and_binary(b4_out, tmp_path):
    """Motif_TF matrix shape=(3, n_TFs), all nonzero values are 1."""
    adata = build_rega_anndata(
        b4_out["re_tg_h5"],
        b4_out["gex_csv"],
        b4_out["binding_npz"],
        b4_out["binding_peaks_txt"],
        b4_out["binding_motifs_txt"],
        MINI_ANNOTATION,
        tmp_path / "rega.h5ad",
        motif_tf_file=MINI_MOTIF_TF,
    )
    mtx = adata.uns["Motif_TF"]
    assert mtx.shape[0] == 3
    assert mtx.dtype == np.int8
    assert set(mtx.flatten().tolist()).issubset({0, 1})


def test_anndata_motif_tf_optional(b4_out, tmp_path):
    """Without motif_tf_file, adata.uns has no 'Motif_TF' key."""
    adata = build_rega_anndata(
        b4_out["re_tg_h5"],
        b4_out["gex_csv"],
        b4_out["binding_npz"],
        b4_out["binding_peaks_txt"],
        b4_out["binding_motifs_txt"],
        MINI_ANNOTATION,
        tmp_path / "rega.h5ad",
    )
    assert "Motif_TF" not in adata.uns
