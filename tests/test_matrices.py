from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from rega.matrices import (
    build_peak_motif_matrix,
    build_re_tg_matrix,
    build_rega_input,
    filter_common_features,
    sort_genes_by_coord,
    sort_peaks_by_coord,
)

FIXTURES = Path(__file__).parent / "fixtures" / "matrices"
MINI_GEX = FIXTURES / "mini_gex.csv"
MINI_RE_TG = FIXTURES / "mini_re_tg.txt"
MINI_PEAK_MOTIF = FIXTURES / "mini_peak_motif.txt"

TAU = 1000.0


def _load_fixtures():
    """Load all three mini fixture files."""
    expr = pd.read_csv(MINI_GEX, index_col=0)
    re_tg = pd.read_csv(MINI_RE_TG, sep="\t")
    pm = pd.read_csv(MINI_PEAK_MOTIF, sep="\t", header=None, names=["peak", "motif"])
    return expr, re_tg, pm


def _filtered():
    """Run filter_common_features on mini fixtures; return (genes, peaks, re_tg_f, pm_f)."""
    expr, re_tg, pm = _load_fixtures()
    return filter_common_features(expr, re_tg, pm)


# ── sort_peaks_by_coord ───────────────────────────────────────────────────────

def test_sort_peaks_by_coord_within_chrom():
    """Peaks on the same chromosome are sorted by start position."""
    result = sort_peaks_by_coord(["chr1_3000_3200", "chr1_1100_1200"])
    assert result == ["chr1_1100_1200", "chr1_3000_3200"]


def test_sort_peaks_by_coord_cross_chrom():
    """chr1 peaks sort before chr2 peaks regardless of start position."""
    peaks = ["chr2_900_1100", "chr1_1100_1200", "chr1_3000_3200"]
    result = sort_peaks_by_coord(peaks)
    assert result.index("chr1_1100_1200") < result.index("chr2_900_1100")
    assert result.index("chr1_3000_3200") < result.index("chr2_900_1100")


# ── sort_genes_by_coord ───────────────────────────────────────────────────────

def test_sort_genes_by_coord_basic():
    """GENE_A(chr1,1000) < GENE_B(chr1,5000) < GENE_C(chr2,1000)."""
    _, re_tg, _ = _load_fixtures()
    result = sort_genes_by_coord({"GENE_A", "GENE_B", "GENE_C"}, re_tg)
    assert result.index("GENE_A") < result.index("GENE_B")
    assert result.index("GENE_B") < result.index("GENE_C")


# ── filter_common_features ────────────────────────────────────────────────────

def test_filter_excludes_gene_not_in_retg():
    """GENE_E (in GEX, absent from RE-TG) is excluded from output genes."""
    genes, _, _, _ = _filtered()
    assert "GENE_E" not in genes


def test_filter_excludes_gene_losing_all_peaks():
    """GENE_D (only peak chr2_2500_2600 absent from motif file) is excluded."""
    genes, _, _, _ = _filtered()
    assert "GENE_D" not in genes


def test_filter_common_peaks_intersection():
    """Only peaks present in both RE-TG and motif file are retained."""
    _, peaks, _, _ = _filtered()
    assert set(peaks) == {"chr1_1100_1200", "chr1_3000_3200", "chr2_900_1100"}
    assert "chr2_2500_2600" not in peaks


def test_filter_returns_sorted_order():
    """Returned gene and peak lists are in genomic coordinate order."""
    genes, peaks, _, _ = _filtered()
    assert genes.index("GENE_A") < genes.index("GENE_C")
    assert peaks.index("chr1_1100_1200") < peaks.index("chr2_900_1100")


# ── build_re_tg_matrix ────────────────────────────────────────────────────────

def test_build_re_tg_matrix_weight_value():
    """GENE_C (TSS=1000) × chr2_900_1100 (mid=1000): dist=0, weight=exp(0)=1.0."""
    genes, peaks, re_tg_f, _ = _filtered()
    W = build_re_tg_matrix(re_tg_f, genes, peaks, tau=TAU)
    gi = genes.index("GENE_C")
    pi = peaks.index("chr2_900_1100")
    assert abs(W[gi, pi] - 1.0) < 1e-9


def test_build_re_tg_matrix_shape():
    """W shape equals (n_valid_genes, n_valid_peaks)."""
    genes, peaks, re_tg_f, _ = _filtered()
    W = build_re_tg_matrix(re_tg_f, genes, peaks, tau=TAU)
    assert W.shape == (len(genes), len(peaks))


def test_build_re_tg_matrix_is_csr():
    """build_re_tg_matrix returns a scipy CSR matrix."""
    genes, peaks, re_tg_f, _ = _filtered()
    W = build_re_tg_matrix(re_tg_f, genes, peaks, tau=TAU)
    assert isinstance(W, scipy.sparse.csr_matrix)


def test_build_re_tg_matrix_takes_max_for_duplicates():
    """Duplicate (gene, peak) pairs keep max weight, not sum or mean."""
    genes = ["GENE_X"]
    peaks = ["chr1_1000_1200"]
    # Row 1: TSS=1100, mid=1100 → dist=0 → weight=exp(0)=1.0
    # Row 2: TSS=2000, mid=1100 → dist=900 → weight=exp(-0.9)≈0.407
    re_tg = pd.DataFrame({
        "gene_name":  ["GENE_X", "GENE_X"],
        "peak_name":  ["chr1_1000_1200", "chr1_1000_1200"],
        "TSS":        [1100, 2000],
        "peak_start": [1000, 1000],
        "peak_end":   [1200, 1200],
    })
    W = build_re_tg_matrix(re_tg, genes, peaks, tau=TAU)
    expected_max = 1.0
    expected_other = np.exp(-900.0 / TAU)
    assert W[0, 0] == pytest.approx(expected_max, rel=1e-6)
    assert W[0, 0] > expected_other   # max, not sum (sum would be >1)


# ── build_peak_motif_matrix ───────────────────────────────────────────────────

def test_build_peak_motif_matrix_binary():
    """All non-zero values in B are exactly 1 (int8)."""
    _, peaks, _, pm_f = _filtered()
    B, _ = build_peak_motif_matrix(pm_f, peaks)
    assert B.nnz > 0
    assert set(B.data.tolist()) == {1}


def test_build_peak_motif_matrix_shape():
    """B shape equals (n_valid_peaks, n_unique_motifs)."""
    _, peaks, _, pm_f = _filtered()
    B, motif_list = build_peak_motif_matrix(pm_f, peaks)
    assert B.shape == (len(peaks), len(motif_list))


# ── build_rega_input ──────────────────────────────────────────────────────────

def test_build_rega_input_all_paths_exist(tmp_path):
    """Returned dict has exactly 5 keys and all output files exist."""
    result = build_rega_input(MINI_GEX, MINI_RE_TG, MINI_PEAK_MOTIF, tmp_path, tau=TAU)
    assert set(result.keys()) == {
        "re_tg_h5", "gex_csv", "binding_npz", "binding_peaks_txt", "binding_motifs_txt",
    }
    for path in result.values():
        assert path.exists(), f"Missing output: {path}"


def test_build_rega_input_h5_0based(tmp_path):
    """HDF5 has attrs['indexing']='0-based' and i/j arrays are non-negative."""
    result = build_rega_input(MINI_GEX, MINI_RE_TG, MINI_PEAK_MOTIF, tmp_path, tau=TAU)
    with h5py.File(result["re_tg_h5"], "r") as h5:
        assert h5.attrs["indexing"] == "0-based"
        assert int(h5["i"][:].min()) >= 0
        assert int(h5["j"][:].min()) >= 0


def test_build_rega_input_binding_npz_loadable(tmp_path):
    """input_binding.npz loads as a sparse matrix with shape (n_peaks, n_motifs)."""
    result = build_rega_input(MINI_GEX, MINI_RE_TG, MINI_PEAK_MOTIF, tmp_path, tau=TAU)
    B = scipy.sparse.load_npz(str(result["binding_npz"]))
    peaks = result["binding_peaks_txt"].read_text().splitlines()
    motifs = result["binding_motifs_txt"].read_text().splitlines()
    assert B.shape == (len(peaks), len(motifs))


def test_build_rega_input_gex_gene_order(tmp_path):
    """input_GEX.csv row order follows genomic coordinate sort (GENE_A < GENE_B < GENE_C)."""
    result = build_rega_input(MINI_GEX, MINI_RE_TG, MINI_PEAK_MOTIF, tmp_path, tau=TAU)
    gex = pd.read_csv(result["gex_csv"], index_col=0)
    gene_order = list(gex.index)
    assert gene_order.index("GENE_A") < gene_order.index("GENE_B")
    assert gene_order.index("GENE_B") < gene_order.index("GENE_C")
