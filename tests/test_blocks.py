from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from rega.anndata_builder import build_rega_anndata
from rega.blocks import get_common_chroms, load_and_check, split_and_save
from rega.matrices import build_rega_input

MATRICES_FIXTURES = Path(__file__).parent / "fixtures" / "matrices"
MINI_ANNOTATION = MATRICES_FIXTURES / "mini_re_tg.txt"

TAU = 1000.0


@pytest.fixture(scope="module")
def adata_h5ad(tmp_path_factory):
    b4 = build_rega_input(
        MATRICES_FIXTURES / "mini_gex.csv",
        MINI_ANNOTATION,
        MATRICES_FIXTURES / "mini_peak_motif.txt",
        tmp_path_factory.mktemp("b6_b4"),
        tau=TAU,
    )
    h5ad = tmp_path_factory.mktemp("b6_b5") / "rega.h5ad"
    build_rega_anndata(
        b4["re_tg_h5"],
        b4["gex_csv"],
        b4["binding_npz"],
        b4["binding_peaks_txt"],
        b4["binding_motifs_txt"],
        MINI_ANNOTATION,
        h5ad,
    )
    return h5ad


# ── load_and_check ────────────────────────────────────────────────────────────

def test_load_and_check_returns_anndata(adata_h5ad):
    """Valid h5ad loads without error and returns AnnData."""
    adata = load_and_check(adata_h5ad)
    assert isinstance(adata, ad.AnnData)


def test_load_and_check_missing_gex_raises(adata_h5ad, tmp_path):
    """Missing obsm['GEX'] raises ValueError mentioning 'GEX'."""
    adata = ad.read_h5ad(adata_h5ad)
    del adata.obsm["GEX"]
    bad = tmp_path / "bad.h5ad"
    adata.write_h5ad(bad)
    with pytest.raises(ValueError, match="GEX"):
        load_and_check(bad)


def test_load_and_check_missing_peak_motif_raises(adata_h5ad, tmp_path):
    """Missing varm['Peak_Motif'] raises ValueError mentioning 'Peak_Motif'."""
    adata = ad.read_h5ad(adata_h5ad)
    del adata.varm["Peak_Motif"]
    bad = tmp_path / "bad.h5ad"
    adata.write_h5ad(bad)
    with pytest.raises(ValueError, match="Peak_Motif"):
        load_and_check(bad)


# ── get_common_chroms ─────────────────────────────────────────────────────────

def test_get_common_chroms_intersection(adata_h5ad):
    """Returns exactly the chromosomes present in both obs['chrom'] and var['peak_chrom']."""
    adata = load_and_check(adata_h5ad)
    chroms = get_common_chroms(adata)
    assert set(chroms) == {"chr1", "chr2"}


def test_get_common_chroms_sort_order(adata_h5ad):
    """chr1 sorts before chr2."""
    adata = load_and_check(adata_h5ad)
    chroms = get_common_chroms(adata)
    assert chroms.index("chr1") < chroms.index("chr2")


# ── split_and_save ────────────────────────────────────────────────────────────

def test_split_and_save_file_count(adata_h5ad, tmp_path):
    """Writes one .npz per chromosome; mini fixture yields 2 blocks."""
    adata = load_and_check(adata_h5ad)
    n = split_and_save(adata, tmp_path)
    assert n == 2
    assert (tmp_path / "chr1.npz").exists()
    assert (tmp_path / "chr2.npz").exists()


def test_split_and_save_arrays_correct(adata_h5ad, tmp_path):
    """chr1 block: X(2,3) W(2,2) B(2,3); chr2 block: X(1,3) W(1,1) B(1,3)."""
    adata = load_and_check(adata_h5ad)
    split_and_save(adata, tmp_path)
    blk1 = np.load(tmp_path / "chr1.npz", allow_pickle=False)
    assert blk1["X"].shape == (2, 3)
    assert blk1["W"].shape == (2, 2)
    assert blk1["B"].shape == (2, 3)
    blk2 = np.load(tmp_path / "chr2.npz", allow_pickle=False)
    assert blk2["X"].shape == (1, 3)
    assert blk2["W"].shape == (1, 1)
    assert blk2["B"].shape == (1, 3)


def test_split_and_save_chroms_filter(adata_h5ad, tmp_path):
    """chroms=['chr1'] writes only chr1.npz and returns 1."""
    adata = load_and_check(adata_h5ad)
    n = split_and_save(adata, tmp_path, chroms=["chr1"])
    assert n == 1
    assert (tmp_path / "chr1.npz").exists()
    assert not (tmp_path / "chr2.npz").exists()


def test_split_and_save_zero_written_raises(adata_h5ad, tmp_path):
    """Requesting a chromosome absent from data raises ValueError with diagnostic."""
    adata = load_and_check(adata_h5ad)
    with pytest.raises(ValueError, match="Possible causes"):
        split_and_save(adata, tmp_path, chroms=["chr99"])


def test_split_and_save_no_pickle(adata_h5ad, tmp_path):
    """Saved .npz loads with allow_pickle=False and contains required keys."""
    adata = load_and_check(adata_h5ad)
    split_and_save(adata, tmp_path)
    blk = np.load(tmp_path / "chr1.npz", allow_pickle=False)
    assert set(blk.files) >= {"X", "W", "B", "genes", "RE", "samples", "motifs"}
