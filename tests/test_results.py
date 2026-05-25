from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse

from rega.results import (
    assemble_B_learned,
    assemble_W_learned,
    build_result_anndata,
    collect_results,
    compute_downstream,
    consistency_checks,
    load_input_anndata,
    load_rega_result,
    normalize_embedding,
)

_NG, _NR, _NM, _NTF, _NS, _K = 4, 5, 3, 2, 3, 2

_GENES   = np.array(["G0", "G1", "G2", "G3"])
_RE      = np.array(["R0", "R1", "R2", "R3", "R4"])
_MOTIFS  = np.array(["M0", "M1", "M2"])
_TFS     = np.array(["G0", "G1"])  # must be a subset of _GENES
_SAMPLES = np.array(["S0", "S1", "S2"])


def _make_adata(rng: np.random.Generator | None = None) -> ad.AnnData:
    if rng is None:
        rng = np.random.default_rng(42)
    W0 = sparse.csr_matrix(rng.random((_NG, _NR)).astype(np.float32))
    adata = ad.AnnData(X=W0)
    adata.obs_names = list(_GENES)
    adata.var_names = list(_RE)
    adata.obsm["GEX"]        = rng.random((_NG, _NS)).astype(np.float32)
    adata.varm["Peak_Motif"] = rng.random((_NR, _NM)).astype(np.float32)
    adata.uns["motif_names"]      = _MOTIFS.copy()
    adata.uns["TF_names"]         = _TFS.copy()
    adata.uns["GEX_sample_names"] = _SAMPLES.copy()
    adata.uns["Motif_TF"]         = rng.random((_NM, _NTF)).astype(np.float32)
    return adata


def _make_ckpt(rng: np.random.Generator | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    return {
        "genes_blocks": [np.array(["G0", "G1"]), np.array(["G2", "G3"])],
        "RE_blocks":    [np.array(["R0", "R1", "R2"]), np.array(["R3", "R4"])],
        "W_blocks":     [
            rng.random((2, 3)).astype(np.float32),
            rng.random((2, 2)).astype(np.float32),
        ],
        "B_blocks":     [
            rng.random((3, _NM)).astype(np.float32),
            rng.random((2, _NM)).astype(np.float32),
        ],
        "V":      rng.random((_NM, _K)).astype(np.float32),
        "H":      rng.random((_K,  _NS)).astype(np.float32),
        "samples": _SAMPLES.copy(),
        "motifs":  _MOTIFS.copy(),
        "loss":    1.5,
        "step":    10,
        "K":       _K,
    }


def _save_adata(adata: ad.AnnData, path: Path) -> Path:
    adata.write_h5ad(path)
    return path


def _save_ckpt(ckpt: dict, path: Path) -> Path:
    torch.save(ckpt, path)
    return path


# ── load_input_anndata ─────────────────────────────────────────────────────────

def test_load_input_anndata_valid(tmp_path):
    """Returns correct gene/RE/motif/sample/TF arrays from a valid input h5ad."""
    h5ad = _save_adata(_make_adata(), tmp_path / "input.h5ad")
    adata, genes, re, motifs, samples, tfs = load_input_anndata(h5ad)
    assert list(genes)   == list(_GENES)
    assert list(re)      == list(_RE)
    assert list(motifs)  == list(_MOTIFS)
    assert list(samples) == list(_SAMPLES)
    assert list(tfs)     == list(_TFS)


def test_load_input_anndata_missing_uns_key(tmp_path):
    """Missing uns key raises ValueError naming the key."""
    adata = _make_adata()
    del adata.uns["TF_names"]
    h5ad = _save_adata(adata, tmp_path / "input.h5ad")
    with pytest.raises(ValueError, match="TF_names"):
        load_input_anndata(h5ad)


def test_load_input_anndata_missing_obsm(tmp_path):
    """Missing obsm['GEX'] raises ValueError."""
    adata = _make_adata()
    del adata.obsm["GEX"]
    h5ad = _save_adata(adata, tmp_path / "input.h5ad")
    with pytest.raises(ValueError, match="GEX"):
        load_input_anndata(h5ad)


# ── load_rega_result ──────────────────────────────────────────────────────────

def test_load_rega_result_valid(tmp_path):
    """Loads valid checkpoint and returns dict with all required fields."""
    pt = _save_ckpt(_make_ckpt(), tmp_path / "final.pt")
    ckpt = load_rega_result(pt)
    for k in ("W_blocks", "B_blocks", "V", "H",
              "genes_blocks", "RE_blocks", "samples", "motifs"):
        assert k in ckpt


def test_load_rega_result_missing_field(tmp_path):
    """Missing checkpoint field raises ValueError naming the field."""
    d = _make_ckpt()
    del d["W_blocks"]
    pt = _save_ckpt(d, tmp_path / "final.pt")
    with pytest.raises(ValueError, match="W_blocks"):
        load_rega_result(pt)


# ── consistency_checks ─────────────────────────────────────────────────────────

def test_consistency_checks_passes():
    """Valid checkpoint + master arrays pass without error."""
    ckpt = _make_ckpt()
    W_blocks, B_blocks, genes_blocks, RE_blocks, V, H = consistency_checks(
        ckpt, _GENES, _RE, _MOTIFS, _SAMPLES
    )
    assert len(W_blocks) == 2
    assert V.shape == (_NM, _K)
    assert H.shape == (_K, _NS)


def test_consistency_checks_samples_mismatch():
    """Mismatched 'samples' array raises ValueError."""
    ckpt = _make_ckpt()
    ckpt["samples"] = np.array(["S0", "S1", "WRONG"])
    with pytest.raises(ValueError, match="samples"):
        consistency_checks(ckpt, _GENES, _RE, _MOTIFS, _SAMPLES)


def test_consistency_checks_re_not_in_master():
    """RE in block but absent from master raises ValueError."""
    ckpt = _make_ckpt()
    ckpt["RE_blocks"] = [np.array(["R0", "R1", "BOGUS"]), np.array(["R3", "R4"])]
    with pytest.raises(ValueError, match="not in input AnnData var_names"):
        consistency_checks(ckpt, _GENES, _RE, _MOTIFS, _SAMPLES)


# ── assemble_W_learned / assemble_B_learned ───────────────────────────────────

def test_assemble_W_learned_shape():
    """W_learned has the correct global shape and positive nnz."""
    ckpt = _make_ckpt()
    _, _, genes_blocks, RE_blocks, _, _ = consistency_checks(
        ckpt, _GENES, _RE, _MOTIFS, _SAMPLES
    )
    W_learned = assemble_W_learned(
        ckpt["W_blocks"], genes_blocks, RE_blocks, _GENES, _RE
    )
    assert W_learned.shape == (_NG, _NR)
    assert W_learned.nnz > 0


def test_assemble_B_learned_shape_and_values():
    """B_learned has the correct global shape and block values in the right rows."""
    rng  = np.random.default_rng(7)
    ckpt = _make_ckpt(rng)
    _, _, _, RE_blocks, _, _ = consistency_checks(
        ckpt, _GENES, _RE, _MOTIFS, _SAMPLES
    )
    B_learned = assemble_B_learned(ckpt["B_blocks"], RE_blocks, _RE, _MOTIFS)
    assert B_learned.shape == (_NR, _NM)
    # R0 is row 0 in RE_blocks[0]; block 0's first row should land at B_learned[0]
    np.testing.assert_allclose(
        B_learned[0, :],
        np.asarray(ckpt["B_blocks"][0], dtype=np.float32)[0, :],
    )


# ── normalize_embedding ───────────────────────────────────────────────────────

def test_normalize_embedding_nonneg():
    """All-nonneg input remains nonneg after normalization."""
    E   = np.random.default_rng(5).random((6, 4)).astype(np.float32)
    out = normalize_embedding(E)
    assert (out >= 0).all()


def test_normalize_embedding_row_l1():
    """Row L1 sums equal 1 after normalization."""
    E   = np.random.default_rng(5).random((6, 4)).astype(np.float32)
    out = normalize_embedding(E)
    np.testing.assert_allclose(np.abs(out).sum(axis=1), 1.0, atol=1e-5)


# ── compute_downstream ─────────────────────────────────────────────────────────

def _build_downstream():
    rng   = np.random.default_rng(42)
    adata = _make_adata(rng)
    ckpt  = _make_ckpt(rng)
    _, _, genes_blocks, RE_blocks, V, H = consistency_checks(
        ckpt, _GENES, _RE, _MOTIFS, _SAMPLES
    )
    W_learned = assemble_W_learned(ckpt["W_blocks"], genes_blocks, RE_blocks, _GENES, _RE)
    B_learned = assemble_B_learned(ckpt["B_blocks"], RE_blocks, _RE, _MOTIFS)
    results   = compute_downstream(
        adata, W_learned, B_learned, V, H, _MOTIFS, _TFS, _SAMPLES
    )
    return results, V, H


def test_compute_downstream_shapes():
    """All downstream arrays have the expected shapes."""
    results, V, H = _build_downstream()
    assert results["BV"].shape           == (_NR, _K)
    assert results["WBV"].shape          == (_NG, _K)
    assert results["H_T"].shape          == (_NS, _K)
    assert results["VH"].shape           == (_NM, _NS)
    assert results["BVH"].shape          == (_NR, _NS)
    assert results["w"].shape            == (_NR,)
    assert results["d"].shape            == (_NR,)
    assert results["Net_RE_TG"].shape    == (_NG, _NR)
    assert results["Net_TF_TG"].shape    == (_NG, _NTF)
    assert results["Motif_TF_PCC"].shape == (_NM, _NTF)


def test_compute_downstream_pcc_nonneg():
    """Motif_TF_PCC contains no negative values (negatives clipped to 0)."""
    results, _, _ = _build_downstream()
    assert (results["Motif_TF_PCC"] >= 0).all()


# ── build_result_anndata / collect_results ────────────────────────────────────

def test_build_result_anndata_keys(tmp_path):
    """Result AnnData contains all required layers, obsm, varm, and uns keys."""
    rng   = np.random.default_rng(9)
    adata = _make_adata(rng)
    ckpt  = _make_ckpt(rng)
    _, _, genes_blocks, RE_blocks, V, H = consistency_checks(
        ckpt, _GENES, _RE, _MOTIFS, _SAMPLES
    )
    W_learned = assemble_W_learned(ckpt["W_blocks"], genes_blocks, RE_blocks, _GENES, _RE)
    B_learned = assemble_B_learned(ckpt["B_blocks"], RE_blocks, _RE, _MOTIFS)
    results   = compute_downstream(
        adata, W_learned, B_learned, V, H, _MOTIFS, _TFS, _SAMPLES
    )
    out_h5ad = tmp_path / "result.h5ad"

    ar = build_result_anndata(
        adata, W_learned, B_learned, V, H,
        results, ckpt, out_h5ad, _MOTIFS, _TFS, _SAMPLES,
    )
    assert out_h5ad.exists()
    for key in ("W0", "Net_RE_TG"):
        assert key in ar.layers
    for key in ("WBV", "WBV_norm", "Net_TF_TG"):
        assert key in ar.obsm
    for key in ("B_learned", "BV", "BV_norm", "Peak_Activity_BVH"):
        assert key in ar.varm
    for key in ("V", "H", "Motif_TF_PCC", "Motif_Activity_VH",
                "REGA_loss", "REGA_step", "REGA_K", "Module_names"):
        assert key in ar.uns


def test_collect_results_writes_h5ad(tmp_path):
    """collect_results runs end-to-end and writes the output h5ad with correct shape."""
    rng  = np.random.default_rng(11)
    h5ad = _save_adata(_make_adata(rng), tmp_path / "input.h5ad")
    pt   = _save_ckpt(_make_ckpt(rng), tmp_path / "final.pt")
    out  = tmp_path / "result.h5ad"
    ar   = collect_results(h5ad, pt, out)
    assert out.exists()
    assert ar.shape == (_NG, _NR)
