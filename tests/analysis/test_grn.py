"""
Unit tests for rega.analysis.grn.

Test cases
----------
1. extract_re_tg: returns DataFrame with correct columns
2. extract_re_tg: min_score filter removes edges at or below threshold
3. extract_re_tg: missing Net_RE_TG raises ValueError
4. compute_tf_mean_expression: known TF has correct mean
5. compute_tf_mean_expression: TF absent from obs_names gets mean 0
6. extract_tf_tg: writes TSV with correct columns and edge count
7. extract_grn: creates both output files
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rega.analysis.grn import (
    compute_tf_mean_expression,
    extract_grn,
    extract_re_tg,
    extract_tf_tg,
)


# ── synthetic adata ────────────────────────────────────────────────────────────

class _FakeAdata:
    """Minimal AnnData stub for testing GRN functions."""

    def __init__(self, n_genes=5, n_peaks=6, n_tfs=3, n_samples=8):
        rng = np.random.default_rng(0)

        self.obs_names = pd.Index([f"Gene{i}" for i in range(n_genes)])
        self.var_names = pd.Index([f"Peak{i}" for i in range(n_peaks)])

        net = rng.random((n_genes, n_peaks)).astype(np.float32)
        net[net < 0.5] = 0.0

        gex = rng.random((n_genes, n_samples)).astype(np.float32)
        net_tf_tg = rng.random((n_genes, n_tfs)).astype(np.float32)

        self.layers = {"Net_RE_TG": sp.csr_matrix(net)}
        self.obsm = {
            "GEX": gex,
            "Net_TF_TG": net_tf_tg,
        }

        tf_names = [f"Gene{i}" for i in range(n_tfs - 1)] + ["UNKNOWN_TF"]
        self.uns = {"TF_names": np.array(tf_names)}


@pytest.fixture
def fake_adata():
    return _FakeAdata()


# ── tests ──────────────────────────────────────────────────────────────────────

def test_extract_re_tg_columns(fake_adata):
    df = extract_re_tg(fake_adata)
    assert list(df.columns) == ["target_gene", "regulatory_element", "score"]


def test_extract_re_tg_min_score_filter(fake_adata):
    df_all = extract_re_tg(fake_adata, min_score=0.0)
    df_filtered = extract_re_tg(fake_adata, min_score=0.9)
    assert df_filtered.shape[0] <= df_all.shape[0]
    if df_filtered.shape[0] > 0:
        assert df_filtered["score"].min() > 0.9


def test_extract_re_tg_missing_layer():
    class _Bad:
        obs_names = pd.Index(["g"])
        var_names = pd.Index(["p"])
        layers = {}

    with pytest.raises(ValueError, match="Net_RE_TG"):
        extract_re_tg(_Bad())


def test_compute_tf_mean_expression_known_tf(fake_adata):
    tf_names, tf_mean_expr = compute_tf_mean_expression(fake_adata)
    assert len(tf_names) == 3
    gene0_idx = list(fake_adata.obs_names).index("Gene0")
    expected = float(np.nanmean(fake_adata.obsm["GEX"][gene0_idx, :]))
    np.testing.assert_allclose(tf_mean_expr[0], expected, rtol=1e-5)


def test_compute_tf_mean_expression_unknown_tf_gets_zero(fake_adata):
    _, tf_mean_expr = compute_tf_mean_expression(fake_adata)
    assert tf_mean_expr[-1] == pytest.approx(0.0)


def test_extract_tf_tg_writes_tsv(fake_adata, tmp_path):
    tf_names, tf_mean_expr = compute_tf_mean_expression(fake_adata)
    out_path = tmp_path / "TF_TG.tsv"
    n_edges = extract_tf_tg(
        fake_adata, out_path, tf_names, tf_mean_expr, min_score=0.0, chunk_size=2
    )
    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == ["target_gene", "TF", "score"]
    assert df.shape[0] == n_edges


def test_extract_grn_creates_files(fake_adata, tmp_path):
    paths = extract_grn(fake_adata, out_dir=tmp_path / "grn")
    assert paths["RE_TG"].exists()
    assert paths["TF_TG"].exists()
