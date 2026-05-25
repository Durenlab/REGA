"""
Unit tests for rega.analysis.driver_tf.

Test cases
----------
1. bh_fdr: monotonicity and clamp to 1
2. bh_fdr: all NaN input returns all NaN
3. read_label_file: parses header-less file correctly
4. read_label_file: detects and skips header row
5. read_label_file: invalid labels raise ValueError
6. extract_tf_expression: known TF gets correct row
7. extract_tf_expression: unknown TF gets zeros and found_mask=False
8. compute_tf_activity: output shape is (n_TF, n_sample)
9. identify_driver_tfs: returns DataFrame with required columns
10. identify_driver_tfs: fewer than 3 matched samples raises ValueError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rega.analysis._utils import bh_fdr, read_label_file
from rega.analysis.driver_tf import (
    compute_tf_activity,
    extract_tf_expression,
    identify_driver_tfs,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_label_df(n_control: int, n_disease: int) -> pd.DataFrame:
    ids = [f"S{i}" for i in range(n_control + n_disease)]
    labels = [0] * n_control + [1] * n_disease
    return pd.DataFrame({"sample_id": ids, "disease_label": labels})


class _FakeAdata:
    def __init__(self, n_genes=6, n_motifs=4, n_tfs=3, n_samples=20):
        rng = np.random.default_rng(1)
        self.obs_names = pd.Index([f"Gene{i}" for i in range(n_genes)])
        sample_names = [f"S{i}" for i in range(n_samples)]

        gex = rng.random((n_genes, n_samples)).astype(np.float32)
        motif_activity = rng.random((n_motifs, n_samples)).astype(np.float32)
        motif_tf = rng.integers(0, 2, size=(n_motifs, n_tfs)).astype(np.float32)
        motif_tf_pcc = rng.random((n_motifs, n_tfs)).astype(np.float32)
        net_tf_tg = rng.random((n_genes, n_tfs)).astype(np.float32)

        tf_names = [f"Gene{i}" for i in range(n_tfs - 1)] + ["UNKNOWN"]
        motif_names = [f"Motif{i}" for i in range(n_motifs)]

        self.obsm = {"GEX": gex, "Net_TF_TG": net_tf_tg}
        self.uns = {
            "Motif_Activity_VH": motif_activity,
            "Motif_TF": motif_tf,
            "Motif_TF_PCC": motif_tf_pcc,
            "TF_names": np.array(tf_names),
            "motif_names": np.array(motif_names),
            "GEX_sample_names": np.array(sample_names),
        }


# ── tests ──────────────────────────────────────────────────────────────────────

def test_bh_fdr_monotone_and_clamped():
    pvals = np.array([0.001, 0.01, 0.05, 0.2, 0.5])
    fdr = bh_fdr(pvals)
    assert (fdr <= 1.0).all()
    assert fdr[-1] >= fdr[0]


def test_bh_fdr_all_nan():
    pvals = np.array([np.nan, np.nan])
    fdr = bh_fdr(pvals)
    assert np.all(np.isnan(fdr))


def test_read_label_file_no_header(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("S0\t0\nS1\t1\nS2\t0\n")
    df = read_label_file(f)
    assert list(df.columns) == ["sample_id", "disease_label"]
    assert df.shape[0] == 3
    assert set(df["disease_label"]) == {0, 1}


def test_read_label_file_with_header(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("sample_id\tdisease_label\nS0\t0\nS1\t1\n")
    df = read_label_file(f)
    assert df.shape[0] == 2


def test_read_label_file_invalid_label(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("sample_id\tdisease_label\nS0\t0\nS1\t2\n")
    with pytest.raises(ValueError, match="0/1"):
        read_label_file(f)


def test_extract_tf_expression_known_tf():
    gex = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gene_names = np.array(["GeneA", "GeneB"])
    tf_names = np.array(["GeneA"])
    tf_expr, found_mask = extract_tf_expression(gex, gene_names, tf_names)
    assert found_mask[0]
    np.testing.assert_array_equal(tf_expr[0], gex[0])


def test_extract_tf_expression_unknown_tf_zeros():
    gex = np.array([[1.0, 2.0]], dtype=np.float32)
    gene_names = np.array(["GeneA"])
    tf_names = np.array(["UNKNOWN"])
    tf_expr, found_mask = extract_tf_expression(gex, gene_names, tf_names)
    assert not found_mask[0]
    np.testing.assert_array_equal(tf_expr[0], [0.0, 0.0])


def test_compute_tf_activity_shape():
    rng = np.random.default_rng(0)
    n_motifs, n_tfs, n_samples = 5, 3, 10
    motif_activity = rng.random((n_motifs, n_samples)).astype(np.float32)
    motif_tf = rng.integers(0, 2, (n_motifs, n_tfs)).astype(np.float32)
    motif_tf_pcc = rng.random((n_motifs, n_tfs)).astype(np.float32)
    tf_expression = rng.random((n_tfs, n_samples)).astype(np.float32)

    activity = compute_tf_activity(motif_activity, motif_tf, motif_tf_pcc, tf_expression, 0.1)
    assert activity.shape == (n_tfs, n_samples)


def test_identify_driver_tfs_columns():
    adata = _FakeAdata()
    label_df = _make_label_df(n_control=10, n_disease=10)
    result = identify_driver_tfs(adata, label_df)
    expected = {"TF", "spearman_rho", "p_value", "FDR", "driver_TF_significant",
                "n_samples", "n_control", "n_disease", "TF_found_in_GEX",
                "mean_activity_control", "mean_activity_disease", "delta_activity"}
    assert expected.issubset(result.columns)


def test_identify_driver_tfs_too_few_samples_raises():
    adata = _FakeAdata()
    label_df = _make_label_df(n_control=1, n_disease=1)
    with pytest.raises(ValueError, match="3 matched samples"):
        identify_driver_tfs(adata, label_df)
