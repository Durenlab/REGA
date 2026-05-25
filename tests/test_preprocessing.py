"""
Unit tests for rega.preprocessing.

Test cases
----------
1. counts + meta           : full pipeline with covariate correction
2. counts, no meta         : covariate correction skipped
3. cpm + meta              : log2(CPM+1) is applied (values in log scale)
4. logcpm, no meta         : no transformation applied (values unchanged)
5. tpm, no meta            : no log transform (values stay in linear scale)
6. logtpm + meta           : covariate correction on log-scale data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rega.preprocessing import (
    preprocess_expression,
    load_metadata,
    load_expression,
    normalize_to_target_scale,
    filter_low_expression,
    align_samples,
    covariate_correction,
    post_process,
)

# ============================================================
# Constants
# ============================================================

N_GENES   = 20   # includes 2 MT genes at the end
N_SAMPLES = 8

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def gene_names() -> list[str]:
    return [f"GENE{i:02d}" for i in range(N_GENES - 2)] + ["MT-CO1", "mt-nd1"]


@pytest.fixture
def sample_names() -> list[str]:
    return [f"S{i:02d}" for i in range(N_SAMPLES)]


@pytest.fixture
def counts_df(gene_names, sample_names) -> pd.DataFrame:
    """Integer-valued raw count matrix."""
    rng  = np.random.default_rng(42)
    data = rng.integers(10, 500, size=(N_GENES, N_SAMPLES)).astype(float)
    return pd.DataFrame(data, index=gene_names, columns=sample_names)


@pytest.fixture
def cpm_df(counts_df) -> pd.DataFrame:
    """CPM matrix derived from counts (linear scale)."""
    lib_sizes = counts_df.sum(axis=0)
    return counts_df.div(lib_sizes, axis=1) * 1e6


@pytest.fixture
def logcpm_df(cpm_df) -> pd.DataFrame:
    """log2(CPM + 1) matrix."""
    return np.log2(cpm_df + 1)


@pytest.fixture
def tpm_df(gene_names, sample_names) -> pd.DataFrame:
    """TPM matrix (linear scale, values 5–500)."""
    rng  = np.random.default_rng(42)
    data = rng.uniform(5.0, 500.0, size=(N_GENES, N_SAMPLES))
    return pd.DataFrame(data, index=gene_names, columns=sample_names)


@pytest.fixture
def logtpm_df(tpm_df) -> pd.DataFrame:
    """log2(TPM + 1) matrix."""
    return np.log2(tpm_df + 1)


@pytest.fixture
def meta_df(sample_names) -> pd.DataFrame:
    """Metadata with one numeric (age) and one categorical (sex) covariate."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample": sample_names,
        "age":    rng.integers(25, 70, size=N_SAMPLES).tolist(),
        "sex":    (["male", "female"] * (N_SAMPLES // 2)),
    })


# ============================================================
# Helpers
# ============================================================

def _save_expr(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path)
    return path


def _save_meta(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ============================================================
# Test 1: counts + meta → full pipeline with covariate correction
# ============================================================

def test_counts_with_meta(tmp_path, counts_df, meta_df):
    expr_file = _save_expr(counts_df, tmp_path / "counts.csv")
    meta_file = _save_meta(meta_df,   tmp_path / "meta.csv")

    result = preprocess_expression(
        expr_file, meta_file=meta_file, input_type="counts"
    )

    # Shape: all 8 samples retained; MT genes removed by default
    assert result.shape[1] == N_SAMPLES
    assert not any(g.startswith(("MT-", "mt-")) for g in result.index)

    # After logCPM conversion and covariate correction, values are log-scale
    assert result.values.max() < 30

    # No remaining negative values (clipped)
    assert result.values.min() >= 0.0


# ============================================================
# Test 2: counts, no meta → covariate correction skipped
# ============================================================

def test_counts_no_meta(tmp_path, counts_df):
    expr_file = _save_expr(counts_df, tmp_path / "counts.csv")

    result = preprocess_expression(expr_file, input_type="counts")

    assert result.shape[1] == N_SAMPLES
    # Values should be in log scale (logCPM)
    assert result.values.max() < 30
    assert result.values.min() >= 0.0
    # MT genes removed
    assert not any(g.startswith(("MT-", "mt-")) for g in result.index)


# ============================================================
# Test 3: cpm + meta → log2(CPM+1) is applied
# ============================================================

def test_cpm_with_meta_applies_log(tmp_path, cpm_df, meta_df):
    expr_file = _save_expr(cpm_df, tmp_path / "cpm.csv")
    meta_file = _save_meta(meta_df, tmp_path / "meta.csv")

    result = preprocess_expression(
        expr_file, meta_file=meta_file, input_type="cpm"
    )

    # Raw CPM values are in the thousands; after log2(CPM+1) they are < 30.
    # If log had not been applied, the max would be >> 100.
    assert result.values.max() < 30
    assert result.values.min() >= 0.0


# ============================================================
# Test 4: logcpm, no meta → values pass through without transformation
# ============================================================

def test_logcpm_no_meta_no_transform(tmp_path, gene_names, sample_names):
    const_val = 5.0
    uniform_df = pd.DataFrame(
        np.full((N_GENES, N_SAMPLES), const_val),
        index=gene_names,
        columns=sample_names,
    )
    expr_file = _save_expr(uniform_df, tmp_path / "logcpm.csv")

    result = preprocess_expression(
        expr_file,
        input_type="logcpm",
        min_expression_threshold=1.0,
    )

    # No transformation → all values should remain at const_val (5.0)
    # MT genes removed, but remaining genes should all be 5.0
    assert np.allclose(result.values, const_val)
    assert not any(g.startswith(("MT-", "mt-")) for g in result.index)


# ============================================================
# Test 5: tpm, no meta → values stay in linear scale (no log applied)
# ============================================================

def test_tpm_no_meta_no_log_transform(tmp_path, tpm_df):
    expr_file = _save_expr(tpm_df, tmp_path / "tpm.csv")

    result = preprocess_expression(
        expr_file,
        input_type="tpm",
        min_expression_threshold=1.0,
    )

    # TPM fixture values are 5–500 (linear).
    # If log2 had been applied: log2(500) ≈ 9.
    # Max > 20 confirms the values remain in linear scale.
    assert result.values.max() > 20
    assert result.values.min() >= 0.0
    assert not any(g.startswith(("MT-", "mt-")) for g in result.index)


# ============================================================
# Test 6: logtpm + meta → covariate correction on log-scale data
# ============================================================

def test_logtpm_with_meta(tmp_path, logtpm_df, meta_df):
    expr_file = _save_expr(logtpm_df, tmp_path / "logtpm.csv")
    meta_file = _save_meta(meta_df,   tmp_path / "meta.csv")

    result = preprocess_expression(
        expr_file, meta_file=meta_file, input_type="logtpm"
    )

    assert result.shape[1] == N_SAMPLES
    # log2(TPM+1) values are in log scale (< 30)
    assert result.values.max() < 30
    assert result.values.min() >= 0.0
    assert not any(g.startswith(("MT-", "mt-")) for g in result.index)


# ============================================================
# Additional: mt_gene_prefix covers both human and mouse conventions
# ============================================================

def test_mt_gene_removal_both_prefixes(tmp_path, gene_names, sample_names):
    """gene_names contains 'MT-CO1' (human) and 'mt-nd1' (mouse)."""
    const_val = 3.0
    df = pd.DataFrame(
        np.full((N_GENES, N_SAMPLES), const_val),
        index=gene_names,
        columns=sample_names,
    )
    expr_file = _save_expr(df, tmp_path / "expr.csv")

    result = preprocess_expression(
        expr_file,
        input_type="logcpm",
        min_expression_threshold=1.0,
        mt_gene_prefix=("MT-", "mt-"),
    )

    remaining = set(result.index)
    assert "MT-CO1" not in remaining
    assert "mt-nd1" not in remaining
    # Non-MT genes should be present
    assert "GENE00" in remaining
