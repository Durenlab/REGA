"""
Unit tests for rega.analysis.disease.

Test cases
----------
1. load_module_activity: returns correct shapes
2. load_module_activity: missing H_T raises ValueError
3. load_module_activity: generates module names if Module_names absent
4. identify_disease_modules: returns DataFrame with required columns
5. identify_disease_modules: FDR column values are in [0, 1]
6. identify_disease_modules: single-class labels raise ValueError
7. identify_disease_modules: fewer than 3 matched samples raises ValueError
8. write_disease_module_results: creates both output files
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rega.analysis.disease import (
    identify_disease_modules,
    load_module_activity,
    write_disease_module_results,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_label_df(n_control: int, n_disease: int, prefix: str = "S") -> pd.DataFrame:
    ids = [f"{prefix}{i}" for i in range(n_control + n_disease)]
    labels = [0] * n_control + [1] * n_disease
    return pd.DataFrame({"sample_id": ids, "disease_label": labels})


class _FakeAdata:
    def __init__(self, n_samples=20, n_modules=4, has_module_names=True):
        rng = np.random.default_rng(2)
        sample_names = [f"S{i}" for i in range(n_samples)]
        H_T = rng.random((n_samples, n_modules)).astype(np.float32)
        module_names = [f"Module_{i+1}" for i in range(n_modules)]

        self.uns = {
            "H_T": H_T,
            "GEX_sample_names": np.array(sample_names),
        }
        if has_module_names:
            self.uns["Module_names"] = np.array(module_names)


# ── tests ──────────────────────────────────────────────────────────────────────

def test_load_module_activity_shapes():
    adata = _FakeAdata(n_samples=20, n_modules=4)
    H_T, sample_names, module_names = load_module_activity(adata)
    assert H_T.shape == (20, 4)
    assert len(sample_names) == 20
    assert len(module_names) == 4


def test_load_module_activity_missing_h_t():
    class _Bad:
        uns = {"GEX_sample_names": np.array(["S0"])}

    with pytest.raises(ValueError, match="H_T"):
        load_module_activity(_Bad())


def test_load_module_activity_auto_module_names():
    adata = _FakeAdata(n_samples=5, n_modules=3, has_module_names=False)
    _, _, module_names = load_module_activity(adata)
    assert module_names[0] == "Module_1"
    assert len(module_names) == 3


def test_identify_disease_modules_columns():
    adata = _FakeAdata(n_samples=20, n_modules=4)
    label_df = _make_label_df(10, 10)
    result = identify_disease_modules(adata, label_df)
    expected = {"module", "spearman_rho", "p_value", "FDR", "disease_module_significant",
                "n_samples", "n_control", "n_disease",
                "mean_activity_control", "mean_activity_disease", "delta_activity"}
    assert expected.issubset(result.columns)
    assert result.shape[0] == 4


def test_identify_disease_modules_fdr_range():
    adata = _FakeAdata(n_samples=20, n_modules=4)
    label_df = _make_label_df(10, 10)
    result = identify_disease_modules(adata, label_df)
    valid_fdr = result["FDR"].dropna()
    assert (valid_fdr >= 0).all() and (valid_fdr <= 1).all()


def test_identify_disease_modules_single_class_raises():
    adata = _FakeAdata(n_samples=10, n_modules=2)
    label_df = _make_label_df(10, 0)
    with pytest.raises(ValueError, match="one class"):
        identify_disease_modules(adata, label_df)


def test_identify_disease_modules_too_few_samples_raises():
    adata = _FakeAdata(n_samples=20, n_modules=2)
    label_df = _make_label_df(1, 1)
    with pytest.raises(ValueError, match="3 matched samples"):
        identify_disease_modules(adata, label_df)


def test_write_disease_module_results_creates_files(tmp_path):
    adata = _FakeAdata(n_samples=20, n_modules=4)
    label_df = _make_label_df(10, 10)
    result = identify_disease_modules(adata, label_df)
    paths = write_disease_module_results(result, out_dir=tmp_path / "out")
    assert paths["all"].exists()
    assert paths["significant"].exists()
