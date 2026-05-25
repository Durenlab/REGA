"""
Unit tests for rega.analysis.re_score.

Test cases
----------
1. extract_re_importance: returns correct columns
2. extract_re_importance: output is sorted descending by RE_importance
3. extract_re_importance: missing column raises ValueError
4. write_re_importance: file is created and readable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rega.analysis.re_score import extract_re_importance, write_re_importance


class _FakeAdata:
    def __init__(self):
        self.var_names = pd.Index(["Peak0", "Peak1", "Peak2"])
        self.var = pd.DataFrame(
            {"RE_importance": np.array([0.3, 0.8, 0.1], dtype=np.float32)},
            index=self.var_names,
        )


def test_extract_re_importance_columns():
    adata = _FakeAdata()
    df = extract_re_importance(adata)
    assert list(df.columns) == ["regulatory_element", "RE_importance"]
    assert df.shape[0] == 3


def test_extract_re_importance_sorted_descending():
    adata = _FakeAdata()
    df = extract_re_importance(adata)
    scores = df["RE_importance"].to_numpy()
    assert (scores[:-1] >= scores[1:]).all()


def test_extract_re_importance_missing_column():
    class _Bad:
        var_names = pd.Index(["p"])
        var = pd.DataFrame(index=pd.Index(["p"]))

    with pytest.raises(ValueError, match="RE_importance"):
        extract_re_importance(_Bad())


def test_write_re_importance_creates_file(tmp_path):
    adata = _FakeAdata()
    out_path = write_re_importance(adata, out_tsv=tmp_path / "out.tsv")
    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == ["regulatory_element", "RE_importance"]
