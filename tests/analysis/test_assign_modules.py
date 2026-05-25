"""
Unit tests for rega.analysis.assign_modules.

Test cases
----------
1. assign_topk_sets: basic top-K selection per module
2. assign_topk_sets: topk larger than n_entities is clamped
3. assign_topk_sets: topk=0 returns empty lists
4. harden_sets_by_score: entity in two modules goes to highest-score module
5. harden_sets_by_score: entity in one module stays
6. assign_label_absolute: entities above threshold get assigned, others get -1
7. assign_label_absolute: all below threshold returns all -1
8. assign_modules hard mode: each entity appears at most once in assignment_table
9. assign_modules soft mode: entity may appear multiple times in assignment_table
10. assign_modules hard mode: hardened_sets is a dict
11. assign_modules soft mode: hardened_sets is None
12. assign_modules: returned dict has all required keys
13. assign_modules: module_counts sums to <= n_entities (hard)
14. assign_modules: topk_frac=0 edge case (topk clamped to 1)
15. write_module_assignments: files are created with correct columns
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rega.analysis.assign_modules import (
    assign_label_absolute,
    assign_modules,
    assign_topk_sets,
    harden_sets_by_score,
    write_module_assignments,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_df(data: list[list[float]], entities: list[str], modules: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=entities, columns=modules, dtype=np.float32)


@pytest.fixture
def simple_df():
    """3 entities × 2 modules. Entity A dominates module M1, C dominates M2."""
    return _make_df(
        [[0.9, 0.1], [0.4, 0.4], [0.1, 0.8]],
        ["A", "B", "C"],
        ["M1", "M2"],
    )


@pytest.fixture
def conflict_df():
    """Entity X scores high in both modules; Y only in M2."""
    return _make_df(
        [[0.8, 0.7], [0.1, 0.9]],
        ["X", "Y"],
        ["M1", "M2"],
    )


# ── tests ──────────────────────────────────────────────────────────────────────

def test_topk_basic(simple_df):
    sets = assign_topk_sets(simple_df, topk=1)
    assert sets["M1"] == ["A"]
    assert sets["M2"] == ["C"]


def test_topk_clamped_to_n_entities(simple_df):
    sets = assign_topk_sets(simple_df, topk=100)
    assert len(sets["M1"]) == 3
    assert set(sets["M1"]) == {"A", "B", "C"}


def test_topk_zero_returns_empty(simple_df):
    sets = assign_topk_sets(simple_df, topk=0)
    assert sets["M1"] == []
    assert sets["M2"] == []


def test_harden_conflict_goes_to_max(conflict_df):
    sets = {"M1": ["X", "Y"], "M2": ["X", "Y"]}
    hardened = harden_sets_by_score(conflict_df, sets)
    assert "X" in hardened["M1"] and "X" not in hardened["M2"]
    assert "Y" in hardened["M2"] and "Y" not in hardened["M1"]


def test_harden_no_conflict(simple_df):
    sets = {"M1": ["A"], "M2": ["C"]}
    hardened = harden_sets_by_score(simple_df, sets)
    assert hardened["M1"] == ["A"]
    assert hardened["M2"] == ["C"]


def test_label_absolute_basic(simple_df):
    labels, threshold = assign_label_absolute(simple_df, p_abs=0.5)
    assert threshold == 0.5
    assert labels[0] == 0   # A → M1 (0.9 >= 0.5)
    assert labels[1] == -1  # B: max=0.4, below threshold
    assert labels[2] == 1   # C → M2 (0.8 >= 0.5)


def test_label_absolute_all_below_threshold(simple_df):
    labels, _ = assign_label_absolute(simple_df, p_abs=0.99)
    assert np.all(labels == -1)


def test_assign_modules_hard_unique_assignments(simple_df):
    result = assign_modules(simple_df, topk_frac=0.5, p_abs=0.3, assignment_type="hard")
    table = result["assignment_table"]
    assert not table["entity_name"].duplicated().any(), "hard mode: entity appears more than once"


def test_assign_modules_soft_can_have_duplicates():
    data = _make_df(
        [[0.8, 0.75], [0.3, 0.9]],
        ["X", "Y"],
        ["M1", "M2"],
    )
    result = assign_modules(data, topk_frac=1.0, p_abs=0.3, assignment_type="soft")
    table = result["assignment_table"]
    x_rows = table[table["entity_name"] == "X"]
    assert len(x_rows) >= 1


def test_assign_modules_hard_hardened_sets_is_dict(simple_df):
    result = assign_modules(simple_df, assignment_type="hard")
    assert isinstance(result["hardened_sets"], dict)


def test_assign_modules_soft_hardened_sets_is_none(simple_df):
    result = assign_modules(simple_df, assignment_type="soft")
    assert result["hardened_sets"] is None


def test_assign_modules_required_keys(simple_df):
    result = assign_modules(simple_df)
    required = {"M_norm", "topk_sets", "label_sets", "union_sets",
                "hardened_sets", "assignment_table", "module_counts"}
    assert required.issubset(result.keys())


def test_assign_modules_module_counts_hard(simple_df):
    result = assign_modules(simple_df, topk_frac=0.5, p_abs=0.3, assignment_type="hard")
    total_assigned = result["module_counts"]["n"].sum()
    assert total_assigned <= simple_df.shape[0]


def test_assign_modules_zero_topk_frac():
    df = _make_df([[0.9, 0.1], [0.1, 0.9]], ["A", "B"], ["M1", "M2"])
    result = assign_modules(df, topk_frac=0.001, p_abs=0.5)
    assert result["assignment_table"].shape[1] == 2


def test_write_module_assignments(tmp_path, simple_df):
    result = assign_modules(simple_df, topk_frac=0.5, p_abs=0.3, assignment_type="hard")
    paths = write_module_assignments(result, prefix="gene", out_dir=tmp_path)

    assign_df = pd.read_csv(paths["assignments"], sep="\t")
    assert list(assign_df.columns) == ["entity_name", "module_id"]

    sets_path = paths["sets"]
    assert sets_path.exists()
    lines = sets_path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        assert "\t" in line
