from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def to_numpy(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def to_str_array(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaN p-values remain NaN."""
    pvals = np.asarray(pvals, dtype=float)
    fdr = np.full_like(pvals, np.nan, dtype=float)

    valid = np.where(np.isfinite(pvals))[0]
    if valid.size == 0:
        return fdr

    p = pvals[valid]
    order = np.argsort(p)
    ranked_p = p[order]

    n = len(ranked_p)
    adjusted = ranked_p * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    fdr_valid = np.empty_like(adjusted)
    fdr_valid[order] = adjusted
    fdr[valid] = fdr_valid

    return fdr


def read_label_file(label_file: str | Path) -> pd.DataFrame:
    """
    Read a two-column disease label file (sample_id, disease_label).

    Accepts files with or without a header row. Labels must be 0 (control) or 1 (disease).
    """
    df = pd.read_csv(str(label_file), sep=None, engine="python", header=None)

    if df.shape[1] < 2:
        raise ValueError("Disease label file must have at least two columns.")

    df = df.iloc[:, :2].copy()
    df.columns = ["sample_id", "disease_label"]

    first_label = str(df.iloc[0, 1]).strip().lower()
    if first_label not in {"0", "1", "0.0", "1.0"}:
        df = df.iloc[1:, :].copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df["disease_label"] = pd.to_numeric(df["disease_label"], errors="coerce")
    df = df.dropna(subset=["sample_id", "disease_label"])
    df["disease_label"] = df["disease_label"].astype(int)

    invalid = sorted(set(df["disease_label"]) - {0, 1})
    if invalid:
        raise ValueError(f"disease_label must be 0/1. Invalid labels found: {invalid}")

    if df["sample_id"].duplicated().any():
        duplicated = df.loc[df["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"Duplicate sample_id found in label file. Examples: {duplicated}")

    return df
