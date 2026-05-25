from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rega.analysis._utils import to_str_array


def extract_re_importance(adata) -> pd.DataFrame:
    """
    Extract RE (regulatory element) importance scores from a REGA result h5ad.

    Reads adata.var['RE_importance']. Returns a DataFrame sorted descending by score.

    Returns
    -------
    DataFrame with columns: regulatory_element, RE_importance
    """
    if "RE_importance" not in adata.var.columns:
        raise ValueError("adata.var['RE_importance'] is missing.")

    df = pd.DataFrame({
        "regulatory_element": to_str_array(adata.var_names),
        "RE_importance": np.asarray(adata.var["RE_importance"], dtype=np.float32),
    })
    return df.sort_values("RE_importance", ascending=False).reset_index(drop=True)


def write_re_importance(adata, out_tsv: Path) -> Path:
    """
    Extract RE importance scores and write to a TSV file.

    Returns the output path.
    """
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    df = extract_re_importance(adata)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[re_score] REs exported: {df.shape[0]}  -> {out_tsv}")

    return out_tsv
