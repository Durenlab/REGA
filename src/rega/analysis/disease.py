from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rega.analysis._utils import bh_fdr, read_label_file, to_str_array


def load_module_activity(adata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the sample × module activity matrix (H_T) from a REGA result h5ad.

    Returns
    -------
    H_T : float32 array of shape (n_sample, n_module)
    sample_names : str array of length n_sample
    module_names : str array of length n_module
    """
    if "H_T" not in adata.uns:
        raise ValueError("adata.uns['H_T'] is missing.")
    if "GEX_sample_names" not in adata.uns:
        raise ValueError("adata.uns['GEX_sample_names'] is missing.")

    H_T = np.asarray(adata.uns["H_T"], dtype=np.float32)
    sample_names = to_str_array(adata.uns["GEX_sample_names"])

    if H_T.ndim != 2:
        raise ValueError(f"adata.uns['H_T'] must be 2D, got shape {H_T.shape}")
    if H_T.shape[0] != len(sample_names):
        raise ValueError(
            f"H_T rows {H_T.shape[0]} != len(GEX_sample_names) {len(sample_names)}"
        )

    if "Module_names" in adata.uns:
        module_names = to_str_array(adata.uns["Module_names"])
        if len(module_names) != H_T.shape[1]:
            raise ValueError(
                f"len(Module_names) {len(module_names)} != H_T columns {H_T.shape[1]}"
            )
    else:
        module_names = np.asarray(
            [f"Module_{i + 1}" for i in range(H_T.shape[1])], dtype=str
        )

    return H_T, sample_names, module_names


def identify_disease_modules(
    adata,
    label_df: pd.DataFrame,
    fdr_cutoff: float = 0.05,
) -> pd.DataFrame:
    """
    Identify disease-associated REGA regulatory modules.

    For each module, computes the Spearman correlation between sample-level module
    activity (H_T[:, module]) and the binary disease label, then applies BH-FDR.

    Parameters
    ----------
    adata : AnnData with REGA result matrices
    label_df : DataFrame with columns sample_id (str) and disease_label (0/1)
    fdr_cutoff : BH-FDR cutoff for the 'disease_module_significant' flag

    Returns
    -------
    DataFrame sorted by significance then FDR, with columns:
      module, spearman_rho, p_value, FDR, disease_module_significant,
      n_samples, n_control, n_disease,
      mean_activity_control, mean_activity_disease, delta_activity
    """
    H_T, sample_names, module_names = load_module_activity(adata)

    sample_to_idx = {s: i for i, s in enumerate(sample_names)}
    matched = label_df[label_df["sample_id"].isin(sample_to_idx)].copy()

    if matched.shape[0] < 3:
        raise ValueError("Fewer than 3 matched samples. Cannot perform correlation robustly.")
    if matched["disease_label"].nunique() < 2:
        raise ValueError("Matched disease_label contains only one class. Need both 0 and 1.")

    matched_idx = np.asarray([sample_to_idx[s] for s in matched["sample_id"]], dtype=int)
    y = matched["disease_label"].to_numpy(dtype=float)
    H_sub = H_T[matched_idx, :]

    n_control = int(np.sum(y == 0))
    n_disease = int(np.sum(y == 1))

    rows = []
    for j, module in enumerate(module_names):
        x = H_sub[:, j].astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x_use, y_use = x[valid], y[valid]

        if len(x_use) < 3 or len(np.unique(x_use)) < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(x_use, y_use)

        rows.append({
            "module": module,
            "spearman_rho": rho,
            "p_value": pval,
            "n_samples": int(len(x_use)),
            "n_control": n_control,
            "n_disease": n_disease,
            "mean_activity_control": float(np.nanmean(x[y == 0])),
            "mean_activity_disease": float(np.nanmean(x[y == 1])),
            "delta_activity": float(np.nanmean(x[y == 1]) - np.nanmean(x[y == 0])),
        })

    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p_value"].to_numpy())
    res["disease_module_significant"] = res["FDR"] < fdr_cutoff

    return res.sort_values(
        ["disease_module_significant", "FDR", "p_value"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def write_disease_module_results(
    result_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Write disease module association results to TSV files.

    Writes:
      disease_module_association_all.tsv
      disease_module_significant.tsv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "disease_module_association_all.tsv"
    sig_path = out_dir / "disease_module_significant.tsv"

    result_df.to_csv(all_path, sep="\t", index=False)
    result_df.loc[result_df["disease_module_significant"]].to_csv(sig_path, sep="\t", index=False)

    return {"all": all_path, "significant": sig_path}
