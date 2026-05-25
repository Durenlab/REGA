from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from rega.analysis._utils import bh_fdr, read_label_file, to_numpy, to_str_array


def load_driver_tf_matrices(adata) -> dict[str, object]:
    """
    Load all matrices required for driver TF identification from a REGA result h5ad.

    Returns a dict with keys:
      motif_activity, motif_tf, motif_tf_pcc, tf_names, motif_names,
      sample_names, gene_names, gex
    """
    required_uns = [
        "Motif_Activity_VH",
        "Motif_TF",
        "Motif_TF_PCC",
        "TF_names",
        "motif_names",
        "GEX_sample_names",
    ]
    for key in required_uns:
        if key not in adata.uns:
            raise ValueError(f"adata.uns['{key}'] is missing.")
    if "GEX" not in adata.obsm:
        raise ValueError("adata.obsm['GEX'] is missing.")

    motif_activity = to_numpy(adata.uns["Motif_Activity_VH"]).astype(np.float32)
    motif_tf = to_numpy(adata.uns["Motif_TF"]).astype(np.float32)
    motif_tf_pcc = to_numpy(adata.uns["Motif_TF_PCC"]).astype(np.float32)
    tf_names = to_str_array(adata.uns["TF_names"])
    motif_names = to_str_array(adata.uns["motif_names"])
    sample_names = to_str_array(adata.uns["GEX_sample_names"])
    gene_names = to_str_array(adata.obs_names)
    gex = to_numpy(adata.obsm["GEX"]).astype(np.float32)

    if motif_activity.shape != (len(motif_names), len(sample_names)):
        raise ValueError(
            f"Motif_Activity_VH shape {motif_activity.shape} != "
            f"(n_motifs={len(motif_names)}, n_samples={len(sample_names)})"
        )
    if motif_tf.shape != (len(motif_names), len(tf_names)):
        raise ValueError(
            f"Motif_TF shape {motif_tf.shape} != "
            f"(n_motifs={len(motif_names)}, n_TFs={len(tf_names)})"
        )
    if motif_tf_pcc.shape != (len(motif_names), len(tf_names)):
        raise ValueError(
            f"Motif_TF_PCC shape {motif_tf_pcc.shape} != "
            f"(n_motifs={len(motif_names)}, n_TFs={len(tf_names)})"
        )
    if gex.shape != (len(gene_names), len(sample_names)):
        raise ValueError(
            f"GEX shape {gex.shape} != "
            f"(n_genes={len(gene_names)}, n_samples={len(sample_names)})"
        )

    return {
        "motif_activity": motif_activity,
        "motif_tf": motif_tf,
        "motif_tf_pcc": motif_tf_pcc,
        "tf_names": tf_names,
        "motif_names": motif_names,
        "sample_names": sample_names,
        "gene_names": gene_names,
        "gex": gex,
    }


def extract_tf_expression(
    gex: np.ndarray,
    gene_names: np.ndarray,
    tf_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a TF × sample expression sub-matrix from the full gene × sample GEX matrix.

    TFs absent from gene_names receive expression = 0 (and found_mask[i] = False).

    Returns
    -------
    tf_expression : float32 array of shape (n_TF, n_sample)
    found_mask : bool array of shape (n_TF,)
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_tfs = len(tf_names)
    n_samples = gex.shape[1]

    tf_expression = np.zeros((n_tfs, n_samples), dtype=np.float32)
    found_mask = np.zeros(n_tfs, dtype=bool)

    for i, tf in enumerate(tf_names):
        if tf in gene_to_idx:
            tf_expression[i, :] = gex[gene_to_idx[tf], :]
            found_mask[i] = True

    return tf_expression, found_mask


def compute_tf_activity(
    motif_activity: np.ndarray,
    motif_tf: np.ndarray,
    motif_tf_pcc: np.ndarray,
    tf_expression: np.ndarray,
    pcc_cutoff: float,
) -> np.ndarray:
    """
    Compute expression-weighted TF activity (TF × sample).

    weight = Motif_TF * Motif_TF_PCC, zeroed where PCC < pcc_cutoff.
    TF_activity_motif = weight.T @ Motif_Activity_VH
    TF_activity = TF_expression * TF_activity_motif  (element-wise, TF × sample)
    """
    weight = motif_tf * motif_tf_pcc
    weight[motif_tf_pcc < pcc_cutoff] = 0.0
    weight[~np.isfinite(weight)] = 0.0

    tf_activity_motif = (weight.T @ motif_activity).astype(np.float32)
    return (tf_expression * tf_activity_motif).astype(np.float32)


def identify_driver_tfs(
    adata,
    label_df: pd.DataFrame,
    pcc_cutoff: float = 0.1,
    fdr_cutoff: float = 0.05,
) -> pd.DataFrame:
    """
    Identify disease-associated driver TFs from a REGA result h5ad.

    Parameters
    ----------
    adata : AnnData with REGA result matrices
    label_df : DataFrame with columns sample_id (str) and disease_label (0/1)
    pcc_cutoff : Motif_TF_PCC threshold; motif-TF pairs below this are zeroed
    fdr_cutoff : BH-FDR cutoff for the 'driver_TF_significant' flag

    Returns
    -------
    DataFrame sorted by significance then FDR, with columns:
      TF, spearman_rho, p_value, FDR, driver_TF_significant,
      n_samples, n_control, n_disease, TF_found_in_GEX,
      mean_activity_control, mean_activity_disease, delta_activity
    """
    data = load_driver_tf_matrices(adata)
    tf_names = data["tf_names"]
    sample_names = data["sample_names"]

    tf_expression, tf_found_mask = extract_tf_expression(
        data["gex"], data["gene_names"], tf_names
    )
    tf_activity = compute_tf_activity(
        data["motif_activity"],
        data["motif_tf"],
        data["motif_tf_pcc"],
        tf_expression,
        pcc_cutoff=pcc_cutoff,
    )

    sample_to_idx = {s: i for i, s in enumerate(sample_names)}
    matched = label_df[label_df["sample_id"].isin(sample_to_idx)].copy()

    if matched.shape[0] < 3:
        raise ValueError("Fewer than 3 matched samples. Cannot perform correlation robustly.")
    if matched["disease_label"].nunique() < 2:
        raise ValueError("Matched disease_label contains only one class. Need both 0 and 1.")

    matched_idx = np.asarray([sample_to_idx[s] for s in matched["sample_id"]], dtype=int)
    y = matched["disease_label"].to_numpy(dtype=float)
    tf_activity_sub = tf_activity[:, matched_idx]

    n_control = int(np.sum(y == 0))
    n_disease = int(np.sum(y == 1))

    rows = []
    for i, tf in enumerate(tf_names):
        x = tf_activity_sub[i, :].astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x_use, y_use = x[valid], y[valid]

        if len(x_use) < 3 or len(np.unique(x_use)) < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(x_use, y_use)

        rows.append({
            "TF": tf,
            "spearman_rho": rho,
            "p_value": pval,
            "n_samples": int(len(x_use)),
            "n_control": n_control,
            "n_disease": n_disease,
            "TF_found_in_GEX": bool(tf_found_mask[i]),
            "mean_activity_control": float(np.nanmean(x[y == 0])),
            "mean_activity_disease": float(np.nanmean(x[y == 1])),
            "delta_activity": float(np.nanmean(x[y == 1]) - np.nanmean(x[y == 0])),
        })

    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p_value"].to_numpy())
    res["driver_TF_significant"] = res["FDR"] < fdr_cutoff

    return res.sort_values(
        ["driver_TF_significant", "FDR", "p_value"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def write_driver_tf_results(
    result_df: pd.DataFrame,
    out_dir: Path,
    tf_activity: np.ndarray | None = None,
    tf_names: np.ndarray | None = None,
    sample_names: np.ndarray | None = None,
) -> dict[str, Path]:
    """
    Write driver TF analysis results to TSV files.

    Always writes:
      disease_driver_TF_association_all.tsv
      disease_driver_TF_significant.tsv

    If tf_activity, tf_names, sample_names are all provided, also writes:
      TF_activity_matrix.tsv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "disease_driver_TF_association_all.tsv"
    sig_path = out_dir / "disease_driver_TF_significant.tsv"

    result_df.to_csv(all_path, sep="\t", index=False)
    result_df.loc[result_df["driver_TF_significant"]].to_csv(sig_path, sep="\t", index=False)

    paths: dict[str, Path] = {"all": all_path, "significant": sig_path}

    if tf_activity is not None and tf_names is not None and sample_names is not None:
        act_path = out_dir / "TF_activity_matrix.tsv"
        pd.DataFrame(tf_activity, index=tf_names, columns=sample_names).to_csv(
            act_path, sep="\t", index_label="TF"
        )
        paths["activity"] = act_path

    return paths
