from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from rega.analysis._utils import to_numpy, to_str_array


def extract_re_tg(adata, min_score: float = 0.0) -> pd.DataFrame:
    """
    Extract the RE→gene regulatory network as a long-format DataFrame.

    Reads adata.layers['Net_RE_TG'] (gene × peak sparse matrix).

    Returns
    -------
    DataFrame with columns: target_gene, regulatory_element, score
    Filtered to score > min_score, sorted descending by score.
    """
    if "Net_RE_TG" not in adata.layers:
        raise ValueError("adata.layers['Net_RE_TG'] is missing.")

    net = adata.layers["Net_RE_TG"]
    if not sparse.issparse(net):
        net = sparse.csr_matrix(net)
    net = net.tocoo()

    gene_names = to_str_array(adata.obs_names)
    re_names = to_str_array(adata.var_names)

    scores = net.data.astype(np.float32)
    keep = np.isfinite(scores) & (scores > min_score)

    df = pd.DataFrame({
        "target_gene": gene_names[net.row[keep]],
        "regulatory_element": re_names[net.col[keep]],
        "score": scores[keep],
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def compute_tf_mean_expression(adata) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean expression across samples for each TF.

    TFs not present in adata.obs_names receive mean expression = 0.

    Returns
    -------
    tf_names : str array of length n_TF
    tf_mean_expr : float32 array of length n_TF
    """
    if "TF_names" not in adata.uns:
        raise ValueError("adata.uns['TF_names'] is missing.")
    if "GEX" not in adata.obsm:
        raise ValueError("adata.obsm['GEX'] is missing.")

    tf_names = to_str_array(adata.uns["TF_names"])
    gene_names = to_str_array(adata.obs_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    gex = to_numpy(adata.obsm["GEX"]).astype(np.float32)

    tf_mean_expr = np.zeros(len(tf_names), dtype=np.float32)
    for j, tf in enumerate(tf_names):
        if tf in gene_to_idx:
            tf_mean_expr[j] = float(np.nanmean(gex[gene_to_idx[tf], :]))

    return tf_names, tf_mean_expr


def extract_tf_tg(
    adata,
    out_path: Path,
    tf_names: np.ndarray,
    tf_mean_expr: np.ndarray,
    min_score: float = 0.0,
    chunk_size: int = 200,
) -> int:
    """
    Extract the expression-weighted TF→gene network and write to a TSV file.

    score = Net_TF_TG[gene, TF] * mean_TF_expression

    Processes TFs in chunks to bound peak memory usage. Returns total edge count.
    """
    if "Net_TF_TG" not in adata.obsm:
        raise ValueError("adata.obsm['Net_TF_TG'] is missing.")

    net_tf_tg = to_numpy(adata.obsm["Net_TF_TG"]).astype(np.float32)
    gene_names = to_str_array(adata.obs_names)

    if net_tf_tg.shape != (len(gene_names), len(tf_names)):
        raise ValueError(
            f"Net_TF_TG shape {net_tf_tg.shape} != "
            f"(n_genes={len(gene_names)}, n_TFs={len(tf_names)})"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_write = True
    total_edges = 0
    n_tfs = len(tf_names)

    for start in range(0, n_tfs, chunk_size):
        end = min(start + chunk_size, n_tfs)
        raw_chunk = net_tf_tg[:, start:end]
        weighted = raw_chunk * tf_mean_expr[start:end].reshape(1, -1)

        row_idx, col_idx = np.where(np.isfinite(weighted) & (weighted > min_score))
        if row_idx.size == 0:
            continue

        df = pd.DataFrame({
            "target_gene": gene_names[row_idx],
            "TF": tf_names[start + col_idx],
            "score": weighted[row_idx, col_idx].astype(np.float32),
        })
        df.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False
        total_edges += df.shape[0]

    if first_write:
        pd.DataFrame(columns=["target_gene", "TF", "score"]).to_csv(
            out_path, sep="\t", index=False
        )

    return total_edges


def extract_grn(
    adata,
    out_dir: Path,
    min_re_tg_score: float = 0.0,
    min_tf_tg_score: float = 0.0,
    chunk_size: int = 200,
) -> dict[str, Path]:
    """
    Extract both GRN tables (RE_TG and TF_TG) to an output directory.

    Writes:
      {out_dir}/RE_TG.tsv  — target_gene, regulatory_element, score
      {out_dir}/TF_TG.tsv  — target_gene, TF, score (expression-weighted)

    Returns a dict of output paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    re_tg_path = out_dir / "RE_TG.tsv"
    tf_tg_path = out_dir / "TF_TG.tsv"

    re_tg_df = extract_re_tg(adata, min_score=min_re_tg_score)
    re_tg_df.to_csv(re_tg_path, sep="\t", index=False)
    print(f"[grn] RE_TG edges: {re_tg_df.shape[0]}  -> {re_tg_path}")

    tf_names, tf_mean_expr = compute_tf_mean_expression(adata)
    n_edges = extract_tf_tg(
        adata,
        out_path=tf_tg_path,
        tf_names=tf_names,
        tf_mean_expr=tf_mean_expr,
        min_score=min_tf_tg_score,
        chunk_size=chunk_size,
    )
    print(f"[grn] TF_TG edges: {n_edges}  -> {tf_tg_path}")

    return {"RE_TG": re_tg_path, "TF_TG": tf_tg_path}
