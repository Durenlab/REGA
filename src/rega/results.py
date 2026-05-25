from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for rega.results but not installed. "
        "Install from https://pytorch.org/get-started/locally/ "
        "to get the correct CUDA/CPU build for your system."
    ) from e

__all__ = [
    "load_input_anndata",
    "load_rega_result",
    "consistency_checks",
    "assemble_W_learned",
    "assemble_B_learned",
    "compute_downstream",
    "build_result_anndata",
    "collect_results",
    "normalize_embedding",
]


def _to_str_array(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _report_missing(name: str, missing_list: list, top: int = 20) -> None:
    print(f"\n  Missing {name} ({len(missing_list)} total). "
          f"First {min(top, len(missing_list))}:")
    for x in missing_list[:top]:
        print(f"    - {x}")


def normalize_embedding(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Two-step L1 normalization: column-wise then row-wise."""
    E = np.asarray(E, dtype=np.float32)
    col_sum = np.abs(E).sum(axis=0, keepdims=True) + eps
    E = E / col_sum
    row_sum = np.abs(E).sum(axis=1, keepdims=True) + eps
    E = E / row_sum
    return E.astype(np.float32)


def load_input_anndata(
    input_h5ad: Path,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the input REGA AnnData (B5 output).

    Returns (adata, genes_master, RE_master, motif_names, sample_names, TF_names).
    Raises ValueError if required uns/obsm/varm keys are absent.
    Caller must ensure input_h5ad exists.
    """
    print(f"[load_input_anndata] Loading: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    print(f"  AnnData shape: {adata.X.shape}")

    for k in ("motif_names", "TF_names", "GEX_sample_names", "Motif_TF"):
        if k not in adata.uns:
            raise ValueError(f"adata.uns['{k}'] is missing.")
    if "GEX" not in adata.obsm:
        raise ValueError("adata.obsm['GEX'] is missing.")
    if "Peak_Motif" not in adata.varm:
        raise ValueError("adata.varm['Peak_Motif'] is missing.")

    genes_master = _to_str_array(adata.obs_names)
    RE_master    = _to_str_array(adata.var_names)
    motif_names  = _to_str_array(adata.uns["motif_names"])
    sample_names = _to_str_array(adata.uns["GEX_sample_names"])
    TF_names     = _to_str_array(adata.uns["TF_names"])

    print(f"  genes={len(genes_master)}, REs={len(RE_master)}, "
          f"motifs={len(motif_names)}, samples={len(sample_names)}, TFs={len(TF_names)}")
    return adata, genes_master, RE_master, motif_names, sample_names, TF_names


def load_rega_result(rega_result_path: Path) -> dict:
    """Load a REGA training checkpoint (final.pt) from model.py.

    Raises ValueError if required keys are absent.
    Caller must ensure the file exists.
    """
    print(f"[load_rega_result] Loading checkpoint: {rega_result_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            ckpt = torch.load(rega_result_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(rega_result_path, map_location="cpu")

    required = ("W_blocks", "B_blocks", "V", "H",
                "genes_blocks", "RE_blocks", "samples", "motifs")
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing required field '{k}'.")

    V_np = _to_numpy(ckpt["V"])
    H_np = _to_numpy(ckpt["H"])
    print(f"  blocks={len(ckpt['W_blocks'])}, V={V_np.shape}, "
          f"H={H_np.shape}, loss={ckpt.get('loss', 'N/A')}")
    return ckpt


def consistency_checks(
    ckpt: dict,
    genes_master: np.ndarray,
    RE_master: np.ndarray,
    motif_names: np.ndarray,
    sample_names: np.ndarray,
) -> tuple[list, list, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Validate that checkpoint blocks are consistent with the input AnnData master arrays.

    Returns (W_blocks, B_blocks, genes_blocks, RE_blocks, V, H) — V and H as float32 numpy.
    Raises ValueError on any inconsistency.
    """
    print("[consistency_checks] Validating block/master alignment ...")

    W_blocks     = ckpt["W_blocks"]
    B_blocks     = ckpt["B_blocks"]
    genes_blocks = [_to_str_array(g) for g in ckpt["genes_blocks"]]
    RE_blocks    = [_to_str_array(r) for r in ckpt["RE_blocks"]]
    V            = _to_numpy(ckpt["V"])
    H            = _to_numpy(ckpt["H"])
    rega_samples = _to_str_array(ckpt["samples"])
    rega_motifs  = _to_str_array(ckpt["motifs"])

    if not (len(W_blocks) == len(B_blocks) == len(genes_blocks) == len(RE_blocks)):
        raise ValueError(
            f"Block count mismatch: "
            f"W={len(W_blocks)}, B={len(B_blocks)}, "
            f"genes={len(genes_blocks)}, RE={len(RE_blocks)}"
        )

    for i, (W_b, B_b, g_b, r_b) in enumerate(
        zip(W_blocks, B_blocks, genes_blocks, RE_blocks)
    ):
        W_arr = _to_numpy(W_b)
        B_arr = _to_numpy(B_b)
        if W_arr.shape != (len(g_b), len(r_b)):
            raise ValueError(
                f"Block {i}: W shape {W_arr.shape} != "
                f"(n_genes={len(g_b)}, n_RE={len(r_b)})"
            )
        if B_arr.shape != (len(r_b), len(motif_names)):
            raise ValueError(
                f"Block {i}: B shape {B_arr.shape} != "
                f"(n_RE={len(r_b)}, n_motifs={len(motif_names)})"
            )

    if not np.array_equal(rega_samples, sample_names):
        raise ValueError(
            "Checkpoint 'samples' does not match input AnnData uns['GEX_sample_names']."
        )
    if not np.array_equal(rega_motifs, motif_names):
        raise ValueError(
            "Checkpoint 'motifs' does not match input AnnData uns['motif_names']."
        )

    if V.shape[0] != len(motif_names):
        raise ValueError(
            f"V.shape[0]={V.shape[0]} != len(motif_names)={len(motif_names)}"
        )
    if H.shape[1] != len(sample_names):
        raise ValueError(
            f"H.shape[1]={H.shape[1]} != len(sample_names)={len(sample_names)}"
        )
    if V.shape[1] != H.shape[0]:
        raise ValueError(
            f"V.shape[1]={V.shape[1]} != H.shape[0]={H.shape[0]} (K mismatch)"
        )

    gene_set = set(genes_master.tolist())
    re_set   = set(RE_master.tolist())

    missing_genes = [g for g_b in genes_blocks for g in g_b.tolist() if g not in gene_set]
    if missing_genes:
        _report_missing("genes (in block but not in input AnnData obs_names)", missing_genes)
        raise ValueError(
            f"{len(missing_genes)} genes from blocks are not in input AnnData obs_names."
        )

    missing_REs = [r for r_b in RE_blocks for r in r_b.tolist() if r not in re_set]
    if missing_REs:
        _report_missing("REs (in block but not in input AnnData var_names)", missing_REs)
        raise ValueError(
            f"{len(missing_REs)} REs from blocks are not in input AnnData var_names."
        )

    print(f"  All checks passed. {len(W_blocks)} blocks, K={V.shape[1]}")
    return W_blocks, B_blocks, genes_blocks, RE_blocks, V, H


def assemble_W_learned(
    W_blocks: list,
    genes_blocks: list[np.ndarray],
    RE_blocks: list[np.ndarray],
    genes_master: np.ndarray,
    RE_master: np.ndarray,
) -> sparse.csr_matrix:
    """Reconstruct the full gene×RE learned W as a sparse CSR matrix.

    Only nonzero entries from each block W are placed into the global matrix.
    Caller ensures all genes/REs in blocks are present in master arrays.
    """
    print("[assemble_W_learned] Assembling sparse W_learned ...")
    gene_to_idx = {g: i for i, g in enumerate(genes_master.tolist())}
    re_to_idx   = {r: i for i, r in enumerate(RE_master.tolist())}

    rows_list, cols_list, data_list = [], [], []
    for W_b, g_b, r_b in zip(W_blocks, genes_blocks, RE_blocks):
        W_arr = _to_numpy(W_b)
        gi = np.array([gene_to_idx[g] for g in g_b], dtype=np.int64)
        ri = np.array([re_to_idx[r]   for r in r_b], dtype=np.int64)
        local_rows, local_cols = np.where(W_arr != 0)
        if local_rows.size == 0:
            continue
        rows_list.append(gi[local_rows])
        cols_list.append(ri[local_cols])
        data_list.append(W_arr[local_rows, local_cols])

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list).astype(np.float32)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float32)

    W_learned = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(genes_master), len(RE_master)),
        dtype=np.float32,
    ).tocsr()
    print(f"  W_learned: {W_learned.shape}, nnz={W_learned.nnz}")
    return W_learned


def assemble_B_learned(
    B_blocks: list,
    RE_blocks: list[np.ndarray],
    RE_master: np.ndarray,
    motif_names: np.ndarray,
) -> np.ndarray:
    """Reconstruct the full RE×motif learned B as a dense float32 array.

    Caller ensures all REs in blocks are present in RE_master.
    """
    print("[assemble_B_learned] Assembling dense B_learned ...")
    re_to_idx = {r: i for i, r in enumerate(RE_master.tolist())}
    B_learned = np.zeros((len(RE_master), len(motif_names)), dtype=np.float32)
    for B_b, r_b in zip(B_blocks, RE_blocks):
        B_arr = _to_numpy(B_b)
        ri = np.array([re_to_idx[r] for r in r_b], dtype=np.int64)
        B_learned[ri, :] = B_arr
    print(f"  B_learned: {B_learned.shape}")
    return B_learned


def compute_downstream(
    adata: ad.AnnData,
    W_learned: sparse.csr_matrix,
    B_learned: np.ndarray,
    V: np.ndarray,
    H: np.ndarray,
    motif_names: np.ndarray,
    TF_names: np.ndarray,
    sample_names: np.ndarray,
) -> dict:
    """Compute all downstream regulatory network quantities.

    Returns a dict with keys: W0, BV, WBV, H_T, VH, BVH, w, y, d,
    Net_RE_TG, Net_TF_TG, WBV_norm, BV_norm, V_norm, H_T_norm, Motif_TF_PCC.

    Raises ValueError if Motif_TF shape is inconsistent or any TF name is absent from obs_names.
    """
    print("[compute_downstream] Computing downstream results ...")

    W0 = adata.X
    if not sparse.issparse(W0):
        W0 = sparse.csr_matrix(W0)
    W0 = W0.tocsr().astype(np.float32)

    BV  = (B_learned @ V).astype(np.float32)         # RE × K
    WBV = (W_learned @ BV).astype(np.float32)        # gene × K
    H_T = H.T.astype(np.float32)                     # sample × K
    VH  = (V @ H).astype(np.float32)                 # motif × sample
    BVH = (B_learned @ VH).astype(np.float32)        # RE × sample

    w = np.asarray(W_learned.sum(axis=0)).flatten().astype(np.float32)  # (RE,)
    y = BVH.sum(axis=1).astype(np.float32)                              # (RE,)
    d = (w * y).astype(np.float32)                                       # (RE,)

    W0_coo   = W0.tocoo()
    gene_idx = W0_coo.row
    re_idx   = W0_coo.col
    if gene_idx.size > 0:
        scores = np.einsum("ik,ik->i", WBV[gene_idx], BV[re_idx]).astype(np.float32)
    else:
        scores = np.array([], dtype=np.float32)
    Net_RE_TG = sparse.coo_matrix(
        (scores, (gene_idx, re_idx)),
        shape=adata.X.shape,
        dtype=np.float32,
    ).tocsr()

    C = np.asarray(adata.uns["Motif_TF"], dtype=np.float32)
    if C.shape != (len(motif_names), len(TF_names)):
        raise ValueError(
            f"Motif_TF shape {C.shape} != "
            f"(n_motifs={len(motif_names)}, n_TFs={len(TF_names)})"
        )
    Net_motif_TG = (WBV @ V.T).astype(np.float32)        # gene × motif
    Net_TF_TG   = (Net_motif_TG @ C).astype(np.float32)  # gene × TF

    WBV_norm = normalize_embedding(WBV)
    BV_norm  = normalize_embedding(BV)
    V_norm   = normalize_embedding(V)
    H_T_norm = normalize_embedding(H_T)

    gex = np.asarray(adata.obsm["GEX"], dtype=np.float32)  # gene × sample
    obs_to_idx = {g: i for i, g in enumerate(list(adata.obs_names))}
    missing_TFs = [t for t in TF_names.tolist() if t not in obs_to_idx]
    if missing_TFs:
        _report_missing("TFs (not in obs_names)", missing_TFs)
        raise ValueError(
            f"{len(missing_TFs)} TF names are not in input AnnData obs_names."
        )
    tf_idx  = np.array([obs_to_idx[t] for t in TF_names.tolist()], dtype=np.int64)
    TF_expr = gex[tf_idx, :]  # TF × sample

    def _zscore_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        M  = M.astype(np.float32)
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True) + eps
        return (M - mu) / sd

    VH_z = _zscore_rows(VH)       # motif × sample
    TF_z = _zscore_rows(TF_expr)  # TF × sample
    Motif_TF_PCC = ((VH_z @ TF_z.T) / float(VH.shape[1])).astype(np.float32)
    Motif_TF_PCC = np.where(Motif_TF_PCC < 0, 0.0, Motif_TF_PCC).astype(np.float32)

    print(f"  BV={BV.shape}, WBV={WBV.shape}, VH={VH.shape}, BVH={BVH.shape}, "
          f"Net_RE_TG nnz={Net_RE_TG.nnz}, Net_TF_TG={Net_TF_TG.shape}, "
          f"Motif_TF_PCC={Motif_TF_PCC.shape} (negatives clipped to 0)")
    return {
        "W0": W0,
        "BV": BV, "WBV": WBV, "H_T": H_T,
        "VH": VH, "BVH": BVH,
        "w": w, "y": y, "d": d,
        "Net_RE_TG": Net_RE_TG,
        "Net_TF_TG": Net_TF_TG,
        "WBV_norm": WBV_norm, "BV_norm": BV_norm,
        "V_norm": V_norm, "H_T_norm": H_T_norm,
        "Motif_TF_PCC": Motif_TF_PCC,
    }


def build_result_anndata(
    adata: ad.AnnData,
    W_learned: sparse.csr_matrix,
    B_learned: np.ndarray,
    V: np.ndarray,
    H: np.ndarray,
    results: dict,
    ckpt: dict,
    out_h5ad: Path,
    motif_names: np.ndarray,
    TF_names: np.ndarray,
    sample_names: np.ndarray,
) -> ad.AnnData:
    """Assemble the result AnnData and write to out_h5ad.

    Raises ValueError if any assembled array has an unexpected shape.
    Caller must ensure out_h5ad.parent exists.
    """
    print("[build_result_anndata] Assembling result AnnData ...")
    adata_result = adata.copy()

    adata_result.X                         = W_learned
    adata_result.layers["W0"]              = results["W0"]
    adata_result.layers["Net_RE_TG"]       = results["Net_RE_TG"]

    adata_result.var["RE_strength_w"]      = results["w"].astype(np.float32)
    adata_result.var["RE_activity_y"]      = results["y"].astype(np.float32)
    adata_result.var["RE_importance"]      = results["d"].astype(np.float32)

    adata_result.obsm["WBV"]              = results["WBV"]
    adata_result.obsm["WBV_norm"]         = results["WBV_norm"]
    adata_result.obsm["Net_TF_TG"]        = results["Net_TF_TG"]

    adata_result.varm["B_learned"]             = B_learned
    adata_result.varm["BV"]                    = results["BV"]
    adata_result.varm["BV_norm"]               = results["BV_norm"]
    adata_result.varm["Peak_Activity_BVH"]     = results["BVH"]

    adata_result.uns["V"]                      = V.astype(np.float32)
    adata_result.uns["V_norm"]                 = results["V_norm"]
    adata_result.uns["H"]                      = H.astype(np.float32)
    adata_result.uns["H_T"]                    = results["H_T"]
    adata_result.uns["H_T_norm"]               = results["H_T_norm"]
    adata_result.uns["Motif_Activity_VH"]      = results["VH"]
    adata_result.uns["Motif_TF_PCC"]           = results["Motif_TF_PCC"]
    adata_result.uns["Module_names"]           = np.array(
        [f"Module_{i + 1}" for i in range(V.shape[1])]
    )
    adata_result.uns["REGA_loss"]              = float(ckpt.get("loss", float("nan")))
    adata_result.uns["REGA_step"]              = int(ckpt.get("step", -1))
    adata_result.uns["REGA_K"]                 = int(ckpt.get("K", V.shape[1]))
    adata_result.uns["REGA_training_info"]     = ckpt.get("training_info", {}) or {}

    n_genes = adata_result.n_obs
    n_REs   = adata_result.n_vars
    n_K     = V.shape[1]
    n_motif = len(motif_names)
    n_TF    = len(TF_names)
    n_samp  = len(sample_names)

    shape_checks = [
        (adata_result.X.shape,                         (n_genes, n_REs),   "X"),
        (adata_result.layers["W0"].shape,              (n_genes, n_REs),   "W0"),
        (adata_result.layers["Net_RE_TG"].shape,       (n_genes, n_REs),   "Net_RE_TG"),
        (adata_result.obsm["WBV"].shape,               (n_genes, n_K),     "WBV"),
        (adata_result.obsm["Net_TF_TG"].shape,         (n_genes, n_TF),    "Net_TF_TG"),
        (adata_result.varm["B_learned"].shape,         (n_REs,   n_motif), "B_learned"),
        (adata_result.varm["BV"].shape,                (n_REs,   n_K),     "BV"),
        (adata_result.varm["Peak_Activity_BVH"].shape, (n_REs,   n_samp),  "BVH"),
        (adata_result.uns["V"].shape,                  (n_motif, n_K),     "V"),
        (adata_result.uns["H"].shape,                  (n_K,     n_samp),  "H"),
        (adata_result.uns["Motif_Activity_VH"].shape,  (n_motif, n_samp),  "VH"),
        (adata_result.uns["Motif_TF_PCC"].shape,       (n_motif, n_TF),    "Motif_TF_PCC"),
    ]
    for actual, expected, name in shape_checks:
        if actual != expected:
            raise ValueError(f"Shape mismatch for '{name}': {actual} != {expected}")

    adata_result.write_h5ad(out_h5ad)
    print(f"  Saved -> {out_h5ad}")
    return adata_result


def collect_results(
    input_h5ad: Path,
    rega_result_path: Path,
    out_h5ad: Path,
) -> ad.AnnData:
    """Run the full REGA result collection pipeline.

    Loads the B5 input h5ad and B8 checkpoint, assembles W_learned and B_learned,
    computes downstream regulatory quantities, and writes the result AnnData to out_h5ad.

    Caller must ensure out_h5ad.parent exists.
    """
    (adata, genes_master, RE_master,
     motif_names, sample_names, TF_names) = load_input_anndata(input_h5ad)

    ckpt = load_rega_result(rega_result_path)

    (W_blocks, B_blocks, genes_blocks, RE_blocks, V, H) = consistency_checks(
        ckpt, genes_master, RE_master, motif_names, sample_names
    )

    W_learned = assemble_W_learned(
        W_blocks, genes_blocks, RE_blocks, genes_master, RE_master
    )
    B_learned = assemble_B_learned(B_blocks, RE_blocks, RE_master, motif_names)

    results = compute_downstream(
        adata, W_learned, B_learned, V, H, motif_names, TF_names, sample_names
    )

    return build_result_anndata(
        adata, W_learned, B_learned, V, H,
        results, ckpt, out_h5ad,
        motif_names, TF_names, sample_names,
    )
