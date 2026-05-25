from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rega.analysis._utils import to_numpy, to_str_array


# ── internal helpers ───────────────────────────────────────────────────────────

def _get_module_names(adata, k: int) -> np.ndarray:
    if "Module_names" in adata.uns:
        names = to_str_array(adata.uns["Module_names"])
        if len(names) != k:
            raise ValueError(
                f"adata.uns['Module_names'] length {len(names)} != matrix n_modules {k}"
            )
        return names
    return np.asarray([f"Module_{i + 1}" for i in range(k)], dtype=str)


def _make_matrix_df(
    matrix,
    row_names: np.ndarray,
    module_names: np.ndarray,
    label: str,
) -> pd.DataFrame:
    matrix = to_numpy(matrix).astype(np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label}: matrix must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] != len(row_names):
        raise ValueError(
            f"{label}: matrix rows {matrix.shape[0]} != row_names length {len(row_names)}"
        )
    if matrix.shape[1] != len(module_names):
        raise ValueError(
            f"{label}: matrix columns {matrix.shape[1]} != module_names length {len(module_names)}"
        )
    return pd.DataFrame(matrix, index=row_names.astype(str), columns=module_names.astype(str))


def _unique_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


# ── public sub-functions ───────────────────────────────────────────────────────

def assign_topk_sets(M: pd.DataFrame, topk: int) -> dict[str, list[str]]:
    """
    For each module, select the top-K entities by score.

    Uses argpartition + stable sort to match R order(..., decreasing=TRUE) with
    ties.method='first'.
    """
    modules = list(M.columns)
    entities = np.asarray(M.index).astype(str)
    mat = M.to_numpy(dtype=np.float32, copy=True)
    mat[~np.isfinite(mat)] = -np.inf

    out: dict[str, list[str]] = {mn: [] for mn in modules}
    if topk <= 0:
        return out

    n = mat.shape[0]
    k = min(topk, n)

    for j, mn in enumerate(modules):
        v = mat[:, j]
        finite_idx = np.where(np.isfinite(v))[0]
        if finite_idx.size == 0:
            continue
        k_j = min(k, finite_idx.size)
        vf = v[finite_idx]
        part = np.argpartition(-vf, k_j - 1)[:k_j]
        selected_local = part[np.argsort(-vf[part], kind="mergesort")]
        out[mn] = entities[finite_idx[selected_local]].tolist()

    return out


def harden_sets_by_score(
    M_score: pd.DataFrame,
    set_list: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Resolve conflicts so each entity is assigned to at most one module (highest score wins).

    Equivalent to R harden_sets_by_score(). np.argmax returns the first max,
    matching R which.max behavior.
    """
    modules = list(M_score.columns)
    mat = M_score.to_numpy(dtype=np.float32, copy=True)
    mat[~np.isfinite(mat)] = -np.inf

    entity_to_idx = {e: i for i, e in enumerate(M_score.index.astype(str))}
    module_to_idx = {m: j for j, m in enumerate(modules)}

    set_list_clean = {mn: list(set_list.get(mn, []) or []) for mn in modules}

    ent_to_cand_mods: dict[str, list[str]] = {}
    entity_order: list[str] = []

    for mn in modules:
        seen: set[str] = set()
        for ent in set_list_clean[mn]:
            if ent in seen or ent not in entity_to_idx:
                continue
            seen.add(ent)
            if ent not in ent_to_cand_mods:
                ent_to_cand_mods[ent] = []
                entity_order.append(ent)
            ent_to_cand_mods[ent].append(mn)

    out: dict[str, list[str]] = {mn: [] for mn in modules}

    for ent in entity_order:
        cand_mods = ent_to_cand_mods[ent]
        if not cand_mods:
            continue
        i = entity_to_idx[ent]
        cand_idx = [module_to_idx[mn] for mn in cand_mods]
        scores = mat[i, cand_idx]
        if np.all(scores == -np.inf):
            continue
        best_mod = cand_mods[int(np.argmax(scores))]
        out[best_mod].append(ent)

    return out


def assign_label_absolute(
    M: pd.DataFrame,
    p_abs: float = 0.25,
) -> tuple[np.ndarray, float]:
    """
    Assign each entity to its highest-scoring module if that score >= p_abs.

    Returns (labels, p_abs) where labels[i] is the module index or -1 if unassigned.
    Equivalent to R assign_label_L1L1(..., mode='absolute') on a pre-normalized matrix.
    """
    mat = M.to_numpy(dtype=np.float32, copy=True)
    mat[~np.isfinite(mat)] = -np.inf

    best_idx = np.argmax(mat, axis=1)
    best_score = mat[np.arange(mat.shape[0]), best_idx]

    labels = np.full(mat.shape[0], fill_value=-1, dtype=np.int64)
    keep = np.where(np.isfinite(best_score) & (best_score >= p_abs))[0]
    labels[keep] = best_idx[keep]

    return labels, p_abs


def assign_modules(
    M: pd.DataFrame,
    topk_frac: float = 0.1,
    p_abs: float = 0.25,
    assignment_type: Literal["hard", "soft"] = "hard",
) -> dict[str, object]:
    """
    Assign entities (rows of M) to regulatory modules (columns of M).

    M must already be L1L1-normalized (use *_norm matrices from rega_result.h5ad).

    Algorithm
    ---------
    1. topK selection per module (topk = ceil(n_entities * topk_frac))
    2. Intermediate harden of topK sets (each entity to its best module)
    3. Absolute-threshold label assignment (entity to max-score module if >= p_abs)
    4. union_sets = topK_hardened ∪ label_sets
    5. hard mode: final harden of union_sets (each entity in exactly one module)
       soft mode: no final harden (entity may appear in multiple modules)

    Parameters
    ----------
    M : entity × module DataFrame, already normalized
    topk_frac : fraction of entities to select per module (0 < topk_frac <= 1)
    p_abs : absolute threshold on normalized score for label assignment
    assignment_type : "hard" (unique module per entity) or "soft" (multi-module allowed)

    Returns
    -------
    dict with keys:
      M_norm           : the input M (already normalized)
      topk_sets        : dict[module -> list[entity]]
      label_sets       : dict[module -> list[entity]]
      union_sets       : dict[module -> list[entity]]
      hardened_sets    : dict[module -> list[entity]] in hard mode; None in soft mode
      assignment_table : DataFrame with columns entity_name, module_id
      module_counts    : DataFrame with columns module, n
    """
    if assignment_type not in ("hard", "soft"):
        raise ValueError(f"assignment_type must be 'hard' or 'soft', got {assignment_type!r}")

    modules = list(M.columns)
    entities = np.asarray(M.index).astype(str)
    n_entities = len(entities)

    topk = max(1, math.ceil(n_entities * topk_frac))

    # step 1-2: topK + intermediate harden
    topk_sets = assign_topk_sets(M, topk=topk)
    topk_hardened = harden_sets_by_score(M, topk_sets)

    # step 3: label sets
    labels, _ = assign_label_absolute(M, p_abs=p_abs)
    label_sets: dict[str, list[str]] = {mn: [] for mn in modules}
    for k_idx, mn in enumerate(modules):
        idx = np.where(labels == k_idx)[0]
        label_sets[mn] = entities[idx].tolist()

    # step 4: union
    union_sets: dict[str, list[str]] = {}
    for mn in modules:
        union_sets[mn] = _unique_preserve_order(
            list(topk_hardened.get(mn, [])) + list(label_sets.get(mn, []))
        )

    # step 5: hard vs soft
    if assignment_type == "hard":
        hardened_sets: dict[str, list[str]] | None = harden_sets_by_score(M, union_sets)
        active_sets = hardened_sets
    else:
        hardened_sets = None
        active_sets = union_sets

    # build assignment_table
    rows: list[dict[str, str]] = []
    for mn, ents in active_sets.items():  # type: ignore[union-attr]
        for ent in ents:
            rows.append({"entity_name": ent, "module_id": mn})
    assignment_table = pd.DataFrame(rows, columns=["entity_name", "module_id"])

    # build module_counts
    module_counts = pd.DataFrame(
        {"module": modules, "n": [len(active_sets[mn]) for mn in modules]}  # type: ignore[index]
    )

    return {
        "M_norm": M,
        "topk_sets": topk_sets,
        "label_sets": label_sets,
        "union_sets": union_sets,
        "hardened_sets": hardened_sets,
        "assignment_table": assignment_table,
        "module_counts": module_counts,
    }


# ── h5ad loader ────────────────────────────────────────────────────────────────

def load_entity_matrices(adata) -> dict[str, pd.DataFrame]:
    """
    Extract the four pre-normalized entity × module matrices from a REGA result h5ad.

    Returns a dict with keys 'gene', 'peak', 'motif', 'sample'.
    All *_norm matrices are used directly without re-normalization.
    """
    required = [
        ("obsm", "WBV_norm"),
        ("varm", "BV_norm"),
        ("uns", "V_norm"),
        ("uns", "H_T_norm"),
    ]
    for loc, key in required:
        if key not in getattr(adata, loc):
            raise ValueError(f"Missing adata.{loc}['{key}']")

    if "motif_names" not in adata.uns:
        raise ValueError("Missing adata.uns['motif_names']")
    if "GEX_sample_names" not in adata.uns:
        raise ValueError("Missing adata.uns['GEX_sample_names']")

    k = to_numpy(adata.obsm["WBV_norm"]).shape[1]
    module_names = _get_module_names(adata, k)

    matrices: dict[str, pd.DataFrame] = {}

    matrices["gene"] = _make_matrix_df(
        adata.obsm["WBV_norm"], to_str_array(adata.obs_names), module_names, "gene/WBV_norm"
    )
    matrices["peak"] = _make_matrix_df(
        adata.varm["BV_norm"], to_str_array(adata.var_names), module_names, "peak/BV_norm"
    )
    matrices["motif"] = _make_matrix_df(
        adata.uns["V_norm"], to_str_array(adata.uns["motif_names"]), module_names, "motif/V_norm"
    )
    matrices["sample"] = _make_matrix_df(
        adata.uns["H_T_norm"],
        to_str_array(adata.uns["GEX_sample_names"]),
        module_names,
        "sample/H_T_norm",
    )

    return matrices


# ── output writers ─────────────────────────────────────────────────────────────

def write_module_assignments(
    result: dict,
    prefix: str,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Write module assignment outputs for one entity type.

    Files written:
      {prefix}_module_assignments.tsv : long format (entity_name, module_id)
      {prefix}_module_sets.tsv        : GMT format (module_id<TAB>entity1<TAB>entity2...)

    Uses result['hardened_sets'] if present (hard mode), else result['union_sets'] (soft mode).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assign_path = out_dir / f"{prefix}_module_assignments.tsv"
    result["assignment_table"].to_csv(assign_path, sep="\t", index=False)

    active_sets: dict[str, list[str]] = (
        result["hardened_sets"] if result["hardened_sets"] is not None else result["union_sets"]
    )

    sets_path = out_dir / f"{prefix}_module_sets.tsv"
    with open(sets_path, "w") as fh:
        for mn, ents in active_sets.items():
            parts = [mn] + list(ents)
            fh.write("\t".join(parts) + "\n")

    return {"assignments": assign_path, "sets": sets_path}
