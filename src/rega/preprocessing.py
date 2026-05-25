"""
Gene expression preprocessing for the REGA pipeline.

Public API
----------
preprocess_expression(expr_file, meta_file, input_type, ...) -> pd.DataFrame

Sub-functions (individually importable)
----------------------------------------
load_metadata, load_expression, normalize_to_target_scale,
filter_low_expression, align_samples, covariate_correction, post_process
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================

PRIOR_COUNT: int = 1
_LOW_EXPR_FRAC: float = 0.05

_VALID_INPUT_TYPES = frozenset({"counts", "cpm", "logcpm", "tpm", "logtpm"})

# Empirical defaults applied *after* scale conversion.
# Applied directly to the post-conversion matrix without further scaling.
# Adjust based on your data distribution.
_DEFAULT_THRESHOLD: dict[str, float] = {
    "counts":  1.0,  # logCPM ≥ 1  (≈ CPM > 1 before log)
    "cpm":     1.0,  # log2(CPM+1) ≥ 1
    "logcpm":  1.0,  # logCPM ≥ 1
    "logtpm":  1.0,  # log(TPM) ≥ 1
    "tpm":     0.5,  # linear TPM ≥ 0.5 (linear scale; most genes expressed above 0.5 TPM)
}

_NA_VALUES = [
    "", " ", "NA", "N/A", "n/a", "na",
    "NaN", "nan", "NULL", "null", ".",
]


# ============================================================
# Public API
# ============================================================

def preprocess_expression(
    expr_file: str | Path,
    meta_file: str | Path | None = None,
    input_type: str = "counts",
    min_expression_threshold: float | None = None,
    min_expression_frac: float = 0.05,
    remove_mt_genes: bool = True,
    mt_gene_prefix: tuple[str, ...] = ("MT-", "mt-"),
) -> pd.DataFrame:
    """
    Preprocess a gene expression matrix for the REGA pipeline.

    Parameters
    ----------
    expr_file : str or Path
        Path to the expression matrix CSV (genes × samples).
        Row index = gene symbols; column headers = sample IDs.
    meta_file : str, Path, or None, optional
        Path to sample metadata CSV.  The first column must contain
        sample IDs matching those in *expr_file*; remaining columns
        are treated as covariates.
        If provided, covariate regression is performed after filtering.
        If None, covariate correction is skipped entirely.
    input_type : str, default ``"counts"``
        Declares the scale of *expr_file* values.  The caller is
        responsible for choosing the correct type; no automatic
        validation against the data is performed.

        ============  ================================================
        value         action before downstream steps
        ============  ================================================
        ``"counts"``  compute logCPM: ``log2(count/lib_size×1e6 + 1)``
        ``"cpm"``     compute log-CPM: ``log2(CPM + 1)``
        ``"logcpm"``  use as-is
        ``"tpm"``     use as-is  (no log transform)
        ``"logtpm"``  use as-is
        ============  ================================================

    min_expression_threshold : float or None, optional
        Minimum expression value for low-expression filtering.
        Applied **directly** to the post-conversion matrix (see
        *input_type*).  A gene is retained only if it exceeds this
        threshold in at least *min_expression_frac* of samples.
        If None, a type-specific empirical default is used:

        ============  =======  =====================================================
        input_type    default  meaning
        ============  =======  =====================================================
        counts        1.0      logCPM ≥ 1
        cpm           1.0      log2(CPM+1) ≥ 1
        logcpm        1.0      logCPM ≥ 1
        logtpm        1.0      log(TPM) ≥ 1
        tpm           0.5      linear TPM ≥ 0.5
        ============  =======  =====================================================

        These are empirical defaults; adjust based on your data distribution.
    min_expression_frac : float, default 0.05
        Minimum fraction of samples that must exceed *min_expression_threshold*
        for a gene to be retained.
    remove_mt_genes : bool, default True
        Remove mitochondrial genes during post-processing.
        The set of genes to remove is controlled by *mt_gene_prefix*.
    mt_gene_prefix : tuple of str, default ``("MT-", "mt-")``
        Gene name prefix(es) used to identify mitochondrial genes.
        Supports both human (``"MT-"``) and mouse (``"mt-"``) conventions.
        Only used when *remove_mt_genes* is True.

    Returns
    -------
    pd.DataFrame
        Preprocessed expression matrix (genes × samples):

        - Scaled according to *input_type* (log-scale for
          counts/cpm/logcpm/logtpm; linear for tpm).
        - Low-expression genes removed.
        - Covariate effects regressed out (only if *meta_file* provided).
        - Negative values clipped to 0.
        - Mitochondrial genes removed (if *remove_mt_genes* is True).

    Raises
    ------
    ValueError
        If *input_type* is not one of the supported values.

    Notes
    -----
    When ``input_type='tpm'``, covariate correction runs on the linear
    scale, which is statistically suboptimal (residuals are not normally
    distributed).  For more rigorous correction, consider converting TPM
    to logTPM before input.
    """
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {sorted(_VALID_INPUT_TYPES)}, "
            f"got {input_type!r}."
        )

    threshold = (
        min_expression_threshold
        if min_expression_threshold is not None
        else _DEFAULT_THRESHOLD[input_type]
    )

    # --- Load ---
    meta = load_metadata(meta_file) if meta_file is not None else None
    expr = load_expression(expr_file)

    # --- Normalize to target scale ---
    expr = normalize_to_target_scale(expr, input_type)

    # --- Filter low-expression genes ---
    expr = filter_low_expression(expr, threshold=threshold, frac=min_expression_frac)

    # --- Align samples + covariate correction ---
    if meta is not None:
        expr, meta_aligned, covariate_cols = align_samples(expr, meta)
        expr = covariate_correction(expr, meta_aligned, covariate_cols)
    else:
        print("[correct]     Skipping covariate correction (no metadata provided).")

    # --- Post-process ---
    expr = post_process(
        expr,
        remove_mt_genes=remove_mt_genes,
        mt_gene_prefix=mt_gene_prefix,
    )

    return expr


# ============================================================
# Sub-functions
# ============================================================

def load_metadata(meta_file: str | Path) -> pd.DataFrame:
    """
    Load sample metadata from a CSV file.

    The first column is renamed to ``"sample"`` and used as the
    sample identifier.  Common NA representations ("NA", "N/A",
    ".", etc.) are recognised and converted to ``NaN``.

    Parameters
    ----------
    meta_file : str or Path

    Returns
    -------
    pd.DataFrame
        Metadata with column[0] = ``"sample"``.
    """
    print(f"[metadata]    Loading {meta_file} ...")
    meta = pd.read_csv(
        meta_file, header=0,
        na_values=_NA_VALUES, keep_default_na=True,
    )
    meta.columns = ["sample"] + list(meta.columns[1:])
    meta["sample"] = meta["sample"].astype(str).str.strip()

    for col in meta.columns[1:]:
        if meta[col].dtype == object:
            orig_na = meta[col].isna()
            stripped = meta[col].astype(str).str.strip()
            stripped = stripped.where(~orig_na, other=np.nan)
            stripped = stripped.replace({"": np.nan})
            meta[col] = stripped

    n_na = meta.iloc[:, 1:].isna().sum()
    print(f"              {meta.shape[0]} samples × {meta.shape[1] - 1} covariates")
    if n_na.any():
        for col, n in n_na.items():
            if n > 0:
                print(f"              Missing in '{col}': {n} samples")
    return meta


def load_expression(expr_file: str | Path) -> pd.DataFrame:
    """
    Load a gene × sample expression matrix from a CSV file.

    Duplicate gene symbols are resolved by keeping the first occurrence.

    Parameters
    ----------
    expr_file : str or Path

    Returns
    -------
    pd.DataFrame
        Expression matrix; gene symbols as index, sample IDs as columns.
    """
    print(f"[expression]  Loading {expr_file} ...")
    expr = pd.read_csv(expr_file, header=0, index_col=0)
    expr.index   = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()
    expr = expr.dropna(axis=1, how="all")
    expr = expr.apply(pd.to_numeric, errors="coerce")

    n_dup = expr.index.duplicated().sum()
    expr  = expr[~expr.index.duplicated(keep="first")]

    msg = f"              {expr.shape[0]} genes × {expr.shape[1]} samples"
    if n_dup:
        msg += f"  ({n_dup} duplicate gene symbols dropped)"
    print(msg)
    return expr


def normalize_to_target_scale(expr: pd.DataFrame, input_type: str) -> pd.DataFrame:
    """
    Convert expression values to the scale used by downstream steps.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene × sample expression matrix.
    input_type : str
        One of ``{"counts", "cpm", "logcpm", "tpm", "logtpm"}``.

    Returns
    -------
    pd.DataFrame
        Transformed matrix.  Scale after transformation:

        - ``counts``, ``cpm``, ``logcpm`` → log2-scale (logCPM).
        - ``tpm`` → linear scale (unchanged).
        - ``logtpm`` → log-scale (unchanged).
    """
    if input_type == "counts":
        print("[normalize]   input_type='counts' → computing logCPM ...")
        lib_sizes = expr.sum(axis=0, skipna=True).clip(lower=1)
        cpm    = expr.div(lib_sizes, axis=1) * 1e6
        result = np.log2(cpm + PRIOR_COUNT)
        print(
            f"              logCPM computed: "
            f"{result.shape[0]} genes × {result.shape[1]} samples"
        )
        return result

    elif input_type == "cpm":
        print("[normalize]   input_type='cpm' → applying log2(CPM + 1) ...")
        return np.log2(expr + PRIOR_COUNT)

    else:  # logcpm | tpm | logtpm
        print(f"[normalize]   input_type='{input_type}' → no transformation applied.")
        return expr.copy()


def filter_low_expression(
    expr: pd.DataFrame,
    threshold: float,
    frac: float = _LOW_EXPR_FRAC,
) -> pd.DataFrame:
    """
    Remove genes expressed below *threshold* in too few samples.

    A gene is retained if at least ``ceil(frac × n_samples)`` samples
    have a value ≥ *threshold*.  The threshold is applied directly to
    the values in *expr* without any further scaling.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene × sample expression matrix (after scale conversion).
    threshold : float
        Minimum expression value.  Interpretation depends on the scale
        of *expr* (log2 for counts/cpm/logcpm/logtpm; linear for tpm).
    frac : float, default 0.05
        Minimum fraction of samples that must exceed *threshold*.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    min_samples = int(np.ceil(frac * expr.shape[1]))
    keep        = (expr >= threshold).sum(axis=1) >= min_samples
    filtered    = expr.loc[keep]
    print(
        f"[filter]      {expr.shape[0]} → {filtered.shape[0]} genes "
        f"(threshold={threshold}, frac={frac:.0%})"
    )
    return filtered


def align_samples(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Align expression matrix and metadata to their common samples.

    Samples present in metadata but missing covariate values are
    dropped.  The output expression matrix follows the metadata
    sample order.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene × sample expression matrix.
    meta : pd.DataFrame
        Metadata with column[0] = ``"sample"``.

    Returns
    -------
    expr_aligned : pd.DataFrame
    meta_aligned : pd.DataFrame
    covariate_cols : list of str

    Raises
    ------
    ValueError
        If expression and metadata share no common samples, or if
        the metadata has no covariate columns.
    """
    print("[align]       Aligning samples ...")
    samples_expr = set(expr.columns)
    samples_meta = list(meta["sample"])

    only_expr = [s for s in expr.columns if s not in set(samples_meta)]
    only_meta = [s for s in samples_meta if s not in samples_expr]
    if only_expr:
        print(f"              In expression but not in metadata: {only_expr}")
    if only_meta:
        print(f"              In metadata but not in expression: {only_meta}")

    common = [s for s in samples_meta if s in samples_expr]
    if not common:
        raise ValueError(
            "Expression matrix and metadata share no common sample IDs. "
            "Check that sample IDs match between files."
        )

    subset_meta = (
        meta[meta["sample"].isin(common)]
        .set_index("sample")
        .loc[common]
        .reset_index()
    )

    covariate_cols = list(subset_meta.columns[1:])
    if not covariate_cols:
        raise ValueError(
            "Metadata has no covariate columns (only the sample column)."
        )

    missing_mask = subset_meta[covariate_cols].isna().any(axis=1)
    if missing_mask.any():
        removed = subset_meta.loc[missing_mask, "sample"].tolist()
        print(f"              Dropped {len(removed)} samples with missing covariates.")
        subset_meta = subset_meta.loc[~missing_mask].reset_index(drop=True)

    common_final = subset_meta["sample"].tolist()
    expr_aligned = expr[common_final]

    print(f"              {len(common_final)} samples retained.")
    print(f"              Covariates: {covariate_cols}")
    return expr_aligned, subset_meta, covariate_cols


def covariate_correction(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    covariate_cols: list[str],
) -> pd.DataFrame:
    """
    Regress out covariate effects from expression values.

    Equivalent to R's ``lm.fit``-based correction: fits a linear model
    with intercept on the covariates, then returns residuals plus the
    intercept term (preserving the mean expression level).

    Parameters
    ----------
    expr : pd.DataFrame
        Gene × sample expression matrix, aligned to *meta*.
    meta : pd.DataFrame
        Aligned metadata with column[0] = ``"sample"``.
    covariate_cols : list of str
        Column names in *meta* to include as covariates.

    Returns
    -------
    pd.DataFrame
        Covariate-corrected expression matrix (same shape as *expr*).

    Raises
    ------
    ValueError
        If the linear fit produces NaN coefficients.
    FloatingPointError
        If the corrected matrix contains non-finite values.
    """
    print("[correct]     Applying covariate correction ...")
    design, design_matrix = _build_design_matrix(meta, covariate_cols)

    X    = expr.values.T                                            # (n_samples, n_genes)
    beta, _, _, _ = np.linalg.lstsq(design_matrix, X, rcond=None)  # (n_covariates, n_genes)

    if np.isnan(beta).any():
        raise ValueError(
            "Covariate correction produced NaN coefficients. "
            "Check for collinearity or constant covariates."
        )

    residuals = X - design_matrix @ beta
    intercept = np.tile(beta[0, :], (X.shape[0], 1))
    X_adj     = residuals + intercept

    adjusted = pd.DataFrame(X_adj.T, index=expr.index, columns=expr.columns)

    n_nan = int(np.isnan(adjusted.values).sum())
    n_inf = int(np.isinf(adjusted.values).sum())
    if n_nan or n_inf:
        raise FloatingPointError(
            f"Covariate correction produced non-finite values "
            f"(NaN={n_nan}, Inf={n_inf}). "
            "Check covariates for collinearity or outliers."
        )

    print(f"              Done: {adjusted.shape[0]} genes × {adjusted.shape[1]} samples.")
    return adjusted


def post_process(
    expr: pd.DataFrame,
    remove_mt_genes: bool = True,
    mt_gene_prefix: tuple[str, ...] = ("MT-", "mt-"),
) -> pd.DataFrame:
    """
    Final cleanup: clip negative values and optionally remove MT genes.

    Negative values can arise after covariate correction; they are
    clipped to 0 to preserve non-negativity expected by downstream steps.

    Parameters
    ----------
    expr : pd.DataFrame
    remove_mt_genes : bool, default True
        Remove mitochondrial genes.
    mt_gene_prefix : tuple of str, default ``("MT-", "mt-")``
        Prefix(es) used to identify mitochondrial genes.
        ``str.startswith`` is called with this tuple directly, so
        multiple prefixes are supported natively.

    Returns
    -------
    pd.DataFrame
    """
    print("[postprocess] Clipping negatives ...")
    expr = expr.clip(lower=0)

    if remove_mt_genes:
        mt_mask = expr.index.str.startswith(mt_gene_prefix)
        n_mt    = int(mt_mask.sum())
        expr    = expr.loc[~mt_mask]
        if n_mt:
            print(f"              Removed {n_mt} mitochondrial gene(s) "
                  f"(prefix: {mt_gene_prefix}).")

    print(f"              Final: {expr.shape[0]} genes × {expr.shape[1]} samples.")
    return expr


# ============================================================
# Internal helpers
# ============================================================

def _build_design_matrix(
    meta: pd.DataFrame,
    covariate_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a design matrix equivalent to R's ``model.matrix(~ ., data=...)``.

    - Numeric-looking object columns are coerced to float.
    - Bool columns and non-numeric object columns become categorical
      with treatment (dummy) coding (first level dropped).
    - Single-level categoricals are dropped with a note printed.
    """
    cov_df = meta[covariate_cols].copy()

    for col in cov_df.columns:
        s = cov_df[col]

        if s.dtype == bool:
            cov_df[col] = s.astype("category")
            continue

        if s.dtype == object:
            s_str = s.astype(str).str.strip().where(s.notna(), other=np.nan)
            s_num = pd.to_numeric(s_str, errors="coerce")
            if s_num.notna().sum() == s_str.notna().sum():
                cov_df[col] = s_num
            else:
                levels = sorted(s_str.dropna().unique().tolist())
                cov_df[col] = pd.Categorical(s_str, categories=levels)

    one_level = [
        c for c in cov_df.columns
        if isinstance(cov_df[c].dtype, pd.CategoricalDtype)
        and cov_df[c].nunique() < 2
    ]
    if one_level:
        print(f"              Removed single-level covariates: {one_level}")
        cov_df = cov_df.drop(columns=one_level)

    if cov_df.empty:
        raise ValueError(
            "All covariates were removed (single-level or empty). "
            "Cannot build a design matrix."
        )

    design        = pd.get_dummies(cov_df, drop_first=True).astype(float)
    design.insert(0, "Intercept", 1.0)
    design_matrix = design.values

    rank           = np.linalg.matrix_rank(design_matrix)
    n_rows, n_cols = design_matrix.shape
    print(f"              Design matrix: {n_rows} × {n_cols}, rank={rank}")
    if rank < n_cols:
        print(
            f"              WARNING: design matrix is rank-deficient "
            f"(rank={rank} < ncol={n_cols}). Check for collinearity."
        )

    return design, design_matrix
