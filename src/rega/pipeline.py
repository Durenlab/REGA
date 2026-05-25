from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

STEPS = ["peaks", "motif", "preprocess", "retg", "build-input", "train", "assemble"]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML pipeline config. Relative paths are resolved against
    the directory containing the config file."""
    base = config_path.parent
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    for section in ("data", "reference", "output", "preprocess", "retg",
                    "motif", "build_input", "train", "pipeline"):
        cfg.setdefault(section, {})

    _require(cfg, "data", "expression_csv")
    _require(cfg, "data", "peak_bed")
    _require(cfg, "reference", "genome")
    _require(cfg, "output", "base_dir")

    for key in ("expression_csv", "peak_bed", "meta_csv", "motif_tf_file"):
        _resolve_path(cfg, "data", key, base)
    for key in ("tss_file", "motif_file"):
        _resolve_path(cfg, "reference", key, base)
    _resolve_path(cfg, "output", "base_dir", base)

    return cfg


def _require(cfg: dict, section: str, key: str) -> None:
    if not cfg[section].get(key):
        raise ValueError(f"config.{section}.{key} is required")


def _resolve_path(cfg: dict, section: str, key: str, base: Path) -> None:
    val = cfg[section].get(key)
    if val is None:
        return
    p = Path(val)
    cfg[section][key] = p if p.is_absolute() else (base / p).resolve()


def _step_outputs(cfg: dict[str, Any]) -> dict[str, Path]:
    base = Path(cfg["output"]["base_dir"])
    prefix = cfg["output"].get("prefix", "sample")
    return {
        "peaks":       base / f"{prefix}_named.bed",
        "motif":       base / f"{prefix}_peak_motif.txt",
        "preprocess":  base / f"{prefix}_gex.csv",
        "retg":        base / f"{prefix}_re_tg.txt",
        "build-input": base / f"{prefix}_rega_input.h5ad",
        "train":       base / "model" / "final.pt",
        "assemble":    base / f"{prefix}_rega_result.h5ad",
    }


def _output_exists(step: str, path: Path) -> bool:
    return path.exists()


def _build_skip_set(
    resume_from: str | None,
    skip_motif: bool,
    skip_train: bool,
    step_outputs: dict[str, Path],
) -> set[str]:
    skip: set[str] = set()

    if resume_from is not None:
        if resume_from not in STEPS:
            raise ValueError(f"--resume-from must be one of {STEPS}, got {resume_from!r}")
        idx = STEPS.index(resume_from)
        for step in STEPS[:idx]:
            out = step_outputs[step]
            if not _output_exists(step, out):
                raise ValueError(
                    f"Cannot resume from '{resume_from}': expected output for step "
                    f"'{step}' not found at {out}"
                )
            skip.add(step)

    if skip_motif:
        skip.add("motif")
    if skip_train:
        skip.add("train")

    return skip


def run_pipeline(
    config_path: Path,
    output_dir: Path | None = None,
    skip_motif: bool = False,
    skip_train: bool = False,
    resume_from: str | None = None,
    keep_intermediate: bool = False,
) -> dict[str, Path]:
    """Run the full REGA pipeline from a YAML config file.

    Returns a dict mapping step name to its output path.
    """
    cfg = load_config(config_path)
    if output_dir is not None:
        cfg["output"]["base_dir"] = output_dir

    base = Path(cfg["output"]["base_dir"])
    prefix = cfg["output"].get("prefix", "sample")
    base.mkdir(parents=True, exist_ok=True)

    step_out = _step_outputs(cfg)
    skip = _build_skip_set(resume_from, skip_motif, skip_train, step_out)

    tmpdir: Path | None = base / "intermediate" if keep_intermediate else None

    for step in STEPS:
        if step in skip:
            logger.info("[pipeline] skipping %s", step)
            continue
        logger.info("[pipeline] running step: %s", step)
        _run_step(step, cfg, step_out, prefix, base, tmpdir, keep_intermediate)
        logger.info("[pipeline] completed: %s → %s", step, step_out[step])

    return step_out


# ── Per-step runners ───────────────────────────────────────────────────────────

def _run_step(
    step: str,
    cfg: dict[str, Any],
    step_out: dict[str, Path],
    prefix: str,
    base: Path,
    tmpdir: Path | None,
    keep_intermediate: bool,
) -> None:
    if step == "peaks":
        _step_peaks(cfg, step_out)
    elif step == "motif":
        _step_motif(cfg, step_out, prefix, keep_intermediate)
    elif step == "preprocess":
        _step_preprocess(cfg, step_out)
    elif step == "retg":
        _step_retg(cfg, step_out, keep_intermediate, tmpdir)
    elif step == "build-input":
        _step_build_input(cfg, step_out, keep_intermediate)
    elif step == "train":
        _step_train(cfg, step_out)
    elif step == "assemble":
        _step_assemble(step_out)


def _step_peaks(cfg: dict, step_out: dict[str, Path]) -> None:
    from rega.peaks import rename_peaks
    step_out["peaks"].parent.mkdir(parents=True, exist_ok=True)
    rename_peaks(
        input_bed=cfg["data"]["peak_bed"],
        output_path=step_out["peaks"],
        sep=cfg.get("peaks", {}).get("sep", "\t"),
    )


def _step_motif(cfg: dict, step_out: dict[str, Path], prefix: str, keep_intermediate: bool) -> None:
    from rega.motif import run_motif_scan
    m = cfg.get("motif", {})
    motif_db = cfg["reference"].get("motif_file")
    if motif_db is None:
        raise ValueError("config.reference.motif_file is required for the motif step")
    out_dir = step_out["motif"].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_motif_scan(
        named_bed=step_out["peaks"],
        genome=cfg["reference"]["genome"],
        motif_file=Path(motif_db),
        output_dir=out_dir,
        output_prefix=prefix,
        homer_bin=m.get("homer_bin", "findMotifsGenome.pl"),
        threads=m.get("threads", 1),
        keep_intermediate=keep_intermediate,
    )


def _step_preprocess(cfg: dict, step_out: dict[str, Path]) -> None:
    from rega.preprocessing import preprocess_expression
    d = cfg["data"]
    p = cfg.get("preprocess", {})
    mt_prefix = p.get("mt_gene_prefix", ["MT-", "mt-"])
    df = preprocess_expression(
        expr_file=d["expression_csv"],
        meta_file=d.get("meta_csv"),
        input_type=d.get("input_type", "counts"),
        min_expression_threshold=p.get("min_expression_threshold"),
        min_expression_frac=p.get("min_expression_frac", 0.05),
        remove_mt_genes=p.get("remove_mt_genes", True),
        mt_gene_prefix=tuple(mt_prefix),
    )
    step_out["preprocess"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(step_out["preprocess"])


def _step_retg(
    cfg: dict,
    step_out: dict[str, Path],
    keep_intermediate: bool,
    tmpdir: Path | None,
) -> None:
    from rega.retg import build_re_tg
    r = cfg.get("retg", {})
    tss = cfg["reference"].get("tss_file")
    if tss is None:
        from rega.reference import ReferenceData
        tss = ReferenceData(cfg["reference"]["genome"]).tss_file
    step_out["retg"].parent.mkdir(parents=True, exist_ok=True)
    build_re_tg(
        tss_file=Path(tss),
        peaks_bed=step_out["peaks"],
        output_tsv=step_out["retg"],
        window=r.get("window", 300_000),
        chroms=r.get("chroms"),
        keep_intermediate=keep_intermediate,
        intermediate_dir=tmpdir,
    )


def _step_build_input(cfg: dict, step_out: dict[str, Path], keep_intermediate: bool) -> None:
    from rega.matrices import build_rega_input
    from rega.anndata_builder import build_rega_anndata
    b = cfg.get("build_input", {})
    motif_tf = cfg["data"].get("motif_tf_file")

    if keep_intermediate:
        inter_dir = step_out["build-input"].parent / "build_input_intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        inter_dir = Path(tempfile.mkdtemp())
        cleanup = True

    try:
        mat = build_rega_input(
            gex_csv=step_out["preprocess"],
            re_tg_file=step_out["retg"],
            peak_motif_file=step_out["motif"],
            output_dir=inter_dir,
            tau=b.get("tau", 1e5),
        )
        step_out["build-input"].parent.mkdir(parents=True, exist_ok=True)
        build_rega_anndata(
            re_tg_h5=mat["re_tg_h5"],
            gex_csv=mat["gex_csv"],
            binding_npz=mat["binding_npz"],
            binding_peaks_txt=mat["binding_peaks_txt"],
            binding_motifs_txt=mat["binding_motifs_txt"],
            annotation_file=step_out["retg"],
            output_h5ad=step_out["build-input"],
            motif_tf_file=Path(motif_tf) if motif_tf else None,
        )
    finally:
        if cleanup:
            shutil.rmtree(inter_dir, ignore_errors=True)


def _step_train(cfg: dict, step_out: dict[str, Path]) -> None:
    from rega.blocks import load_and_check, split_and_save
    from rega.model import load_blocks, train_rega, TrainingConfig
    import torch
    t = cfg.get("train", {})
    r = cfg.get("retg", {})

    dtype_map: dict[str, Any] = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(str(t.get("dtype", "float32")), torch.float32)

    config = TrainingConfig(
        K=t.get("K", 20),
        eps=t.get("eps", 1e-10),
        dtype=dtype,
        lam_W=t.get("lam_W", 0.0),
        lam_B=t.get("lam_B", 0.0),
        lam_V=t.get("lam_V", 0.0),
        lam_H=t.get("lam_H", 0.0),
        clamp_min=t.get("clamp_min", 0.0),
        phase1_inner_iters=t.get("phase1_inner_iters", 2000),
        phase1_outer_iters=t.get("phase1_outer_iters", 30),
        phase2_inner_iters=t.get("phase2_inner_iters", 500),
        phase2_min_outer=t.get("phase2_min_outer", 3000),
        phase2_max_outer=t.get("phase2_max_outer", 6000),
        early_stop_patience=t.get("early_stop_patience", 50),
        early_stop_rel_tol=t.get("early_stop_rel_tol", 1e-5),
        log_every_inner=t.get("log_every_inner", 500),
        V_seed=t.get("V_seed", 123),
        H_seed=t.get("H_seed", 1234),
    )

    model_dir = step_out["train"].parent
    model_dir.mkdir(parents=True, exist_ok=True)

    keep_blocks = t.get("keep_blocks", False)
    if keep_blocks:
        blocks_dir = model_dir / "blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        blocks_dir = Path(tempfile.mkdtemp())
        cleanup = True

    try:
        adata = load_and_check(step_out["build-input"])
        split_and_save(adata, output_dir=blocks_dir, chroms=r.get("chroms"))
        block_data = load_blocks(blocks_dir)
        device = t.get("device") or None
        train_rega(block_data, output_dir=model_dir, config=config, device=device)
    finally:
        if cleanup:
            shutil.rmtree(blocks_dir, ignore_errors=True)


def _step_assemble(step_out: dict[str, Path]) -> None:
    from rega.results import collect_results
    step_out["assemble"].parent.mkdir(parents=True, exist_ok=True)
    collect_results(
        input_h5ad=step_out["build-input"],
        rega_result_path=step_out["train"],
        out_h5ad=step_out["assemble"],
    )
