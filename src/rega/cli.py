from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import click

logger = logging.getLogger(__name__)

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
_ANALYSIS_ENTITY_TYPES = ["gene", "peak", "motif", "sample", "all"]
_ASSIGNMENT_TYPES = ["hard", "soft", "both"]
_STEPS = ["peaks", "motif", "preprocess", "retg", "build-input", "train", "assemble"]


# ── Error handling ─────────────────────────────────────────────────────────────

@contextmanager
def _handle_errors(debug: bool):
    try:
        yield
    except FileNotFoundError as exc:
        click.echo(f"Error: file not found — {exc.filename}", err=True)
        sys.exit(2)
    except Exception as exc:
        if debug:
            raise
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ── Root group ─────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="rega")
@click.option(
    "--log-level",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Root logger verbosity.",
)
@click.option("--debug/--no-debug", default=False, help="Show full traceback on error.")
@click.pass_context
def main(ctx: click.Context, log_level: str, debug: bool) -> None:
    """REGA: regulatory element-guided gene expression analysis."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    _setup_logging(log_level)


# ── Step 1: peaks ──────────────────────────────────────────────────────────────

@main.command()
@click.argument("input_bed", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--sep", default="\t", show_default=True, help="Input field separator.")
@click.pass_context
def peaks(ctx: click.Context, input_bed: Path, output_path: Path, sep: str) -> None:
    """Step 1: rename peaks — append chr_start_end name column to BED."""
    from rega.peaks import rename_peaks
    with _handle_errors(ctx.obj["debug"]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = rename_peaks(input_bed, output_path, sep=sep)
        click.echo(f"[peaks] wrote {out}")


# ── Step 2: motif ──────────────────────────────────────────────────────────────

@main.command()
@click.argument("named_bed", type=click.Path(exists=True, path_type=Path))
@click.option("--genome", required=True, help="Reference genome (e.g. hg38).")
@click.option(
    "--motif-db",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="HOMER motif database file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option("--prefix", required=True, help="Output file prefix (e.g. naiveCD4T).")
@click.option(
    "--homer-bin",
    default="findMotifsGenome.pl",
    show_default=True,
    help="HOMER binary name or full path.",
)
@click.option("--threads", default=1, show_default=True, type=int)
@click.option("--keep-intermediate", is_flag=True, default=False)
@click.pass_context
def motif(
    ctx: click.Context,
    named_bed: Path,
    genome: str,
    motif_db: Path,
    output_dir: Path,
    prefix: str,
    homer_bin: str,
    threads: int,
    keep_intermediate: bool,
) -> None:
    """Step 2: run HOMER motif scan → {prefix}_peak_motif.txt."""
    from rega.motif import run_motif_scan
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)
        out = run_motif_scan(
            named_bed=named_bed,
            genome=genome,
            motif_file=motif_db,
            output_dir=output_dir,
            output_prefix=prefix,
            homer_bin=homer_bin,
            threads=threads,
            keep_intermediate=keep_intermediate,
        )
        click.echo(f"[motif] wrote {out}")


# ── Step 3: preprocess ─────────────────────────────────────────────────────────

@main.command()
@click.argument("expr_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output CSV path.")
@click.option(
    "--meta",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Sample metadata CSV (enables covariate correction).",
)
@click.option(
    "--input-type",
    type=click.Choice(["counts", "cpm", "logcpm", "tpm", "logtpm"], case_sensitive=False),
    default="counts",
    show_default=True,
    help="Scale of the input expression matrix.",
)
@click.option(
    "--min-expr-threshold",
    type=float,
    default=None,
    help="Low-expression filter threshold (default depends on --input-type).",
)
@click.option(
    "--min-expr-frac",
    type=float,
    default=0.05,
    show_default=True,
    help="Min fraction of samples expressing a gene.",
)
@click.option("--no-remove-mt", is_flag=True, default=False, help="Keep mitochondrial genes.")
@click.option(
    "--mt-prefix",
    multiple=True,
    default=("MT-", "mt-"),
    show_default=True,
    help="MT gene name prefix(es) (repeatable).",
)
@click.pass_context
def preprocess(
    ctx: click.Context,
    expr_file: Path,
    output: Path,
    meta: Path | None,
    input_type: str,
    min_expr_threshold: float | None,
    min_expr_frac: float,
    no_remove_mt: bool,
    mt_prefix: tuple[str, ...],
) -> None:
    """Step 3: preprocess gene expression (normalize, filter, covariate correction)."""
    from rega.preprocessing import preprocess_expression
    with _handle_errors(ctx.obj["debug"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        df = preprocess_expression(
            expr_file=expr_file,
            meta_file=meta,
            input_type=input_type,
            min_expression_threshold=min_expr_threshold,
            min_expression_frac=min_expr_frac,
            remove_mt_genes=not no_remove_mt,
            mt_gene_prefix=tuple(mt_prefix),
        )
        df.to_csv(output)
        click.echo(f"[preprocess] wrote {output} ({df.shape[0]} genes × {df.shape[1]} samples)")


# ── Step 4: retg ───────────────────────────────────────────────────────────────

@main.command()
@click.argument("peaks_bed", type=click.Path(exists=True, path_type=Path))
@click.option("--tss", required=True, type=click.Path(exists=True, path_type=Path), help="TSS reference file (4-col TSV).")
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output RE-TG TSV.")
@click.option(
    "--window",
    default=300_000,
    show_default=True,
    type=int,
    help="TSS window size in bp (each side).",
)
@click.option("--chrom", multiple=True, default=None, help="Restrict to chromosome(s) (repeatable).")
@click.option("--keep-intermediate", is_flag=True, default=False)
@click.pass_context
def retg(
    ctx: click.Context,
    peaks_bed: Path,
    tss: Path,
    output: Path,
    window: int,
    chrom: tuple[str, ...],
    keep_intermediate: bool,
) -> None:
    """Step 4: build RE-TG pair table via TSS-window + bedtools intersect."""
    from rega.retg import build_re_tg
    with _handle_errors(ctx.obj["debug"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        build_re_tg(
            tss_file=tss,
            peaks_bed=peaks_bed,
            output_tsv=output,
            window=window,
            chroms=list(chrom) if chrom else None,
            keep_intermediate=keep_intermediate,
        )
        click.echo(f"[retg] wrote {output}")


# ── Steps 5+6: build-input ────────────────────────────────────────────────────

@main.command("build-input")
@click.option("--gex", required=True, type=click.Path(exists=True, path_type=Path), help="Preprocessed GEX CSV.")
@click.option("--re-tg", required=True, type=click.Path(exists=True, path_type=Path), help="RE-TG TSV (from retg).")
@click.option("--peak-motif", required=True, type=click.Path(exists=True, path_type=Path), help="Peak-motif TSV (from motif).")
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output rega_input.h5ad.")
@click.option("--tau", default=1e5, show_default=True, type=float, help="Distance-decay scale.")
@click.option(
    "--motif-tf",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Motif-TF mapping TSV (optional).",
)
@click.pass_context
def build_input(
    ctx: click.Context,
    gex: Path,
    re_tg: Path,
    peak_motif: Path,
    output: Path,
    tau: float,
    motif_tf: Path | None,
) -> None:
    """Steps 5+6: build REGA input matrices → assemble rega_input.h5ad."""
    from rega.matrices import build_rega_input
    from rega.anndata_builder import build_rega_anndata
    with _handle_errors(ctx.obj["debug"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp())
        try:
            mat = build_rega_input(
                gex_csv=gex,
                re_tg_file=re_tg,
                peak_motif_file=peak_motif,
                output_dir=tmpdir,
                tau=tau,
            )
            build_rega_anndata(
                re_tg_h5=mat["re_tg_h5"],
                gex_csv=mat["gex_csv"],
                binding_npz=mat["binding_npz"],
                binding_peaks_txt=mat["binding_peaks_txt"],
                binding_motifs_txt=mat["binding_motifs_txt"],
                annotation_file=re_tg,
                output_h5ad=output,
                motif_tf_file=motif_tf,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        click.echo(f"[build-input] wrote {output}")


# ── Steps 7+8: train ──────────────────────────────────────────────────────────

@main.command()
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for final.pt and loss log.",
)
@click.option(
    "--keep-blocks",
    is_flag=True,
    default=False,
    help="Retain per-chromosome .npz blocks in output-dir/blocks/ after training.",
)
@click.option("--K", default=20, show_default=True, type=int, help="Number of regulatory modules.")
@click.option("--device", default=None, help="cuda|cpu (default: auto-detect).")
@click.option("--phase1-inner", default=2000, show_default=True, type=int)
@click.option("--phase1-outer", default=30, show_default=True, type=int)
@click.option("--phase2-inner", default=500, show_default=True, type=int)
@click.option("--phase2-min-outer", default=3000, show_default=True, type=int)
@click.option("--phase2-max-outer", default=6000, show_default=True, type=int)
@click.option("--patience", default=50, show_default=True, type=int, help="Early-stop patience (outer iters).")
@click.option("--rel-tol", default=1e-5, show_default=True, type=float, help="Relative loss improvement threshold.")
@click.option("--lam-w", default=0.0, show_default=True, type=float, help="L1 penalty on W.")
@click.option("--lam-b", default=0.0, show_default=True, type=float, help="L1 penalty on B.")
@click.option("--lam-v", default=0.0, show_default=True, type=float, help="L1 penalty on V.")
@click.option("--lam-h", default=0.0, show_default=True, type=float, help="L1 penalty on H.")
@click.pass_context
def train(
    ctx: click.Context,
    input_h5ad: Path,
    output_dir: Path,
    keep_blocks: bool,
    k: int,
    device: str | None,
    phase1_inner: int,
    phase1_outer: int,
    phase2_inner: int,
    phase2_min_outer: int,
    phase2_max_outer: int,
    patience: int,
    rel_tol: float,
    lam_w: float,
    lam_b: float,
    lam_v: float,
    lam_h: float,
) -> None:
    """Train REGA model from prepared input AnnData.

    INPUT_H5AD: rega_input.h5ad (from `rega build-input`)

    The training automatically prepares per-chromosome blocks internally.
    Use --keep-blocks to retain them for inspection.
    """
    from rega.blocks import load_and_check, split_and_save
    from rega.model import load_blocks, train_rega, TrainingConfig
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)

        if keep_blocks:
            blocks_dir = output_dir / "blocks"
            blocks_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        else:
            blocks_dir = Path(tempfile.mkdtemp())
            cleanup = True

        try:
            adata = load_and_check(input_h5ad)
            split_and_save(adata, output_dir=blocks_dir)
            block_data = load_blocks(blocks_dir)
            config = TrainingConfig(
                K=k,
                lam_W=lam_w,
                lam_B=lam_b,
                lam_V=lam_v,
                lam_H=lam_h,
                phase1_inner_iters=phase1_inner,
                phase1_outer_iters=phase1_outer,
                phase2_inner_iters=phase2_inner,
                phase2_min_outer=phase2_min_outer,
                phase2_max_outer=phase2_max_outer,
                early_stop_patience=patience,
                early_stop_rel_tol=rel_tol,
            )
            out = train_rega(block_data, output_dir=output_dir, config=config, device=device)
        finally:
            if cleanup:
                shutil.rmtree(blocks_dir, ignore_errors=True)

        click.echo(f"[train] wrote {out}")


# ── Step 9: assemble ───────────────────────────────────────────────────────────

@main.command()
@click.option("--input-h5ad", required=True, type=click.Path(exists=True, path_type=Path), help="rega_input.h5ad from build-input.")
@click.option("--result", required=True, type=click.Path(exists=True, path_type=Path), help="final.pt from train.")
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output rega_result.h5ad.")
@click.pass_context
def assemble(ctx: click.Context, input_h5ad: Path, result: Path, output: Path) -> None:
    """Step 9: collect REGA training result → rega_result.h5ad."""
    from rega.results import collect_results
    with _handle_errors(ctx.obj["debug"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        collect_results(input_h5ad=input_h5ad, rega_result_path=result, out_h5ad=output)
        click.echo(f"[assemble] wrote {output}")


# ── analysis subgroup ──────────────────────────────────────────────────────────

@main.group()
def analysis() -> None:
    """Downstream analysis on rega_result.h5ad."""


@analysis.command("modules")
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option(
    "--entity-type",
    type=click.Choice(_ANALYSIS_ENTITY_TYPES, case_sensitive=False),
    default="all",
    show_default=True,
    help="Entity type to assign (or 'all' for all four).",
)
@click.option(
    "--assignment-type",
    type=click.Choice(_ASSIGNMENT_TYPES, case_sensitive=False),
    default="hard",
    show_default=True,
    help="'hard' = unique module per entity, 'soft' = multi-module allowed, 'both' = run both.",
)
@click.option("--topk-frac", default=0.1, show_default=True, type=float, help="Top-K fraction per module.")
@click.option("--p-abs", default=0.25, show_default=True, type=float, help="Absolute score threshold for label assignment.")
@click.pass_context
def analysis_modules(
    ctx: click.Context,
    input_h5ad: Path,
    output_dir: Path,
    entity_type: str,
    assignment_type: str,
    topk_frac: float,
    p_abs: float,
) -> None:
    """Assign genes/peaks/motifs/samples to regulatory modules."""
    import anndata as ad
    from rega.analysis import assign_modules, load_entity_matrices, write_module_assignments
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(input_h5ad)
        matrices = load_entity_matrices(adata)
        etypes = list(matrices.keys()) if entity_type == "all" else [entity_type]
        atypes = ["hard", "soft"] if assignment_type == "both" else [assignment_type]
        for et in etypes:
            for at in atypes:
                result = assign_modules(matrices[et], topk_frac=topk_frac, p_abs=p_abs, assignment_type=at)
                sub = output_dir / et / at
                sub.mkdir(parents=True, exist_ok=True)
                write_module_assignments(result, prefix=et, out_dir=sub)
                click.echo(f"[modules] {et}/{at} → {sub}")


@analysis.command("grn")
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--min-re-tg", default=0.0, show_default=True, type=float, help="Min score threshold for RE-TG edges.")
@click.option("--min-tf-tg", default=0.0, show_default=True, type=float, help="Min score threshold for TF-TG edges.")
@click.pass_context
def analysis_grn(
    ctx: click.Context,
    input_h5ad: Path,
    output_dir: Path,
    min_re_tg: float,
    min_tf_tg: float,
) -> None:
    """Extract RE-TG and TF-TG GRN tables."""
    import anndata as ad
    from rega.analysis import extract_grn
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(input_h5ad)
        extract_grn(adata, out_dir=output_dir, min_re_tg_score=min_re_tg, min_tf_tg_score=min_tf_tg)


@analysis.command("re-score")
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output TSV path.")
@click.pass_context
def analysis_re_score(ctx: click.Context, input_h5ad: Path, output: Path) -> None:
    """Extract RE importance scores to a TSV file."""
    import anndata as ad
    from rega.analysis import write_re_importance
    with _handle_errors(ctx.obj["debug"]):
        output.parent.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(input_h5ad)
        write_re_importance(adata, out_tsv=output)


@analysis.command("disease")
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--label",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV with columns: sample_id, disease_label (0/1).",
)
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--fdr-cutoff", default=0.05, show_default=True, type=float)
@click.pass_context
def analysis_disease(
    ctx: click.Context,
    input_h5ad: Path,
    label: Path,
    output_dir: Path,
    fdr_cutoff: float,
) -> None:
    """Identify disease-associated regulatory modules."""
    import anndata as ad
    import pandas as pd
    from rega.analysis import identify_disease_modules, write_disease_module_results
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(input_h5ad)
        label_df = pd.read_csv(label)
        result = identify_disease_modules(adata, label_df, fdr_cutoff=fdr_cutoff)
        write_disease_module_results(result, out_dir=output_dir)
        n_sig = int(result["disease_module_significant"].sum())
        click.echo(f"[disease] {n_sig} significant module(s) at FDR < {fdr_cutoff}")


@analysis.command("driver-tf")
@click.argument("input_h5ad", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--label",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV with columns: sample_id, disease_label (0/1).",
)
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--pcc-cutoff", default=0.1, show_default=True, type=float, help="Motif_TF_PCC threshold.")
@click.option("--fdr-cutoff", default=0.05, show_default=True, type=float)
@click.pass_context
def analysis_driver_tf(
    ctx: click.Context,
    input_h5ad: Path,
    label: Path,
    output_dir: Path,
    pcc_cutoff: float,
    fdr_cutoff: float,
) -> None:
    """Identify disease-associated driver TFs."""
    import anndata as ad
    import pandas as pd
    from rega.analysis import identify_driver_tfs, write_driver_tf_results
    with _handle_errors(ctx.obj["debug"]):
        output_dir.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(input_h5ad)
        label_df = pd.read_csv(label)
        result = identify_driver_tfs(adata, label_df, pcc_cutoff=pcc_cutoff, fdr_cutoff=fdr_cutoff)
        write_driver_tf_results(result, out_dir=output_dir)
        n_sig = int(result["driver_TF_significant"].sum())
        click.echo(f"[driver-tf] {n_sig} significant driver TF(s) at FDR < {fdr_cutoff}")


# ── Full pipeline ──────────────────────────────────────────────────────────────

@main.command("pipeline")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML pipeline config file.",
)
@click.option("--skip-motif", is_flag=True, default=False, help="Use existing peak_motif.txt (skip HOMER).")
@click.option("--skip-train", is_flag=True, default=False, help="Use existing final.pt (skip training).")
@click.option(
    "--resume-from",
    default=None,
    type=click.Choice(_STEPS),
    help="Skip all steps before this one (requires prior outputs to exist).",
)
@click.option("--keep-intermediate", is_flag=True, default=False, help="Keep intermediate build-input files.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config output.base_dir.",
)
@click.pass_context
def pipeline(
    ctx: click.Context,
    config: Path,
    skip_motif: bool,
    skip_train: bool,
    resume_from: str | None,
    keep_intermediate: bool,
    output_dir: Path | None,
) -> None:
    """Run the full REGA pipeline from a YAML config file."""
    from rega.pipeline import run_pipeline
    with _handle_errors(ctx.obj["debug"]):
        run_pipeline(
            config_path=config,
            output_dir=output_dir,
            skip_motif=skip_motif,
            skip_train=skip_train,
            resume_from=resume_from,
            keep_intermediate=keep_intermediate,
        )
        click.echo("[pipeline] complete")
