from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rega.cli import main

RUNNER = CliRunner()


# ── Help & version ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args,keywords", [
    (
        ["--help"],
        ["peaks", "motif", "preprocess", "retg", "build-input",
         "train", "assemble", "analysis", "pipeline"],
    ),
    (
        ["analysis", "--help"],
        ["modules", "grn", "re-score", "disease", "driver-tf"],
    ),
])
def test_help_shows_subcommands(args, keywords):
    result = RUNNER.invoke(main, args)
    assert result.exit_code == 0, result.output
    for kw in keywords:
        assert kw in result.output


@pytest.mark.parametrize("cmd", [
    "peaks", "preprocess", "retg", "motif", "build-input",
    "train", "assemble",
])
def test_core_command_help(cmd):
    result = RUNNER.invoke(main, [cmd, "--help"])
    assert result.exit_code == 0, result.output
    assert "--help" in result.output


@pytest.mark.parametrize("subcmd", ["modules", "grn", "re-score", "disease", "driver-tf"])
def test_analysis_command_help(subcmd):
    result = RUNNER.invoke(main, ["analysis", subcmd, "--help"])
    assert result.exit_code == 0, result.output
    assert "--help" in result.output


def test_version():
    result = RUNNER.invoke(main, ["--version"])
    assert result.exit_code == 0
    # Should contain a digit (version number)
    assert any(c.isdigit() for c in result.output)


# ── Error handling ─────────────────────────────────────────────────────────────

def test_missing_input_file_exits_nonzero(tmp_path):
    """click's exists=True on INPUT_BED → non-zero exit when file absent."""
    result = RUNNER.invoke(main, ["peaks", str(tmp_path / "nope.bed"), str(tmp_path / "out.bed")])
    assert result.exit_code != 0


def test_generic_error_exits_1(tmp_path):
    bed = tmp_path / "input.bed"
    bed.write_text("chr1\t100\t200\n")
    out = tmp_path / "out.bed"
    with patch("rega.peaks.rename_peaks", side_effect=RuntimeError("boom")):
        result = RUNNER.invoke(main, ["peaks", str(bed), str(out)])
    assert result.exit_code == 1
    assert "Error" in result.stderr


def test_debug_flag_reraises(tmp_path):
    """--debug should let the original exception propagate (stored in result.exception)."""
    bed = tmp_path / "input.bed"
    bed.write_text("chr1\t100\t200\n")
    out = tmp_path / "out.bed"
    with patch("rega.peaks.rename_peaks", side_effect=RuntimeError("debug-boom")):
        result = RUNNER.invoke(main, ["--debug", "peaks", str(bed), str(out)])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


# ── peaks command ──────────────────────────────────────────────────────────────

def test_peaks_command_calls_rename_peaks(tmp_path):
    bed = tmp_path / "raw.bed"
    bed.write_text("chr1\t100\t200\n")
    out = tmp_path / "named.bed"
    with patch("rega.peaks.rename_peaks", return_value=out) as mock_fn:
        result = RUNNER.invoke(main, ["peaks", str(bed), str(out)])
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


# ── preprocess command ────────────────────────────────────────────────────────

def test_preprocess_command(tmp_path):
    import pandas as pd
    expr = tmp_path / "expr.csv"
    expr.write_text("gene,s1,s2\nGENE1,10,20\nGENE2,5,8\n")
    out = tmp_path / "gex.csv"
    fake_df = pd.DataFrame({"s1": [1.0], "s2": [2.0]}, index=["GENE1"])
    with patch("rega.preprocessing.preprocess_expression", return_value=fake_df):
        result = RUNNER.invoke(main, [
            "preprocess", str(expr),
            "--output", str(out),
            "--input-type", "counts",
        ])
    assert result.exit_code == 0, result.output
    assert "preprocess" in result.output


# ── pipeline command ───────────────────────────────────────────────────────────

def test_pipeline_missing_config():
    result = RUNNER.invoke(main, ["pipeline", "--config", "/no/such/config.yaml"])
    assert result.exit_code != 0


def test_pipeline_invalid_resume_choice():
    """--resume-from with a value outside the allowed set → click UsageError."""
    result = RUNNER.invoke(main, ["pipeline", "--config", "/dev/null", "--resume-from", "badstep"])
    assert result.exit_code != 0


def test_pipeline_valid_config_parsed(tmp_path):
    """A syntactically valid config should at least pass YAML loading + skip-flag logic."""
    gex = tmp_path / "expr.csv"; gex.touch()
    bed = tmp_path / "peaks.bed"; bed.touch()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data:\n"
        f"  expression_csv: {gex}\n"
        f"  peak_bed: {bed}\n"
        f"reference:\n"
        f"  genome: hg38\n"
        f"output:\n"
        f"  base_dir: {tmp_path / 'out'}\n"
        f"  prefix: test\n"
    )
    # Patch _run_step so no real computation happens; verify exit 0
    with patch("rega.pipeline._run_step"):
        result = RUNNER.invoke(main, ["pipeline", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "complete" in result.output


# ── analysis modules command ──────────────────────────────────────────────────

def test_analysis_modules_command(tmp_path):
    h5ad = tmp_path / "result.h5ad"
    h5ad.touch()
    out = tmp_path / "modules"
    mock_adata = MagicMock()
    mock_matrices = {
        "gene":   MagicMock(),
        "peak":   MagicMock(),
        "motif":  MagicMock(),
        "sample": MagicMock(),
    }
    mock_result = {
        "assignment_table": MagicMock(),
        "module_counts":    MagicMock(),
    }
    with patch("anndata.read_h5ad", return_value=mock_adata), \
         patch("rega.analysis.load_entity_matrices", return_value=mock_matrices), \
         patch("rega.analysis.assign_modules", return_value=mock_result), \
         patch("rega.analysis.write_module_assignments"):
        result = RUNNER.invoke(main, [
            "analysis", "modules", str(h5ad),
            "--output-dir", str(out),
            "--entity-type", "all",
        ])
    assert result.exit_code == 0, result.output


# ── analysis grn command ──────────────────────────────────────────────────────

def test_analysis_grn_command(tmp_path):
    h5ad = tmp_path / "result.h5ad"
    h5ad.touch()
    out = tmp_path / "grn"
    mock_adata = MagicMock()
    with patch("anndata.read_h5ad", return_value=mock_adata), \
         patch("rega.analysis.extract_grn"):
        result = RUNNER.invoke(main, [
            "analysis", "grn", str(h5ad),
            "--output-dir", str(out),
        ])
    assert result.exit_code == 0, result.output


# ── train command ─────────────────────────────────────────────────────────────

def _mock_train_stack(tmp_path):
    """Shared mock set for train command tests. Returns (h5ad path, output dir)."""
    h5ad = tmp_path / "rega_input.h5ad"
    h5ad.touch()
    out = tmp_path / "model"
    return h5ad, out


def test_train_command_default(tmp_path):
    """Train with default flags: blocks go to a temp dir that is cleaned up."""
    h5ad, out = _mock_train_stack(tmp_path)
    mock_adata = MagicMock()
    mock_block_data = MagicMock()
    with patch("rega.blocks.load_and_check", return_value=mock_adata), \
         patch("rega.blocks.split_and_save", return_value=3), \
         patch("rega.model.load_blocks", return_value=mock_block_data), \
         patch("rega.model.train_rega", return_value=out / "final.pt"):
        result = RUNNER.invoke(main, [
            "train", str(h5ad),
            "--output-dir", str(out),
        ])
    assert result.exit_code == 0, result.output
    assert "[train]" in result.output
    # blocks dir inside output-dir should NOT exist (cleanup happened)
    assert not (out / "blocks").exists()


def test_train_keep_blocks(tmp_path):
    """--keep-blocks should create output-dir/blocks/ and leave it in place."""
    h5ad, out = _mock_train_stack(tmp_path)
    mock_adata = MagicMock()
    mock_block_data = MagicMock()

    captured_blocks_dir: list[Path] = []

    def fake_split(adata, output_dir, chroms=None):
        captured_blocks_dir.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write a dummy .npz so the directory looks non-empty
        (output_dir / "chr1.npz").touch()
        return 1

    with patch("rega.blocks.load_and_check", return_value=mock_adata), \
         patch("rega.blocks.split_and_save", side_effect=fake_split), \
         patch("rega.model.load_blocks", return_value=mock_block_data), \
         patch("rega.model.train_rega", return_value=out / "final.pt"):
        result = RUNNER.invoke(main, [
            "train", str(h5ad),
            "--output-dir", str(out),
            "--keep-blocks",
        ])
    assert result.exit_code == 0, result.output
    # blocks dir is inside output-dir and was not removed
    assert len(captured_blocks_dir) == 1
    expected_blocks = out / "blocks"
    assert captured_blocks_dir[0] == expected_blocks
    assert expected_blocks.exists()


def test_train_help_mentions_build_input(tmp_path):
    """train --help should mention build-input so users know what INPUT_H5AD is."""
    result = RUNNER.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "build-input" in result.output
