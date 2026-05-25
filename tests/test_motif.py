from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rega.motif import process_output, run_homer, run_motif_scan

# Minimal HOMER -find raw output (PositionID/Offset/Sequence/MotifName/Strand/Score)
_HOMER_HEADER = "PositionID\tOffset\tSequence\tMotifName\tStrand\tMotifScore\n"
_HOMER_BODY = (
    "chr1_1100_1200\t5\tACGT\tAP1/encode\t+\t8.5\n"
    "chr1_1100_1200\t10\tTTGA\tCTCF/encode\t-\t7.2\n"
    "chr1_3000_3200\t5\tGCTA\tAP1/encode\t+\t9.1\n"
    "chr1_1100_1200\t5\tACGT\tAP1/encode\t+\t8.5\n"  # exact duplicate
)
_HOMER_RAW = _HOMER_HEADER + _HOMER_BODY


def _fake_homer(*args, **kwargs):
    """Mock side_effect: write HOMER-format content to stdout file, return success."""
    stdout_handle = kwargs.get("stdout")
    if stdout_handle is not None:
        stdout_handle.write(_HOMER_RAW)
    return MagicMock(returncode=0, stderr="")


# ── process_output ────────────────────────────────────────────────────────────

def test_process_output_extracts_peak_motif(tmp_path):
    """Extracts col[0] (peak_name) and col[3] (motif_name) from raw HOMER output."""
    raw = tmp_path / "raw.txt"
    raw.write_text(_HOMER_RAW)
    out = tmp_path / "peak_motif.txt"
    process_output(raw, out)
    pairs = [tuple(l.split("\t")) for l in out.read_text().splitlines()]
    assert ("chr1_1100_1200", "AP1/encode") in pairs
    assert ("chr1_1100_1200", "CTCF/encode") in pairs
    assert ("chr1_3000_3200", "AP1/encode") in pairs


def test_process_output_deduplicates(tmp_path):
    """Duplicate (peak_name, motif_name) pairs appear exactly once."""
    raw = tmp_path / "raw.txt"
    raw.write_text(_HOMER_RAW)
    out = tmp_path / "peak_motif.txt"
    process_output(raw, out)
    pairs = [tuple(l.split("\t")) for l in out.read_text().splitlines()]
    assert pairs.count(("chr1_1100_1200", "AP1/encode")) == 1


def test_process_output_sorted(tmp_path):
    """Output rows are sorted by peak_name (string sort)."""
    raw = tmp_path / "raw.txt"
    raw.write_text(_HOMER_RAW)
    out = tmp_path / "peak_motif.txt"
    process_output(raw, out)
    peak_names = [l.split("\t")[0] for l in out.read_text().splitlines()]
    assert peak_names == sorted(peak_names)


def test_process_output_skips_short_lines(tmp_path):
    """Lines with fewer than 4 fields are silently skipped."""
    raw = tmp_path / "raw.txt"
    raw.write_text(_HOMER_HEADER + "only\ttwo\n" + "chr1_100_200\t0\tACGT\tMOTIF1\t+\t5.0\n")
    out = tmp_path / "peak_motif.txt"
    process_output(raw, out)
    assert out.read_text().strip() == "chr1_100_200\tMOTIF1"


def test_process_output_returns_path(tmp_path):
    """Returns the output_file Path."""
    raw = tmp_path / "raw.txt"
    raw.write_text(_HOMER_HEADER)
    out = tmp_path / "peak_motif.txt"
    assert process_output(raw, out) == out


# ── run_homer ─────────────────────────────────────────────────────────────────

def test_run_homer_command_args(tmp_path):
    """Assembles the correct findMotifsGenome.pl command with all expected flags."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", side_effect=_fake_homer) as mock_run:
        run_homer(
            bed, "hg38", motif, tmp_path, "test",
            homer_bin="/homer/bin/findMotifsGenome.pl",
            threads=4,
        )

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/homer/bin/findMotifsGenome.pl"
    assert str(bed) in cmd
    assert "hg38" in cmd
    assert "-find" in cmd
    assert str(motif) in cmd
    assert "-p" in cmd and "4" in cmd
    assert "-size" in cmd and "given" in cmd


def test_run_homer_empty_output_raises(tmp_path):
    """returncode=0 but empty stdout file raises RuntimeError mentioning 'empty output'."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
        with pytest.raises(RuntimeError, match="empty output"):
            run_homer(bed, "hg38", motif, tmp_path, "test")


def test_run_homer_nonzero_returncode_raises(tmp_path):
    """Non-zero HOMER returncode raises RuntimeError containing the code."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", return_value=MagicMock(returncode=1, stderr="error")):
        with pytest.raises(RuntimeError, match="return code 1"):
            run_homer(bed, "hg38", motif, tmp_path, "test")


def test_run_homer_returns_raw_path(tmp_path):
    """Returns Path pointing to {output_prefix}.txt."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", side_effect=_fake_homer):
        raw = run_homer(bed, "hg38", motif, tmp_path, "myprefix")

    assert raw == tmp_path / "myprefix.txt"


# ── run_motif_scan ────────────────────────────────────────────────────────────

def test_run_motif_scan_produces_peak_motif_file(tmp_path):
    """Produces {prefix}_peak_motif.txt with correct deduplicated content."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", side_effect=_fake_homer):
        out = run_motif_scan(bed, "hg38", motif, tmp_path, "test")

    assert out == tmp_path / "test_peak_motif.txt"
    lines = out.read_text().splitlines()
    assert len(lines) == 3  # 4 raw rows minus 1 duplicate


def test_run_motif_scan_removes_intermediates_by_default(tmp_path):
    """Raw HOMER output and working directory are removed when keep_intermediate=False."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", side_effect=_fake_homer):
        run_motif_scan(bed, "hg38", motif, tmp_path, "test")

    assert not (tmp_path / "test.txt").exists()
    assert not (tmp_path / "test_motif_output").exists()


def test_run_motif_scan_keep_intermediate(tmp_path):
    """keep_intermediate=True preserves the raw output file and HOMER working directory."""
    bed = tmp_path / "peaks_named.bed"
    bed.touch()
    motif = tmp_path / "motifs.txt"
    motif.touch()

    with patch("rega.motif.subprocess.run", side_effect=_fake_homer):
        run_motif_scan(bed, "hg38", motif, tmp_path, "test", keep_intermediate=True)

    assert (tmp_path / "test.txt").exists()
    assert (tmp_path / "test_motif_output").exists()
