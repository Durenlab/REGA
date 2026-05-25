from __future__ import annotations

from pathlib import Path

import pytest

from rega.peaks import format_peak_name, rename_peaks


def test_format_peak_name_basic():
    """Canonical chr_start_end format is produced."""
    assert format_peak_name("chr1", "1000", "2000") == "chr1_1000_2000"


def test_format_peak_name_preserves_raw_string():
    """String values are used as-is; no integer conversion occurs."""
    assert format_peak_name("chr1", "0100", "0200") == "chr1_0100_0200"


def test_rename_peaks_basic_three_col(tmp_path):
    """Three-column BED produces correct four-column output with peak names."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\nchr2\t500\t600\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out)
    rows = out.read_text().splitlines()
    assert rows == [
        "chr1\t100\t200\tchr1_100_200",
        "chr2\t500\t600\tchr2_500_600",
    ]


def test_rename_peaks_extra_columns_discarded(tmp_path):
    """Columns beyond the first three (BED6 etc.) are silently discarded."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\tpeakA\t0\t+\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out)
    rows = out.read_text().splitlines()
    assert len(rows) == 1
    assert rows[0] == "chr1\t100\t200\tchr1_100_200"


def test_rename_peaks_skips_blank_lines(tmp_path):
    """Blank lines in the input are not written to the output."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\n\nchr2\t500\t600\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out)
    rows = out.read_text().splitlines()
    assert len(rows) == 2


def test_rename_peaks_skips_comment_lines(tmp_path):
    """Comment lines (starting with '#') are not written to the output."""
    bed = tmp_path / "in.bed"
    bed.write_text("# header\nchr1\t100\t200\n# another comment\nchr2\t500\t600\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out)
    rows = out.read_text().splitlines()
    assert len(rows) == 2
    assert not any(r.startswith("#") for r in rows)


def test_rename_peaks_returns_output_path(tmp_path):
    """Return value is a Path object equal to output_path."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\n")
    out = tmp_path / "out.bed"
    result = rename_peaks(bed, out)
    assert isinstance(result, Path)
    assert result == out


def test_rename_peaks_writes_to_disk(tmp_path):
    """Output file is created on disk after the call."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\n")
    out = tmp_path / "out.bed"
    assert not out.exists()
    rename_peaks(bed, out)
    assert out.exists()


def test_rename_peaks_all_comments_and_blanks(tmp_path):
    """Input with only comments and blank lines produces an empty output file."""
    bed = tmp_path / "in.bed"
    bed.write_text("# comment\n\n# another\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out)
    assert out.read_text() == ""


def test_rename_peaks_malformed_line_raises_valueerror(tmp_path):
    """A line with fewer than 3 fields raises ValueError with the physical line number."""
    bed = tmp_path / "in.bed"
    # Line 1: valid; line 2: only 2 fields → physical line 2
    bed.write_text("chr1\t100\t200\nchr2\t500\n")
    out = tmp_path / "out.bed"
    with pytest.raises(ValueError, match="Line 2"):
        rename_peaks(bed, out)


def test_rename_peaks_space_separator(tmp_path):
    """Space-separated BED is parsed correctly when sep=' '."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1 100 200\nchr2 500 600\n")
    out = tmp_path / "out.bed"
    rename_peaks(bed, out, sep=" ")
    rows = out.read_text().splitlines()
    assert rows == [
        "chr1\t100\t200\tchr1_100_200",
        "chr2\t500\t600\tchr2_500_600",
    ]


def test_rename_peaks_streams_large_input(tmp_path):
    """Verify streaming I/O by processing 10000-line file without
    loading everything in memory. Functional test, not a benchmark."""
    input_bed = tmp_path / "large.bed"
    with input_bed.open("w") as f:
        for i in range(10000):
            f.write(f"chr1\t{i * 100}\t{(i + 1) * 100}\n")
    output = tmp_path / "out.bed"
    rename_peaks(input_bed, output)
    assert output.exists()
    lines = output.read_text().splitlines()
    assert len(lines) == 10000
    assert lines[0].split("\t") == ["chr1", "0", "100", "chr1_0_100"]
    assert lines[-1].split("\t") == ["chr1", "999900", "1000000", "chr1_999900_1000000"]


def test_rename_peaks_missing_input(tmp_path):
    """Missing input file raises FileNotFoundError with informative message."""
    missing = tmp_path / "does_not_exist.bed"
    output = tmp_path / "out.bed"
    with pytest.raises(FileNotFoundError, match="Input BED file not found"):
        rename_peaks(missing, output)


def test_rename_peaks_missing_output_dir(tmp_path):
    """Missing output directory raises FileNotFoundError with clear hint."""
    bed = tmp_path / "in.bed"
    bed.write_text("chr1\t100\t200\n")
    nonexistent_dir_output = tmp_path / "nonexistent_subdir" / "out.bed"
    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        rename_peaks(bed, nonexistent_dir_output)
