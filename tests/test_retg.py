from __future__ import annotations

from pathlib import Path

import pytest

from rega.retg import (
    ChromosomeMismatchError,
    build_re_tg,
    expand_tss_windows,
    intersect_peaks_with_windows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "retg"
TSS_MINI = FIXTURES / "tss_mini.txt"
PEAKS_MINI = FIXTURES / "peaks_mini.bed"
PEAKS_NOCHR = FIXTURES / "peaks_nochr.bed"

WINDOW = 1000  # small window matched to the mini fixture coordinates


@pytest.fixture
def windows_mini(tmp_path):
    """Pre-built expanded-TSS BED (window=1000) for use in intersect tests."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW)
    return out


# ── expand_tss_windows ────────────────────────────────────────────────────────


def test_expand_tss_windows_row_count(tmp_path):
    """6 TSS entries with window=1000 produce 6 output rows."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW)
    rows = [l for l in out.read_text().splitlines() if l]
    assert len(rows) == 6


def test_expand_tss_windows_column_values(tmp_path):
    """GENE_A row (TSS=5000, window=1000) has correct values in all 6 columns."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW)
    gene_a = next(r for r in out.read_text().splitlines() if "GENE_A" in r)
    cols = gene_a.split("\t")
    assert cols[0] == "chr1"
    assert cols[1] == "4000"
    assert cols[2] == "6000"
    assert cols[3] == "GENE_A"
    assert cols[4] == "5000"
    assert cols[5] == "+"


def test_expand_tss_windows_negative_start_clamped(tmp_path):
    """GENE_E (TSS=500, window=1000): region_start clamped to 0, not -500."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW)
    gene_e = next(r for r in out.read_text().splitlines() if "GENE_E" in r)
    assert int(gene_e.split("\t")[1]) == 0


def test_expand_tss_windows_chroms_whitelist(tmp_path):
    """chroms=["chr1"] excludes GENE_F (chr2); output has 5 rows."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW, chroms=["chr1"])
    rows = [l for l in out.read_text().splitlines() if l]
    assert len(rows) == 5
    assert not any("chr2" in r for r in rows)


def test_expand_tss_windows_chroms_empty_list(tmp_path):
    """chroms=[] produces a file with 0 data rows (file still exists)."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW, chroms=[])
    assert out.exists()
    assert len([l for l in out.read_text().splitlines() if l]) == 0


def test_expand_tss_windows_chroms_none_keeps_all(tmp_path):
    """chroms=None retains all 6 rows including chr2."""
    out = tmp_path / "windows.bed"
    expand_tss_windows(TSS_MINI, out, window=WINDOW, chroms=None)
    rows = [l for l in out.read_text().splitlines() if l]
    assert len(rows) == 6
    assert any("chr2" in r for r in rows)


def test_expand_tss_windows_returns_output_path(tmp_path):
    """Return value is the output_bed Path."""
    out = tmp_path / "windows.bed"
    result = expand_tss_windows(TSS_MINI, out, window=WINDOW)
    assert result == out


# ── intersect_peaks_with_windows ──────────────────────────────────────────────


def test_intersect_row_count(tmp_path, windows_mini):
    """Intersect of mini fixture gives 5 data rows plus 1 header (6 total lines)."""
    out = tmp_path / "re_tg.txt"
    intersect_peaks_with_windows(windows_mini, PEAKS_MINI, out)
    lines = [l for l in out.read_text().splitlines() if l]
    assert len(lines) == 6


def test_intersect_header_correct(tmp_path, windows_mini):
    """First line of output matches the expected 10-column header exactly."""
    out = tmp_path / "re_tg.txt"
    intersect_peaks_with_windows(windows_mini, PEAKS_MINI, out)
    header = out.read_text().splitlines()[0]
    expected = (
        "chrom\tregion_start\tregion_end\tgene_name\tTSS\tstrand\t"
        "peak_chrom\tpeak_start\tpeak_end\tpeak_name"
    )
    assert header == expected


def test_intersect_one_peak_two_genes(tmp_path, windows_mini):
    """chr1_15600_15900 appears in 2 rows: one for GENE_C and one for GENE_D."""
    out = tmp_path / "re_tg.txt"
    intersect_peaks_with_windows(windows_mini, PEAKS_MINI, out)
    data_rows = out.read_text().splitlines()[1:]
    matching = [r for r in data_rows if "chr1_15600_15900" in r]
    assert len(matching) == 2
    genes = {r.split("\t")[3] for r in matching}
    assert genes == {"GENE_C", "GENE_D"}


def test_intersect_out_of_window_peak_absent(tmp_path, windows_mini):
    """chr1_12000_12500 (outside all windows) does not appear in output."""
    out = tmp_path / "re_tg.txt"
    intersect_peaks_with_windows(windows_mini, PEAKS_MINI, out)
    assert "chr1_12000_12500" not in out.read_text()


def test_intersect_mismatch_raises(tmp_path, windows_mini):
    """Peaks file without 'chr' prefix raises ChromosomeMismatchError."""
    out = tmp_path / "re_tg.txt"
    with pytest.raises(ChromosomeMismatchError):
        intersect_peaks_with_windows(windows_mini, PEAKS_NOCHR, out)


def test_intersect_mismatch_error_contains_chrom_samples(tmp_path, windows_mini):
    """ChromosomeMismatchError message includes chrom samples from both input files."""
    out = tmp_path / "re_tg.txt"
    with pytest.raises(ChromosomeMismatchError) as exc_info:
        intersect_peaks_with_windows(windows_mini, PEAKS_NOCHR, out)
    msg = str(exc_info.value)
    assert "chr1" in msg   # from windows_bed
    assert "'1'" in msg    # from peaks_nochr (list repr: ['1'])


def test_intersect_no_output_file_on_mismatch(tmp_path, windows_mini):
    """output_tsv does not exist after ChromosomeMismatchError (atomic write)."""
    out = tmp_path / "re_tg.txt"
    with pytest.raises(ChromosomeMismatchError):
        intersect_peaks_with_windows(windows_mini, PEAKS_NOCHR, out)
    assert not out.exists()


def test_intersect_returns_output_path(tmp_path, windows_mini):
    """Return value equals output_tsv."""
    out = tmp_path / "re_tg.txt"
    result = intersect_peaks_with_windows(windows_mini, PEAKS_MINI, out)
    assert result == out


# ── build_re_tg ───────────────────────────────────────────────────────────────


def test_build_re_tg_end_to_end(tmp_path):
    """End-to-end with default settings: output_tsv exists with 5 data rows."""
    out = tmp_path / "re_tg.txt"
    build_re_tg(TSS_MINI, PEAKS_MINI, out, window=WINDOW)
    lines = [l for l in out.read_text().splitlines() if l]
    assert len(lines) == 6  # 1 header + 5 data


def test_build_re_tg_no_intermediate_in_output_dir(tmp_path):
    """keep_intermediate=False: no *_tss_windows.bed appears in output_tsv's directory."""
    out = tmp_path / "re_tg.txt"
    build_re_tg(TSS_MINI, PEAKS_MINI, out, window=WINDOW, keep_intermediate=False)
    assert list(tmp_path.glob("*_tss_windows.bed")) == []


def test_build_re_tg_keep_intermediate_true(tmp_path):
    """keep_intermediate=True: intermediate BED is retained in intermediate_dir."""
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir()
    out = tmp_path / "re_tg.txt"
    build_re_tg(
        TSS_MINI, PEAKS_MINI, out, window=WINDOW,
        keep_intermediate=True, intermediate_dir=inter_dir,
    )
    assert len(list(inter_dir.glob("*_tss_windows.bed"))) == 1


def test_build_re_tg_chroms_filter(tmp_path):
    """chroms=["chr1"] excludes GENE_F: 4 data rows in output."""
    out = tmp_path / "re_tg.txt"
    build_re_tg(TSS_MINI, PEAKS_MINI, out, window=WINDOW, chroms=["chr1"])
    lines = [l for l in out.read_text().splitlines() if l]
    assert len(lines) == 5  # 1 header + 4 data


def test_build_re_tg_returns_path(tmp_path):
    """Return value equals output_tsv."""
    out = tmp_path / "re_tg.txt"
    result = build_re_tg(TSS_MINI, PEAKS_MINI, out, window=WINDOW)
    assert result == out
