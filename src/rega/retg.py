from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pybedtools

__all__ = [
    "build_re_tg",
    "expand_tss_windows",
    "intersect_peaks_with_windows",
    "ChromosomeMismatchError",
]

_RETG_HEADER = (
    "chrom\tregion_start\tregion_end\tgene_name\tTSS\tstrand\t"
    "peak_chrom\tpeak_start\tpeak_end\tpeak_name"
)


class ChromosomeMismatchError(ValueError):
    """Raised when bedtools intersect returns 0 rows, likely a chromosome naming mismatch."""


def _sample_chroms(path: Path, n: int = 5) -> list[str]:
    """Return up to n unique chromosome names (first column) from a BED-like file."""
    seen: list[str] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            chrom = line.split("\t", 1)[0].rstrip()
            if chrom not in seen:
                seen.append(chrom)
            if len(seen) >= n:
                break
    return seen


def expand_tss_windows(
    tss_file: Path,
    output_bed: Path,
    window: int = 300_000,
    chroms: list[str] | None = None,
) -> Path:
    """Write a 6-col BED of TSS ± window regions; return output_bed.

    Input tss_file: 4-col TSV, no header — chrom, TSS_pos, gene_name, strand.
    Output columns: chrom, region_start, region_end, gene_name, TSS_pos, strand.

    chroms: whitelist of chromosome names to retain.  Matching is exact and
    case-sensitive ("chr1" does not match "CHR1" or "Chr1").  None keeps all
    chromosomes.  An empty list ([]) is valid and produces a file with zero
    data rows.
    """
    with open(tss_file) as fin, open(output_bed, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, tss_str, gene, strand = line.rstrip("\n").split("\t", 3)
            if chroms is not None and chrom not in chroms:
                continue
            tss_pos = int(tss_str)
            # Clamp to 0: TSS near chromosome origin can produce negative coords
            # that bedtools rejects.
            region_start = max(0, tss_pos - window)
            region_end = tss_pos + window
            # TODO: clamp region_end at chromosome length once chrom_sizes
            #       is available via ReferenceData
            fout.write(
                f"{chrom}\t{region_start}\t{region_end}\t{gene}\t{tss_pos}\t{strand}\n"
            )
    return output_bed


def intersect_peaks_with_windows(
    windows_bed: Path,
    peaks_bed: Path,
    output_tsv: Path,
) -> Path:
    """Run bedtools intersect -wa -wb; write 10-col TSV with header; return output_tsv.

    Output columns: chrom, region_start, region_end, gene_name, TSS, strand,
    peak_chrom, peak_start, peak_end, peak_name.

    output_tsv is written atomically: the file is absent if this function raises,
    so callers never receive a partial or empty result file.

    Raises ChromosomeMismatchError if intersect returns 0 rows.  The error
    message includes sample chromosome names from both inputs to aid diagnosis.

    Requires bedtools to be discoverable by pybedtools (on PATH, or configured
    via pybedtools.helpers.set_bedtools_path before calling).
    """
    tmp_dir = tempfile.mkdtemp(prefix="retg_bedtools_")
    old_tempdir = pybedtools.get_tempdir()
    try:
        pybedtools.set_tempdir(tmp_dir)
        intersected = pybedtools.BedTool(str(windows_bed)).intersect(
            pybedtools.BedTool(str(peaks_bed)), wa=True, wb=True
        )

        staging = Path(tmp_dir) / "intersect_staging.tsv"
        row_count = 0
        with open(staging, "w") as fout:
            fout.write(_RETG_HEADER + "\n")
            for interval in intersected:
                fout.write(str(interval))
                row_count += 1

        if row_count == 0:
            win_chroms = _sample_chroms(windows_bed)
            peak_chroms = _sample_chroms(peaks_bed)
            raise ChromosomeMismatchError(
                "bedtools intersect returned 0 rows — likely chromosome name mismatch.\n"
                f"  Window chromosomes (first {len(win_chroms)}):  {win_chroms}\n"
                f"  Peak chromosomes  (first {len(peak_chroms)}): {peak_chroms}\n"
                "Suggestion: normalize chromosome names to the same convention "
                "(e.g., add/remove 'chr' prefix)."
            )

        shutil.move(str(staging), str(output_tsv))
    finally:
        pybedtools.set_tempdir(old_tempdir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_tsv


def build_re_tg(
    tss_file: Path,
    peaks_bed: Path,
    output_tsv: Path,
    window: int = 300_000,
    chroms: list[str] | None = None,
    keep_intermediate: bool = False,
    intermediate_dir: Path | None = None,
) -> Path:
    """Build the RE-TG pair table: expand TSS windows → bedtools intersect → write TSV.

    Orchestrates expand_tss_windows then intersect_peaks_with_windows.  The
    intermediate expanded-TSS BED file is managed as follows:

    keep_intermediate=False (default): written to a temporary directory and
        deleted on function exit.  intermediate_dir is ignored.
    keep_intermediate=True, intermediate_dir=None: written to output_tsv.parent.
    keep_intermediate=True, intermediate_dir=Path: written to intermediate_dir
        (caller must ensure the directory exists).
    """
    if keep_intermediate:
        inter_dir: Path = (
            intermediate_dir if intermediate_dir is not None else output_tsv.parent
        )
        own_tmp: str | None = None
    else:
        own_tmp = tempfile.mkdtemp(prefix="retg_inter_")
        inter_dir = Path(own_tmp)

    try:
        windows_bed = inter_dir / f"{output_tsv.stem}_tss_windows.bed"
        expand_tss_windows(tss_file, windows_bed, window=window, chroms=chroms)
        print(f"[build_re_tg] TSS windows written: {windows_bed}")
        intersect_peaks_with_windows(windows_bed, peaks_bed, output_tsv)
        print(f"[build_re_tg] RE-TG table written: {output_tsv}")
    finally:
        if own_tmp is not None:
            shutil.rmtree(own_tmp, ignore_errors=True)

    return output_tsv
