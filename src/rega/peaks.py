from __future__ import annotations

from pathlib import Path
from typing import Iterator


def format_peak_name(chrom: str, start: str, end: str) -> str:
    """Return the canonical REGA peak name for a genomic interval.

    Concatenates *chrom*, *start*, and *end* with underscores.  Raw string
    values are used without integer conversion, so leading zeros and any
    other formatting in the original BED file are preserved.

    Parameters
    ----------
    chrom : str
        Chromosome name (e.g. ``"chr1"``).
    start : str
        Interval start coordinate as a string (e.g. ``"1000"``).
    end : str
        Interval end coordinate as a string (e.g. ``"2000"``).

    Returns
    -------
    str
        Peak name in ``{chrom}_{start}_{end}`` format.

    Examples
    --------
    >>> format_peak_name("chr1", "1000", "2000")
    'chr1_1000_2000'
    """
    return f"{chrom}_{start}_{end}"


def _iter_bed_records(
    path: Path,
    sep: str,
) -> Iterator[tuple[str, str, str]]:
    """Open *path* and yield ``(chrom, start, end)`` for each data line.

    Blank lines and lines starting with ``'#'`` are skipped silently.
    Line numbers in error messages are physical (1-based, counting every
    line in the file including blank and comment lines), matching the line
    numbers shown by text editors.

    Parameters
    ----------
    path : Path
        BED file to read.
    sep : str
        Field separator used to split each data line.

    Yields
    ------
    tuple[str, str, str]
        ``(chrom, start, end)`` extracted from the first three fields of
        each data line.

    Raises
    ------
    ValueError
        If a data line contains fewer than 3 fields.  The message includes
        the physical line number, file path, field count, and raw line text
        to aid debugging.
    """
    with open(path) as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(sep)
            if len(fields) < 3:
                raise ValueError(
                    f"Line {line_num} in {path}: expected at least 3 fields, "
                    f"got {len(fields)}: {raw!r}"
                )
            yield fields[0], fields[1], fields[2]


def rename_peaks(
    input_bed: str | Path,
    output_path: str | Path,
    sep: str = "\t",
) -> Path:
    """Read a BED file, append a chr_start_end peak-name column, and write
    a four-column BED file to *output_path*.

    Parameters
    ----------
    input_bed : str or Path
        Input BED file (tab-separated by default, at least 3 columns).
        Comment lines (starting with ``'#'``) and blank lines are skipped.
        Columns beyond the first three are discarded.
    output_path : str or Path
        Full path for the output file.  The parent directory must already
        exist; this function does not create directories.
    sep : str
        Field separator used when reading *input_bed*.  Default ``"\t"``.

    Returns
    -------
    Path
        Path object pointing to the written output file.

    Raises
    ------
    FileNotFoundError
        If *input_bed* does not exist, or if the parent directory of
        *output_path* does not exist.
    ValueError
        If any data line in *input_bed* contains fewer than 3 fields.

    Notes
    -----
    Input is processed line-by-line (streaming); the full file is never
    loaded into memory.  The output always uses tab (``"\\t"``) as the
    field separator regardless of *sep*.

    Examples
    --------
    >>> from pathlib import Path
    >>> path = rename_peaks("peaks.bed", "peaks_named.bed")  # doctest: +SKIP
    """
    input_bed = Path(input_bed)
    output_path = Path(output_path)

    # Fail early before touching the output file
    if not input_bed.exists():
        raise FileNotFoundError(f"Input BED file not found: {input_bed}")

    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}. "
            f"Create it first or pass an existing directory."
        )

    count = 0
    with open(output_path, "w") as outfile:
        for chrom, start, end in _iter_bed_records(input_bed, sep):
            peak_name = format_peak_name(chrom, start, end)
            outfile.write(f"{chrom}\t{start}\t{end}\t{peak_name}\n")
            count += 1

    print(f"[peaks] {count} peaks written → {output_path}")
    return output_path
