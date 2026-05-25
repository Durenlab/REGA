from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["run_homer", "process_output", "run_motif_scan"]


def run_homer(
    named_bed: Path,
    genome: str,
    motif_file: Path,
    output_dir: Path,
    output_prefix: str,
    homer_bin: str | Path = "findMotifsGenome.pl",
    threads: int = 1,
) -> Path:
    """Call HOMER findMotifsGenome.pl -find and return path to the raw output file.

    HOMER writes per-peak motif hits to stdout; the file is tab-separated with a header line.
    A working directory ({output_prefix}_motif_output/) is created inside output_dir by HOMER.

    Raises RuntimeError if HOMER exits with a non-zero return code.
    Caller must ensure output_dir exists.
    """
    intermediate_dir = output_dir / f"{output_prefix}_motif_output"
    intermediate_dir.mkdir(exist_ok=True)
    raw_output = output_dir / f"{output_prefix}.txt"

    cmd = [
        str(homer_bin),
        str(named_bed),
        genome,
        str(intermediate_dir),
        "-size", "given",
        "-p", str(threads),
        "-find", str(motif_file),
    ]
    print(f"[run_homer] {' '.join(cmd)}")

    with open(raw_output, "w") as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"HOMER exited with return code {result.returncode}.\n"
            f"stderr (last 2000 chars): {result.stderr[-2000:]}"
        )
    if raw_output.stat().st_size == 0:
        raise RuntimeError(
            f"HOMER returned 0 but produced empty output: {raw_output}. "
            "Check that the genome package is installed and the peak BED is valid."
        )

    print(f"[run_homer] raw output: {raw_output}")
    return raw_output


def process_output(raw_homer_output: Path, output_file: Path) -> Path:
    """Extract (peak_name, motif_name) pairs from raw HOMER -find output; write deduplicated TSV.

    HOMER -find raw format (tab-separated, one header line):
      PositionID  Offset  Sequence  MotifName  Strand  MotifScore

    Output: 2-col TSV (no header), sorted by peak_name, with duplicates removed.
    Caller must ensure output_file.parent exists.
    """
    rows: list[tuple[str, str]] = []
    with open(raw_homer_output) as fh:
        next(fh)  # skip header
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            rows.append((fields[0], fields[3]))

    rows.sort(key=lambda x: x[0])

    seen: set[tuple[str, str]] = set()
    with open(output_file, "w") as out:
        for row in rows:
            if row not in seen:
                out.write(f"{row[0]}\t{row[1]}\n")
                seen.add(row)

    print(f"[process_output] {len(seen)} unique (peak, motif) pairs → {output_file}")
    return output_file


def run_motif_scan(
    named_bed: Path,
    genome: str,
    motif_file: Path,
    output_dir: Path,
    output_prefix: str,
    homer_bin: str | Path = "findMotifsGenome.pl",
    threads: int = 1,
    keep_intermediate: bool = False,
) -> Path:
    """Run HOMER motif scan and post-process into {output_prefix}_peak_motif.txt.

    Wraps run_homer() and process_output(). When keep_intermediate=False (default),
    the raw HOMER output file and working directory are removed after the final file
    is written.

    Returns path to the final peak_motif TSV. Caller must ensure output_dir exists.
    """
    print(f"[run_motif_scan] bed={named_bed.name}, genome={genome}, prefix={output_prefix}")
    raw = run_homer(
        named_bed, genome, motif_file, output_dir, output_prefix,
        homer_bin=homer_bin, threads=threads,
    )
    peak_motif = output_dir / f"{output_prefix}_peak_motif.txt"
    process_output(raw, peak_motif)

    if not keep_intermediate:
        raw.unlink(missing_ok=True)
        shutil.rmtree(output_dir / f"{output_prefix}_motif_output", ignore_errors=True)
        print(f"[run_motif_scan] removed intermediates")

    return peak_motif
