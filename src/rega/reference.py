from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ReferenceData:
    """Encapsulates paths to reference genome files used by the REGA pipeline.

    Parameters
    ----------
    genome : str
        Reference genome identifier, e.g. ``"hg38"``, ``"hg19"``, ``"mm10"``.
        Used to construct default file names from templates.
    data_dir : str or Path
        Directory containing reference files.  Defaults to ``"reference_data"``
        (relative to the working directory).  Pass an absolute path when calling
        from outside the repository root.
    overrides : dict[str, str | Path] or None
        Per-file path overrides.  Keys must match entries in ``_FILE_TEMPLATES``
        (e.g. ``"tss_file"``).  An override takes precedence over the template
        path for that key.  Unknown keys raise ``ValueError`` at construction
        time to catch typos early.

    Raises
    ------
    ValueError
        If *overrides* contains keys not present in ``_FILE_TEMPLATES``.

    Examples
    --------
    Standard usage:

    >>> ref = ReferenceData(genome="hg38")
    >>> ref.tss_file
    PosixPath('reference_data/TSS_hg38.txt')

    Custom data directory:

    >>> ref = ReferenceData(genome="hg38", data_dir="/data/refs")

    Override a single file:

    >>> ref = ReferenceData(
    ...     genome="rn6",
    ...     overrides={"tss_file": "/path/to/your_TSS.txt"},
    ... )

    Check existence before use:

    >>> tss_path = ref.require("tss_file")  # raises FileNotFoundError if missing

    Notes
    -----
    **Adding custom genomes**

    To use a custom or not-yet-supported genome, you have two options:

    1. Add to reference directory (recommended for repeated use):
       Name your file ``TSS_<genome>.txt`` and place it in ``data_dir``.
       Then construct: ``ReferenceData(genome="<genome>")``.

    2. Use ``overrides`` (recommended for one-off use):

       >>> ref = ReferenceData(
       ...     genome="rn6",
       ...     overrides={"tss_file": "/path/to/your_TSS.txt"},
       ... )

       The ``genome`` field becomes a free-form label; no template
       lookup happens for keys in ``overrides``.

    TODO (v0.2+): Support ``REGA_REFERENCE_DIR`` environment variable for
    multi-directory reference lookup (e.g. lab-shared references).
    """

    # Single source of truth for file-name templates.
    # Key → template string (format receives genome=<genome>).
    # To add a new file type: add one entry here and one @property below.
    _FILE_TEMPLATES: dict[str, str] = {
        "tss_file": "TSS_{genome}.txt",
        # TODO: "chrom_sizes": "chrom_sizes_{genome}.txt",
        # TODO: "blacklist":   "blacklist_{genome}.bed",
        # TODO: "motif_db":    "motifs_{genome}/",
    }

    def __init__(
        self,
        genome: str = "hg38",
        data_dir: str | Path = "reference_data",
        overrides: dict[str, str | Path] | None = None,
    ) -> None:
        self._genome = genome
        self._data_dir = Path(data_dir)
        if overrides:
            unknown = set(overrides) - set(self._FILE_TEMPLATES)
            if unknown:
                raise ValueError(
                    f"Unknown override keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(self._FILE_TEMPLATES)}"
                )
            self._overrides: dict[str, Path] = {k: Path(v) for k, v in overrides.items()}
        else:
            self._overrides = {}
        logger.debug(
            "ReferenceData initialised: genome=%s, data_dir=%s, overrides=%s",
            genome,
            self._data_dir,
            list(self._overrides),
        )

    # ------------------------------------------------------------------
    # Read-only accessors for constructor arguments
    # ------------------------------------------------------------------

    @property
    def genome(self) -> str:
        """Reference genome identifier (e.g. ``"hg38"``)."""
        return self._genome

    @property
    def data_dir(self) -> Path:
        """Resolved base directory for reference files."""
        return self._data_dir

    # ------------------------------------------------------------------
    # Internal helper — all file properties funnel through here
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Return the Path for *key*, honouring overrides."""
        if key in self._overrides:
            return self._overrides[key]
        template = self._FILE_TEMPLATES[key]
        return self._data_dir / template.format(genome=self._genome)

    # ------------------------------------------------------------------
    # File-path properties (one per entry in _FILE_TEMPLATES)
    # ------------------------------------------------------------------

    @property
    def tss_file(self) -> Path:
        """Path to the TSS annotation file (not checked for existence)."""
        return self._resolve("tss_file")

    # TODO: add chrom_sizes, blacklist, motif_db properties here

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def require(self, file_key: str) -> Path:
        """Return the path for *file_key* and assert the file exists.

        Parameters
        ----------
        file_key : str
            A key from ``_FILE_TEMPLATES`` (e.g. ``"tss_file"``).

        Returns
        -------
        Path
            Resolved path to the file.

        Raises
        ------
        KeyError
            If *file_key* is not a recognised template key.
        FileNotFoundError
            If the resolved path does not exist.  The error message is
            scenario-specific: override path missing, genome not found but
            others exist, or no reference files at all.
        """
        if file_key not in self._FILE_TEMPLATES and file_key not in self._overrides:
            raise KeyError(
                f"Unknown file key '{file_key}'. "
                f"Known keys: {sorted(self._FILE_TEMPLATES)}"
            )
        path = self._resolve(file_key)
        if not path.exists():
            if file_key in self._overrides:
                # Scenario A: user supplied an explicit path that does not exist
                raise FileNotFoundError(
                    f"Custom {file_key} not found at user-provided path: {path}.\n"
                    f"Please check the path you supplied in overrides."
                )
            available = self.list_supported_genomes(self._data_dir)
            if available:
                # Scenario B: data_dir has other genomes but not the one requested
                expected_filename = self._FILE_TEMPLATES[file_key].format(
                    genome=self._genome
                )
                raise FileNotFoundError(
                    f"Reference file for genome '{self._genome}' not found at {path}.\n"
                    f"Available genomes in {self._data_dir}: {available}.\n"
                    f"To use a custom genome, provide your own file via:\n"
                    f"  ReferenceData(genome='{self._genome}', "
                    f"overrides={{'{file_key}': '/path/to/your_file'}})\n"
                    f"Or place {expected_filename} in {self._data_dir}."
                )
            # Scenario C: data_dir is empty or does not exist
            raise FileNotFoundError(
                f"No reference files found in {self._data_dir}.\n"
                f"Either:\n"
                f"  (a) place TSS files there (e.g. TSS_hg38.txt), or\n"
                f"  (b) provide a custom path via:\n"
                f"      ReferenceData(genome='{self._genome}', "
                f"overrides={{'{file_key}': '/path/to/your_file'}})"
            )
        logger.debug("require(%s) → %s [exists]", file_key, path)
        return path

    @classmethod
    def list_supported_genomes(cls, data_dir: str | Path = "reference_data") -> list[str]:
        """Scan *data_dir* for ``TSS_*.txt`` files and return genome identifiers.

        Parameters
        ----------
        data_dir : str or Path
            Directory to scan.  Defaults to ``"reference_data"``.

        Returns
        -------
        list[str]
            Sorted list of genome identifiers for which a TSS file exists,
            e.g. ``["hg38"]``.  Returns an empty list if *data_dir* does not
            exist or contains no matching files.
        """
        data_dir = Path(data_dir)
        _tss_pattern = re.compile(r"^TSS_(.+)\.txt$")
        genomes: list[str] = []
        if data_dir.is_dir():
            for f in sorted(data_dir.iterdir()):
                m = _tss_pattern.match(f.name)
                if m:
                    genomes.append(m.group(1))
        logger.debug("list_supported_genomes(%s) → %s", data_dir, genomes)
        return genomes

    def __repr__(self) -> str:
        parts = [
            f"genome={self._genome!r}",
            f"data_dir={str(self._data_dir)!r}",
        ]
        if self._overrides:
            parts.append(f"overrides={dict(self._overrides)!r}")
        return f"ReferenceData({', '.join(parts)})"
