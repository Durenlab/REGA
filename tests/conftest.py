from __future__ import annotations

import sys
from pathlib import Path

import pybedtools


def pytest_configure(config):
    """Auto-configure pybedtools to find bedtools alongside the Python interpreter."""
    bedtools_bin = Path(sys.executable).parent / "bedtools"
    if bedtools_bin.exists():
        pybedtools.helpers.set_bedtools_path(str(bedtools_bin.parent))
