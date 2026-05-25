from __future__ import annotations

from pathlib import Path

import pytest

from rega.reference import ReferenceData


def test_tss_file_default_path():
    """Default TSS path is constructed from genome + data_dir template."""
    ref = ReferenceData(genome="hg38", data_dir="/fake/dir")
    assert ref.tss_file == Path("/fake/dir/TSS_hg38.txt")


def test_tss_file_override():
    """Override path takes precedence over the template-derived path."""
    ref = ReferenceData(genome="hg38", overrides={"tss_file": "/my/tss.txt"})
    assert ref.tss_file == Path("/my/tss.txt")


def test_require_existing_file(tmp_path):
    """require() returns the path when the file exists on disk."""
    tss = tmp_path / "TSS_hg38.txt"
    tss.write_text("chr1\t100\t200\n")
    ref = ReferenceData(genome="hg38", data_dir=tmp_path)
    assert ref.require("tss_file") == tss


def test_require_missing_dir_falls_to_scenario_c():
    """data_dir does not exist on disk → Scenario C path."""
    ref = ReferenceData(genome="hg19", data_dir="/nonexistent")
    with pytest.raises(FileNotFoundError, match="No reference files found"):
        ref.require("tss_file")


def test_require_unknown_key():
    """require() raises KeyError for unrecognised file key."""
    ref = ReferenceData(genome="hg38")
    with pytest.raises(KeyError, match="Unknown file key"):
        ref.require("unknown_key")


def test_list_supported_genomes(tmp_path):
    """list_supported_genomes() returns only TSS_*.txt matches, sorted."""
    (tmp_path / "TSS_hg38.txt").write_text("")
    (tmp_path / "TSS_mm10.txt").write_text("")
    (tmp_path / "other_file.txt").write_text("")
    result = ReferenceData.list_supported_genomes(tmp_path)
    assert result == ["hg38", "mm10"]


def test_list_supported_genomes_missing_dir():
    """list_supported_genomes() returns [] when data_dir does not exist."""
    assert ReferenceData.list_supported_genomes("/nonexistent") == []


def test_unknown_override_key():
    """Unknown override key raises ValueError at __init__ time."""
    with pytest.raises(ValueError, match="Unknown override keys"):
        ReferenceData(genome="hg38", overrides={"typo_key": "/some/path"})


def test_require_override_missing(tmp_path):
    """require() Scenario A: override path does not exist on disk."""
    ref = ReferenceData(
        genome="rn6",
        data_dir=tmp_path,
        overrides={"tss_file": "/nonexistent/path.txt"},
    )
    with pytest.raises(FileNotFoundError, match="user-provided path"):
        ref.require("tss_file")


def test_require_missing_with_other_genomes(tmp_path):
    """require() Scenario B: data_dir has other genomes but not the requested one."""
    (tmp_path / "TSS_hg38.txt").write_text("")
    ref = ReferenceData(genome="rn6", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Available genomes.*hg38"):
        ref.require("tss_file")


def test_require_no_references(tmp_path):
    """require() Scenario C: data_dir exists but contains no TSS files."""
    ref = ReferenceData(genome="hg38", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No reference files found"):
        ref.require("tss_file")


def test_custom_genome_via_overrides(tmp_path):
    """End-to-end: custom genome via overrides, real file, require() succeeds."""
    custom_tss = tmp_path / "my_rn6_tss.txt"
    custom_tss.write_text("chr1\t100\t200\n")
    ref = ReferenceData(
        genome="rn6",
        overrides={"tss_file": custom_tss},
    )
    assert ref.tss_file == custom_tss
    assert ref.require("tss_file") == custom_tss
