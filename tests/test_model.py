from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rega.model import (
    BlockData,
    TrainingConfig,
    frob_loss,
    load_blocks,
    load_checkpoint,
    save_checkpoint,
    train_rega,
)

# Minimal config: 2 outer×2 inner each phase, stops at max_outer=3
_FAST = TrainingConfig(
    K=2,
    phase1_inner_iters=2,
    phase1_outer_iters=2,
    phase2_inner_iters=2,
    phase2_min_outer=1,
    phase2_max_outer=3,
    early_stop_patience=5,
    early_stop_rel_tol=1e-12,
    log_every_inner=100,
    V_seed=0,
    H_seed=1,
)

_NG = 2   # genes per block
_NR = 3   # RE (peaks) per block
_NM = 3   # motifs
_NS = 2   # samples


def _write_blocks(block_dir: Path, n_blocks: int = 2) -> None:
    block_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    samples = np.array(["S1", "S2"])
    motifs = np.array(["M1", "M2", "M3"])
    for i in range(n_blocks):
        chrom = f"chr{i + 1}"
        np.savez_compressed(
            str(block_dir / f"{chrom}.npz"),
            X=rng.random((_NG, _NS)).astype(np.float32),
            W=rng.random((_NG, _NR)).astype(np.float32),
            B=rng.random((_NR, _NM)).astype(np.float32),
            genes=np.array([f"G{i}{j}" for j in range(_NG)]),
            RE=np.array([f"{chrom}_{j*100}_{j*100+50}" for j in range(_NR)]),
            samples=samples,
            motifs=motifs,
            gene_chr=np.array([chrom] * _NG),
            re_chr=np.array([chrom] * _NR),
        )


# ── load_blocks ───────────────────────────────────────────────────────────────

def test_load_blocks_returns_block_data(tmp_path):
    """Returns BlockData with correct list lengths and global arrays."""
    _write_blocks(tmp_path, n_blocks=2)
    bd = load_blocks(tmp_path)
    assert len(bd.X_blocks) == 2
    assert len(bd.chr_names) == 2
    assert bd.samples.tolist() == ["S1", "S2"]
    assert bd.motifs.tolist() == ["M1", "M2", "M3"]


def test_load_blocks_no_files_raises(tmp_path):
    """ValueError when no chr*.npz files exist."""
    with pytest.raises(ValueError, match="No chr"):
        load_blocks(tmp_path)


def test_load_blocks_chroms_filter(tmp_path):
    """chroms=['chr1'] loads only chr1.npz."""
    _write_blocks(tmp_path, n_blocks=2)
    bd = load_blocks(tmp_path, chroms=["chr1"])
    assert len(bd.X_blocks) == 1
    assert bd.chr_names == ["chr1"]


def test_load_blocks_samples_mismatch_raises(tmp_path):
    """ValueError when samples are inconsistent across blocks."""
    _write_blocks(tmp_path, n_blocks=2)
    z = np.load(str(tmp_path / "chr2.npz"), allow_pickle=False)
    np.savez_compressed(
        str(tmp_path / "chr2.npz"),
        **{k: z[k] for k in z.files if k != "samples"},
        samples=np.array(["S1", "S99"]),
    )
    with pytest.raises(ValueError, match="samples inconsistent"):
        load_blocks(tmp_path)


def test_load_blocks_missing_array_raises(tmp_path):
    """ValueError when a required array key is absent from an .npz file."""
    rng = np.random.default_rng(0)
    np.savez_compressed(
        str(tmp_path / "chr1.npz"),
        X=rng.random((_NG, _NS)).astype(np.float32),
        W=rng.random((_NG, _NR)).astype(np.float32),
        # B is missing
        genes=np.array(["G0", "G1"]),
        RE=np.array(["chr1_0_50", "chr1_100_150", "chr1_200_250"]),
        samples=np.array(["S1", "S2"]),
        motifs=np.array(["M1", "M2", "M3"]),
        gene_chr=np.array(["chr1", "chr1"]),
        re_chr=np.array(["chr1"] * _NR),
    )
    with pytest.raises(ValueError, match="missing 'B'"):
        load_blocks(tmp_path)


def test_load_blocks_nan_cleaned(tmp_path):
    """NaN values in X are replaced with 0 (no longer non-finite after load)."""
    np.savez_compressed(
        str(tmp_path / "chr1.npz"),
        X=np.full((_NG, _NS), np.nan, dtype=np.float32),
        W=np.ones((_NG, _NR), dtype=np.float32),
        B=np.ones((_NR, _NM), dtype=np.float32),
        genes=np.array(["G0", "G1"]),
        RE=np.array(["chr1_0_50", "chr1_100_150", "chr1_200_250"]),
        samples=np.array(["S1", "S2"]),
        motifs=np.array(["M1", "M2", "M3"]),
        gene_chr=np.array(["chr1", "chr1"]),
        re_chr=np.array(["chr1"] * _NR),
    )
    bd = load_blocks(tmp_path)
    assert np.isfinite(bd.X_blocks[0]).all()


# ── TrainingConfig ────────────────────────────────────────────────────────────

def test_training_config_defaults():
    """Default TrainingConfig matches original run_REGA_final.py constants."""
    cfg = TrainingConfig()
    assert cfg.K == 20
    assert cfg.phase1_outer_iters == 30
    assert cfg.phase2_min_outer == 3000
    assert cfg.early_stop_patience == 50
    assert cfg.V_seed == 123


# ── frob_loss ─────────────────────────────────────────────────────────────────

def test_frob_loss_returns_finite_nonneg_scalar(tmp_path):
    """frob_loss returns a finite non-negative scalar tensor on CPU."""
    _write_blocks(tmp_path, n_blocks=1)
    bd = load_blocks(tmp_path)
    X = [torch.tensor(x) for x in bd.X_blocks]
    W = [torch.tensor(w) for w in bd.W_blocks]
    B = [torch.tensor(b) for b in bd.B_blocks]
    V = torch.rand(_NM, _FAST.K)
    H = torch.rand(_FAST.K, _NS)
    loss = frob_loss(X, W, B, V, H, _FAST)
    assert float(loss) >= 0.0
    assert torch.isfinite(loss)


# ── train_rega ��───────────────────────────────────────────────────────────────

def test_train_rega_returns_existing_pt(tmp_path):
    """train_rega returns path to an existing final.pt file."""
    blocks = tmp_path / "blocks"
    out = tmp_path / "out"
    out.mkdir()
    _write_blocks(blocks, n_blocks=2)
    bd = load_blocks(blocks)
    path = train_rega(bd, out, config=_FAST, device="cpu")
    assert path.exists()
    assert path.suffix == ".pt"


def test_train_rega_checkpoint_has_required_keys(tmp_path):
    """Checkpoint contains all expected top-level keys."""
    blocks = tmp_path / "blocks"
    out = tmp_path / "out"
    out.mkdir()
    _write_blocks(blocks, n_blocks=2)
    bd = load_blocks(blocks)
    ckpt = load_checkpoint(train_rega(bd, out, config=_FAST, device="cpu"))
    for key in ("W_blocks", "B_blocks", "V", "H", "loss", "chr_names",
                "genes_blocks", "RE_blocks", "samples", "motifs", "K"):
        assert key in ckpt, f"missing key: {key}"


def test_train_rega_loss_is_finite(tmp_path):
    """Loss stored in checkpoint is a finite float."""
    blocks = tmp_path / "blocks"
    out = tmp_path / "out"
    out.mkdir()
    _write_blocks(blocks, n_blocks=2)
    bd = load_blocks(blocks)
    ckpt = load_checkpoint(train_rega(bd, out, config=_FAST, device="cpu"))
    assert np.isfinite(ckpt["loss"])


def test_train_rega_loss_logs_written(tmp_path):
    """Both phase loss log CSVs are written to output_dir."""
    blocks = tmp_path / "blocks"
    out = tmp_path / "out"
    out.mkdir()
    _write_blocks(blocks, n_blocks=2)
    bd = load_blocks(blocks)
    train_rega(bd, out, config=_FAST, device="cpu")
    assert (out / "phase1_loss_log.csv").exists()
    assert (out / "phase2_loss_log.csv").exists()


# ── save / load checkpoint roundtrip ─────────────────────────────────────────

def test_save_load_checkpoint_roundtrip(tmp_path):
    """save_checkpoint / load_checkpoint preserves V, H shapes and step."""
    _write_blocks(tmp_path / "blocks", n_blocks=1)
    bd = load_blocks(tmp_path / "blocks")
    V = torch.rand(_NM, _FAST.K)
    H = torch.rand(_FAST.K, _NS)
    W = [torch.rand(_NG, _NR)]
    B = [torch.rand(_NR, _NM)]
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, 7, W, B, V, H, 3.14, ["chr1"], bd, _FAST)
    ckpt = load_checkpoint(path)
    assert ckpt["V"].shape == V.shape
    assert ckpt["H"].shape == H.shape
    assert ckpt["step"] == 7
    assert abs(ckpt["loss"] - 3.14) < 1e-6
