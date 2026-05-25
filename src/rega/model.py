from __future__ import annotations

import csv
import glob
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for rega.model but not installed. "
        "Install from https://pytorch.org/get-started/locally/ "
        "to get the correct CUDA/CPU build for your system."
    ) from e

__all__ = [
    "TrainingConfig",
    "BlockData",
    "load_blocks",
    "train_rega",
    "frob_loss",
    "save_checkpoint",
    "load_checkpoint",
]


@dataclass
class TrainingConfig:
    """Hyperparameters for two-phase REGA multiplicative-update training.

    All defaults reproduce the original run_REGA_final.py settings.
    Use dataclasses.replace(config, K=10, ...) to create modified configs.
    """
    # Model
    K: int = 20
    # Numerics
    eps: float = 1e-10
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    # Regularization (non-negative L1 on W, B, V, H)
    lam_W: float = 0.0
    lam_B: float = 0.0
    lam_V: float = 0.0
    lam_H: float = 0.0
    clamp_min: float = 0.0
    # Phase 1 — initialization (fixed outer iterations)
    phase1_inner_iters: int = 2000
    phase1_outer_iters: int = 30
    # Phase 2 — refinement with early stopping
    phase2_inner_iters: int = 500
    phase2_min_outer: int = 3000
    phase2_max_outer: int = 6000
    early_stop_patience: int = 50
    early_stop_rel_tol: float = 1e-5
    # Logging
    log_every_inner: int = 500
    # Reproducibility
    V_seed: int = 123
    H_seed: int = 1234


@dataclass
class BlockData:
    """Per-chromosome matrix blocks loaded from .npz files (B6 output)."""
    X_blocks: list[np.ndarray]
    W_blocks: list[np.ndarray]
    B_blocks: list[np.ndarray]
    chr_names: list[str]
    genes_blocks: list[np.ndarray]
    RE_blocks: list[np.ndarray]
    samples: np.ndarray
    motifs: np.ndarray


def load_blocks(block_dir: Path, chroms: list[str] | None = None) -> BlockData:
    """Load per-chromosome .npz blocks from block_dir into a BlockData.

    Validates per-block and cross-block consistency (array shapes, samples, motifs).
    NaN/Inf values in X, W, B are silently replaced with 0.

    Raises ValueError if no files are found, required arrays are missing,
    or consistency checks fail.
    """
    npz_files = sorted(glob.glob(str(block_dir / "chr*.npz")))
    if not npz_files:
        raise ValueError(f"No chr*.npz files found in {block_dir}")

    X_blocks: list[np.ndarray] = []
    W_blocks: list[np.ndarray] = []
    B_blocks: list[np.ndarray] = []
    chr_names: list[str] = []
    genes_blocks: list[np.ndarray] = []
    RE_blocks: list[np.ndarray] = []
    samples: np.ndarray | None = None
    motifs: np.ndarray | None = None

    for f in npz_files:
        chr_name = Path(f).stem
        if chroms is not None and chr_name not in chroms:
            continue

        z = np.load(f, allow_pickle=False)
        for key in ("X", "W", "B", "genes", "RE", "samples", "motifs"):
            if key not in z.files:
                raise ValueError(f"{chr_name}.npz missing '{key}' array")

        X = z["X"].astype(np.float32)
        W = z["W"].astype(np.float32)
        B = z["B"].astype(np.float32)

        if X.size == 0 or W.size == 0 or B.size == 0:
            continue

        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(W).all():
            W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(B).all():
            B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

        genes_i = np.asarray(z["genes"]).astype(str)
        RE_i = np.asarray(z["RE"]).astype(str)
        samples_i = np.asarray(z["samples"]).astype(str)
        motifs_i = np.asarray(z["motifs"]).astype(str)

        # Per-block shape consistency
        if X.shape[0] != len(genes_i):
            raise ValueError(f"{chr_name}: X rows {X.shape[0]} != genes {len(genes_i)}")
        if W.shape[0] != len(genes_i):
            raise ValueError(f"{chr_name}: W rows {W.shape[0]} != genes {len(genes_i)}")
        if W.shape[1] != len(RE_i):
            raise ValueError(f"{chr_name}: W cols {W.shape[1]} != RE {len(RE_i)}")
        if B.shape[0] != len(RE_i):
            raise ValueError(f"{chr_name}: B rows {B.shape[0]} != RE {len(RE_i)}")
        if B.shape[1] != len(motifs_i):
            raise ValueError(f"{chr_name}: B cols {B.shape[1]} != motifs {len(motifs_i)}")

        # Cross-block samples / motifs consistency
        if samples is None:
            samples = samples_i
        elif not np.array_equal(samples, samples_i):
            raise ValueError(f"{chr_name}: samples inconsistent with previous blocks")
        if motifs is None:
            motifs = motifs_i
        elif not np.array_equal(motifs, motifs_i):
            raise ValueError(f"{chr_name}: motifs inconsistent with previous blocks")

        X_blocks.append(X)
        W_blocks.append(W)
        B_blocks.append(B)
        chr_names.append(chr_name)
        genes_blocks.append(genes_i)
        RE_blocks.append(RE_i)

    if not X_blocks:
        raise ValueError(f"No valid blocks loaded from {block_dir}")

    n = X_blocks[0].shape[1]
    p = B_blocks[0].shape[1]
    for i, (X, B) in enumerate(zip(X_blocks, B_blocks)):
        if X.shape[1] != n:
            raise ValueError(f"Block {i}: sample count {X.shape[1]} != {n}")
        if B.shape[1] != p:
            raise ValueError(f"Block {i}: motif count {B.shape[1]} != {p}")

    print(f"[load_blocks] {len(X_blocks)} blocks; n={n}; p={p}; "
          f"samples={len(samples)}; motifs={len(motifs)}")
    return BlockData(
        X_blocks=X_blocks, W_blocks=W_blocks, B_blocks=B_blocks,
        chr_names=chr_names, genes_blocks=genes_blocks, RE_blocks=RE_blocks,
        samples=samples, motifs=motifs,
    )


def frob_loss(
    X_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    """Frobenius norm squared loss sum_c ||X_c - W_c B_c V H||^2 plus optional L1 regularization."""
    device, dtype = V.device, V.dtype
    loss = torch.tensor(0.0, device=device, dtype=dtype)
    Z = V @ H
    for Xc, Wc, Bc in zip(X_blocks, W_blocks, B_blocks):
        diff = Xc - Wc @ (Bc @ Z)
        loss = loss + (diff * diff).sum()
    if config.lam_W:
        loss = loss + config.lam_W * sum(w.sum() for w in W_blocks)
    if config.lam_B:
        loss = loss + config.lam_B * sum(b.sum() for b in B_blocks)
    if config.lam_V:
        loss = loss + config.lam_V * V.sum()
    if config.lam_H:
        loss = loss + config.lam_H * H.sum()
    return loss


def save_checkpoint(
    path: Path,
    step: int,
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    loss: float,
    chr_names: list[str],
    block_data: BlockData,
    config: TrainingConfig,
    training_info: dict[str, Any] | None = None,
) -> None:
    """Save a self-contained training checkpoint to path."""
    ckpt = {
        "step": step,
        "W_blocks": [w.detach().cpu() for w in W_blocks],
        "B_blocks": [b.detach().cpu() for b in B_blocks],
        "V": V.detach().cpu(),
        "H": H.detach().cpu(),
        "loss": float(loss),
        "chr_names": chr_names,
        "genes_blocks": block_data.genes_blocks,
        "RE_blocks": block_data.RE_blocks,
        "samples": block_data.samples,
        "motifs": block_data.motifs,
        "K": config.K,
        "EPS": config.eps,
        "training_info": training_info,
    }
    torch.save(ckpt, path)
    print(f"[save_checkpoint] {path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by save_checkpoint; returns the raw dict."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")
    print(f"[load_checkpoint] step={ckpt.get('step')}, loss={ckpt.get('loss'):.6e}")
    return ckpt


def _write_loss_log_header(log_path: Path) -> None:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["phase", "outer", "loss", "relative_loss_change", "time_sec"])


def _append_loss_log(
    log_path: Path, phase: str, outer: int,
    loss: float, rel_change: float, elapsed: float,
) -> None:
    with open(log_path, "a", newline="") as f:
        rel_str = "" if not np.isfinite(rel_change) else f"{rel_change:.6e}"
        csv.writer(f).writerow([phase, outer, f"{loss:.6e}", rel_str, f"{elapsed:.3f}"])


def mu_train(
    X_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    config: TrainingConfig,
    inner_iters: int,
    outer_iters: int,
    device: str,
    phase_name: str = "phase",
    loss_log_path: Path | None = None,
    final_ckpt_path: Path | None = None,
    early_stop: bool = False,
    min_outer: int = 0,
    max_outer: int | None = None,
    block_data: BlockData | None = None,
    training_info: dict[str, Any] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, float, int]:
    """Single-phase multiplicative-update training loop.

    Phase 1 (early_stop=False): runs exactly outer_iters outer steps.
    Phase 2 (early_stop=True): patience counting begins after min_outer outer steps;
      stops when n_stagnant >= patience or it >= max_outer.

    Returns (W_blocks, B_blocks, V, H, final_loss, n_outer_iters).
    Raises FloatingPointError if loss becomes NaN/Inf.
    """
    if not (len(X_blocks) == len(W_blocks) == len(B_blocks)):
        raise ValueError("X_blocks, W_blocks, B_blocks must have the same length")

    eps = config.eps
    dtype = config.dtype

    if loss_log_path is not None:
        _write_loss_log_header(loss_log_path)

    with torch.no_grad():
        init_loss = float(frob_loss(X_blocks, W_blocks, B_blocks, V, H, config).cpu())
    if not np.isfinite(init_loss):
        raise FloatingPointError(f"[{phase_name}] initial loss is non-finite: {init_loss}")
    print(f"[{phase_name}] initial loss = {init_loss:.6e}")

    prev_loss = init_loss
    last_loss = init_loss
    n_stagnant = 0
    it = 0

    while True:
        it += 1
        t0 = time.time()

        with torch.no_grad():
            Z = V @ H
            ZZt = Z @ Z.t()

            for Xc, Wc, Bc in zip(X_blocks, W_blocks, B_blocks):
                Yc = Bc @ Z
                Wc.mul_((Xc @ Yc.t()) / (Wc @ (Yc @ Yc.t()) + config.lam_W + eps))
                Wc.clamp_(min=config.clamp_min)

                Bc.mul_(((Wc.t() @ Xc) @ Z.t()) / ((Wc.t() @ Wc) @ Bc @ ZZt + config.lam_B + eps))
                Bc.clamp_(min=config.clamp_min)

            p = B_blocks[0].shape[1]
            n_samples = H.shape[1]
            S1 = torch.zeros(p, n_samples, device=V.device, dtype=dtype)
            S2 = torch.zeros(p, p, device=V.device, dtype=dtype)
            for Xc, Wc, Bc in zip(X_blocks, W_blocks, B_blocks):
                S1 += Bc.t() @ (Wc.t() @ Xc)
                S2 += Bc.t() @ (Wc.t() @ Wc) @ Bc

            for inner in range(1, inner_iters + 1):
                HHt = H @ H.t()
                V.mul_((S1 @ H.t()) / (S2 @ V @ HHt + config.lam_V + eps))
                V.clamp_(min=config.clamp_min)

                G = V.t() @ S2 @ V
                H.mul_((V.t() @ S1) / (G @ H + config.lam_H + eps))
                H.clamp_(min=config.clamp_min)

                if inner % config.log_every_inner == 0:
                    loss_inner = float(frob_loss(X_blocks, W_blocks, B_blocks, V, H, config).cpu())
                    print(f"  [{phase_name}][Inner {inner}/{inner_iters}] loss={loss_inner:.6e}")

            loss_val = float(frob_loss(X_blocks, W_blocks, B_blocks, V, H, config).cpu())

        if not np.isfinite(loss_val):
            raise FloatingPointError(
                f"[{phase_name}] loss became non-finite at outer {it}: {loss_val}"
            )

        elapsed = time.time() - t0
        denom = abs(prev_loss) if prev_loss != 0.0 else 1.0
        rel_change = (prev_loss - loss_val) / denom
        print(f"[{phase_name}][Outer {it:04d}] loss={loss_val:.6e}  rel={rel_change:+.3e}  t={elapsed:.3f}s")

        if loss_log_path is not None:
            _append_loss_log(loss_log_path, phase_name, it, loss_val, rel_change, elapsed)

        last_loss = loss_val

        if not early_stop:
            if it >= outer_iters:
                break
        else:
            if it > min_outer:
                if rel_change < config.early_stop_rel_tol:
                    n_stagnant += 1
                else:
                    n_stagnant = 0
                if n_stagnant >= config.early_stop_patience:
                    print(f"[{phase_name}] Early stopping at outer {it}: "
                          f"{n_stagnant} stagnant outers (tol={config.early_stop_rel_tol:.1e}).")
                    break
            if max_outer is not None and it >= max_outer:
                print(f"[{phase_name}] Reached max_outer={max_outer}, stopping.")
                break

        prev_loss = loss_val

    if final_ckpt_path is not None and block_data is not None:
        t_info = dict(training_info) if training_info else {}
        t_info.update({
            "phase_name": phase_name,
            "final_outer_iter": it,
            "final_loss": last_loss,
            "inner_iters": inner_iters,
            "early_stop": early_stop,
            "min_outer": min_outer if early_stop else None,
            "max_outer": max_outer if early_stop else None,
            "early_stop_patience": config.early_stop_patience if early_stop else None,
            "early_stop_rel_tol": config.early_stop_rel_tol if early_stop else None,
        })
        save_checkpoint(
            final_ckpt_path, it, W_blocks, B_blocks, V, H,
            last_loss, block_data.chr_names, block_data, config, t_info,
        )

    return W_blocks, B_blocks, V, H, last_loss, it


def train_rega(
    block_data: BlockData,
    output_dir: Path,
    config: TrainingConfig | None = None,
    device: str | None = None,
) -> Path:
    """Run two-phase multiplicative-update training and save final.pt.

    Phase 1: initialization with fixed outer iterations.
    Phase 2: refinement with early stopping.

    Returns path to the final.pt checkpoint. Caller must ensure output_dir exists.
    """
    if config is None:
        config = TrainingConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[train_rega] device={device}, K={config.K}, dtype={config.dtype}")

    X_blocks = [torch.tensor(x, device=device, dtype=config.dtype) for x in block_data.X_blocks]
    W_blocks = [torch.tensor(w, device=device, dtype=config.dtype) for w in block_data.W_blocks]
    B_blocks = [torch.tensor(b, device=device, dtype=config.dtype) for b in block_data.B_blocks]

    p = B_blocks[0].shape[1]
    n = X_blocks[0].shape[1]
    gen_V = torch.Generator(device=device).manual_seed(config.V_seed)
    V = torch.rand(p, config.K, generator=gen_V, device=device, dtype=config.dtype)
    gen_H = torch.Generator(device=device).manual_seed(config.H_seed)
    H = torch.rand(config.K, n, generator=gen_H, device=device, dtype=config.dtype)

    print(f"\n[Phase 1] inner={config.phase1_inner_iters}, outer={config.phase1_outer_iters}")
    W_blocks, B_blocks, V, H, phase1_loss, phase1_iters = mu_train(
        X_blocks, W_blocks, B_blocks, V, H,
        config=config,
        inner_iters=config.phase1_inner_iters,
        outer_iters=config.phase1_outer_iters,
        device=device,
        phase_name="phase1",
        loss_log_path=output_dir / "phase1_loss_log.csv",
        early_stop=False,
    )
    print(f"[Phase 1 done] loss={phase1_loss:.6e} ({phase1_iters} outer)")

    print(f"\n[Phase 2] inner={config.phase2_inner_iters}, "
          f"min={config.phase2_min_outer}, max={config.phase2_max_outer}")
    final_ckpt = output_dir / "final.pt"
    W_blocks, B_blocks, V, H, final_loss, phase2_iters = mu_train(
        X_blocks, W_blocks, B_blocks, V, H,
        config=config,
        inner_iters=config.phase2_inner_iters,
        outer_iters=config.phase2_max_outer,
        device=device,
        phase_name="phase2",
        loss_log_path=output_dir / "phase2_loss_log.csv",
        final_ckpt_path=final_ckpt,
        early_stop=True,
        min_outer=config.phase2_min_outer,
        max_outer=config.phase2_max_outer,
        block_data=block_data,
        training_info={
            "lam_W": config.lam_W, "lam_B": config.lam_B,
            "lam_V": config.lam_V, "lam_H": config.lam_H,
            "phase1_inner_iters": config.phase1_inner_iters,
            "phase1_outer_iters": config.phase1_outer_iters,
            "phase1_final_loss": phase1_loss,
            "phase1_iters": phase1_iters,
        },
    )
    print(f"[train_rega done] Phase2 loss={final_loss:.6e} ({phase2_iters} outer)")
    return final_ckpt
