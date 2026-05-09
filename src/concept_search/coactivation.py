"""Coactivation kernel: pairwise activation similarity from a real corpus.

The angular-RBF-on-decoder-direction kernel was the wrong prior for this SAE:
TopK training pushes decoder columns near-orthogonal regardless of concept
similarity, so geometric proximity carries little signal about concept
relatedness. Coactivation measures it directly: features that get high
activations on the same tokens are concept-related, by definition.

We use *pre-TopK ReLU* activations rather than post-TopK, because TopK
competition systematically suppresses the joint firing of similar features
(they fight for the same top-k slots on the same tokens). Pre-TopK ReLU is
what each feature's encoder *wants* to do at every token, before competition.

Output: an angle matrix `theta[i, j] = arccos(cos_ij)` in radians, for use
with the existing AngularRBFKernel. Diagonal is exactly 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CoactivationResult:
    angles: np.ndarray              # [N, N], radians, float32
    cosine: np.ndarray              # [N, N], float32
    feature_indices: np.ndarray     # [N], int64 — original SAE feature indices
    n_tokens_used: int
    sae_path: str
    activations_path: str


def build_coactivation(
    sae_path: str | Path,
    activations_path: str | Path,
    feature_indices: np.ndarray,
    n_tokens: int = 100_000,
    batch_size: int = 4096,
    seed: int = 0,
    progress: bool = True,
) -> CoactivationResult:
    """Pairwise pre-TopK ReLU activation cosine on a token sample.

    Args:
        sae_path: SAE checkpoint with state_dict[W_enc, b_enc] and a top-level
            'mean' tensor of shape [d_model].
        activations_path: an .npy file of shape [n_total_tokens, d_model] giving
            the residual stream at the SAE's hooked layer. Read via mmap.
        feature_indices: which SAE feature columns to include. Order is preserved
            in the output rows/cols.
        n_tokens: random sample size (without replacement, sorted for sequential
            mmap access).
        batch_size: tokens per encoder forward.
        seed: RNG for token sampling.
    """
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    n_features = len(feature_indices)

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    W_enc = sd["W_enc"][feature_indices].float().contiguous()    # [N, d_model]
    b_enc = sd["b_enc"][feature_indices].float().contiguous()    # [N]
    mean_vec = ckpt["mean"].float()                              # [d_model]
    d_model = mean_vec.shape[0]
    if W_enc.shape[1] != d_model:
        raise ValueError(f"W_enc dim {W_enc.shape[1]} != mean dim {d_model}")
    del ckpt, sd

    acts = np.load(activations_path, mmap_mode="r")
    if acts.ndim != 2 or acts.shape[1] != d_model:
        raise ValueError(f"activations.npy must be [n_tokens, {d_model}], got {acts.shape}")
    n_total = acts.shape[0]
    n_sample = min(n_tokens, n_total)

    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n_total, size=n_sample, replace=False))

    sum_sq = torch.zeros(n_features, dtype=torch.float64)
    sum_pair = torch.zeros((n_features, n_features), dtype=torch.float64)

    n_done = 0
    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        batch_idx = sample_idx[start:end]
        x = torch.from_numpy(acts[batch_idx].astype(np.float32, copy=False))
        x_centered = x - mean_vec                          # [B, d_model]
        pre = torch.relu(x_centered @ W_enc.T + b_enc)     # [B, N], dense
        pre64 = pre.to(torch.float64)
        sum_sq += (pre64 * pre64).sum(dim=0)
        sum_pair += pre64.T @ pre64
        n_done = end
        if progress:
            print(f"  encoded {n_done}/{n_sample} tokens "
                  f"({100 * n_done / n_sample:.1f}%)", flush=True)

    norms = sum_sq.sqrt().clamp_min(1e-12)
    cosine = (sum_pair / (norms.unsqueeze(0) * norms.unsqueeze(1))).clamp(0.0, 1.0)
    cosine_np = cosine.to(torch.float32).numpy()
    np.fill_diagonal(cosine_np, 1.0)

    angles = np.arccos(np.clip(cosine_np, 0.0, 1.0)).astype(np.float32)
    np.fill_diagonal(angles, 0.0)

    return CoactivationResult(
        angles=angles,
        cosine=cosine_np,
        feature_indices=feature_indices,
        n_tokens_used=n_done,
        sae_path=str(sae_path),
        activations_path=str(activations_path),
    )


def save(result: CoactivationResult, path: str | Path) -> None:
    np.savez(
        path,
        angles=result.angles,
        cosine=result.cosine,
        feature_indices=result.feature_indices,
        n_tokens_used=np.int64(result.n_tokens_used),
        sae_path=np.array(result.sae_path),
        activations_path=np.array(result.activations_path),
    )


def load(path: str | Path) -> CoactivationResult:
    d = np.load(path, allow_pickle=False)
    return CoactivationResult(
        angles=d["angles"],
        cosine=d["cosine"],
        feature_indices=d["feature_indices"],
        n_tokens_used=int(d["n_tokens_used"]),
        sae_path=str(d["sae_path"]),
        activations_path=str(d["activations_path"]),
    )
