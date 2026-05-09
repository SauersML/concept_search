"""Load SAE decoder columns, persistence-based active-feature filter, and labeled eval TSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class SAEData:
    """Decoder columns restricted to active features.

    `decoder` rows are the unit-norm decoder column for each active feature
    (shape: [n_active, d_model]). `feature_indices` maps row → original SAE
    feature index (in the full 0..n_features-1 space). `fire_counts` carries
    the per-active-feature fire count from the persistence cache.
    """

    decoder: torch.Tensor          # [n_active, d_model], float32, unit-norm rows
    feature_indices: np.ndarray    # [n_active], int64
    fire_counts: np.ndarray        # [n_active], int64
    d_model: int
    layer: int


def load_sae_decoder(
    sae_path: str | Path,
    persistence_path: str | Path,
    layer: int = 40,
    min_fire_count: int = 1,
) -> SAEData:
    """Load decoder columns for active features only.

    The full SAE checkpoint stores W_dec ∈ R^{d_model × n_features}; we read it,
    pick out columns at indices listed in the persistence cache's
    `active_features`, and return them as rows of a torch.Tensor.
    """
    persistence = np.load(persistence_path)
    active = persistence["active_features"].astype(np.int64)
    fire_counts = persistence["fire_counts"].astype(np.int64)

    mask = fire_counts >= min_fire_count
    active = active[mask]
    fire_counts = fire_counts[mask]

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    W_dec = ckpt["state_dict"]["W_dec"].float()   # [d_model, n_features]
    d_model = W_dec.shape[0]
    cols = W_dec[:, active].T.contiguous().clone()  # [n_active, d_model]
    del ckpt, W_dec

    norms = cols.norm(dim=1, keepdim=True).clamp_min(1e-10)
    cols = cols / norms

    return SAEData(
        decoder=cols,
        feature_indices=active,
        fire_counts=fire_counts,
        d_model=d_model,
        layer=layer,
    )


def load_labels(*tsv_paths: str | Path) -> pd.DataFrame:
    """Load labeled-feature TSVs into a single DataFrame.

    Expects columns: feature_idx, score, n_tool_calls, response_len, response.
    Returns a DataFrame indexed by feature_idx with `score` as float.
    """
    frames = []
    for p in tsv_paths:
        df = pd.read_csv(p, sep="\t")
        frames.append(df[["feature_idx", "score"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["score"])
    out["feature_idx"] = out["feature_idx"].astype(np.int64)
    out["score"] = out["score"].astype(np.float32)
    out = out.drop_duplicates(subset=["feature_idx"], keep="first")
    return out.set_index("feature_idx")


def restrict_to_labeled(sae: SAEData, labels: pd.DataFrame) -> tuple[SAEData, np.ndarray]:
    """Subset SAEData to features that appear in the labels DataFrame.

    Returns (subset_sae, scores) where `scores[i]` is the label for row i of
    `subset_sae.decoder`.
    """
    label_set = set(labels.index.tolist())
    keep = np.array([i for i, idx in enumerate(sae.feature_indices)
                     if int(idx) in label_set])
    if keep.size == 0:
        raise ValueError("No labeled features intersect active features.")
    sub = SAEData(
        decoder=sae.decoder[keep].clone(),
        feature_indices=sae.feature_indices[keep],
        fire_counts=sae.fire_counts[keep],
        d_model=sae.d_model,
        layer=sae.layer,
    )
    scores = np.array([float(labels.loc[int(idx), "score"])
                       for idx in sub.feature_indices], dtype=np.float32)
    return sub, scores
