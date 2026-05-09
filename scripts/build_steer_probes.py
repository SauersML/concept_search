"""Build the sae_steer probes NPZ that the probe server loads at startup.

The probe server scans a probes-dir for files matching `*_probes_layer{N}.npz`.
Each file contains:
    directions   [n_features, d_model]  unit-norm decoder columns
    labels       [n_features]            string labels (we use "feat_{idx}")
    description  ()                      free text

This script reads `--feature-indices` (or a default set of representative
labeled features), pulls the corresponding decoder columns out of the SAE
checkpoint, normalizes them, and writes the NPZ where the probe server expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from concept_search.data import load_labels


DEFAULT_SAE = "/models/sae/k25-145M-16x-k64.pt"
DEFAULT_LABELS = [
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
]
DEFAULT_OUT = "/home/athuser/assistant-axis-exp/probes/k25/sae_steer_probes_layer40.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--feature-indices", type=int, nargs="+", default=None,
                   help="Specific feature indices to include. If omitted, uses "
                        "5 representative features from the labeled set.")
    p.add_argument("--n-default-features", type=int, default=5)
    p.add_argument("--layer", type=int, default=40)
    p.add_argument("--output", default=DEFAULT_OUT)
    return p.parse_args()


def pick_default_features(labels_paths: list[str], n: int) -> list[int]:
    """Pick n features spanning the rating distribution: 1 highest, 1 lowest,
    rest evenly spaced by rank."""
    labels = load_labels(*labels_paths)
    sorted_df = labels.sort_values("score", ascending=False)
    n_total = len(sorted_df)
    if n >= n_total:
        return [int(idx) for idx in sorted_df.index]
    positions = np.linspace(0, n_total - 1, n).astype(int)
    return [int(sorted_df.index[p]) for p in positions]


def main() -> None:
    args = parse_args()

    if args.feature_indices is not None:
        feats = list(args.feature_indices)
    else:
        feats = pick_default_features(args.labels, args.n_default_features)

    print(f"Building probes for {len(feats)} features:")
    labels_df = None
    try:
        labels_df = load_labels(*args.labels)
    except FileNotFoundError as e:
        print(f"  (label TSVs not present here, skipping rating annotations: {e})")
    for f in feats:
        score = (labels_df.loc[f, "score"]
                 if labels_df is not None and f in labels_df.index else None)
        print(f"  feat_{f}  rating={score}")

    print(f"\nLoading SAE checkpoint: {args.sae}")
    ckpt = torch.load(args.sae, map_location="cpu", weights_only=False)
    W_dec = ckpt["state_dict"]["W_dec"].float().numpy()  # [d_model, n_features]

    cols = W_dec[:, np.array(feats, dtype=np.int64)].T.copy()  # [n_features, d_model]
    norms = np.linalg.norm(cols, axis=1, keepdims=True)
    cols = cols / (norms + 1e-12)
    del ckpt, W_dec

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        directions=cols.astype(np.float32),
        labels=np.array([f"feat_{f}" for f in feats]),
        description=np.array(f"sae_steer probes for live agentic-eval test "
                             f"(layer {args.layer})"),
    )
    print(f"\nSaved {out_path} "
          f"(directions shape {cols.shape}, layer {args.layer})")


if __name__ == "__main__":
    main()
