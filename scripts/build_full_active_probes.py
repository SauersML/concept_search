"""Build a probe NPZ containing the SAE decoder columns for ALL active features.

Loaded by the probe server as a separate probe set (e.g. "all_active") so the
agentic-eval orchestrator can reference any active feature by index in
intervention lists, without rebuilding the existing sae_steer probe set.

This is the one-time cost for supporting steer_feature("name", strength) —
the concept resolver picks top-K features per concept name and the orchestrator
sends them as a multi-feature intervention list, which the server expands by
indexing into this probe set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


DEFAULT_SAE = "/models/sae/k25-145M-16x-k64.pt"
DEFAULT_PERSISTENCE = (
    "/home/athuser/assistant-axis-exp/results/persistence_final/k25/"
    "sae_persistence_arrays.npz"
)
DEFAULT_OUT = (
    "/home/athuser/assistant-axis-exp/probes/k25/all_active_probes_layer40.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--persistence", default=DEFAULT_PERSISTENCE)
    p.add_argument("--min-fire-count", type=int, default=1)
    p.add_argument("--output", default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading persistence cache: {args.persistence}")
    persistence = np.load(args.persistence)
    active = persistence["active_features"].astype(np.int64)
    fire_counts = persistence["fire_counts"].astype(np.int64)
    mask = fire_counts >= args.min_fire_count
    active = active[mask]
    print(f"  {len(active)} active features (fire_count >= {args.min_fire_count})")

    print(f"Loading SAE checkpoint: {args.sae}")
    ckpt = torch.load(args.sae, map_location="cpu", weights_only=False)
    W_dec = ckpt["state_dict"]["W_dec"].float()    # [d_model, n_features]
    cols = W_dec[:, active].T.contiguous().clone()  # [n_active, d_model]
    norms = cols.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cols = cols / norms
    del ckpt, W_dec

    labels = np.array([f"feat_{int(i)}" for i in active])
    print(f"Saving {out} (directions {tuple(cols.shape)}, "
          f"~{cols.numel() * 4 / 1e6:.0f} MB)")
    np.savez(
        out,
        directions=cols.numpy().astype(np.float32),
        labels=labels,
        description=np.array(
            f"All active SAE features for steer_feature() concept resolution "
            f"(n={len(active)}, layer 40)"
        ),
    )
    print("done")


if __name__ == "__main__":
    main()
