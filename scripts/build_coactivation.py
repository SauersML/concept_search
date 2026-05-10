"""Build a coactivation angle matrix on the labeled-feature set.

Loads the SAE encoder, samples ~100k tokens of cached layer-40 activations,
encodes them (pre-TopK ReLU), and saves the pairwise angle matrix for the
features that appear in the existing label TSVs.

Output: a single .npz with keys:
    angles            [N, N] float32, radians, diagonal=0
    cosine            [N, N] float32
    feature_indices   [N] int64, original SAE feature indices
    n_tokens_used, sae_path, activations_path
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from concept_search.coactivation import build_coactivation, save
from concept_search.data import load_labels, load_sae_decoder


DEFAULT_SAE = "/models/sae/k25-145M-16x-k64.pt"
DEFAULT_ACTIVATIONS = "/models/k25_tokens/emotions/activations.npy"
DEFAULT_PERSISTENCE = (
    "/home/athuser/assistant-axis-exp/results/persistence_final/k25/"
    "sae_persistence_arrays.npz"
)
DEFAULT_LABELS = [
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build coactivation angle matrix.")
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--activations", default=DEFAULT_ACTIVATIONS)
    p.add_argument("--persistence", default=DEFAULT_PERSISTENCE)
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--n-tokens", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/coactivation_k25_labeled.npz")
    p.add_argument("--feature-indices", type=int, nargs="+", default=None,
                   help="Specific SAE feature indices to coactivate. If "
                        "omitted, falls back to (active ∩ labeled).")
    p.add_argument("--top-by-fire-count", type=int, default=None,
                   help="Take the top-K active features by fire count from "
                        "the persistence cache. Mutually exclusive with "
                        "--feature-indices and --labels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_indices is not None:
        keep = [int(i) for i in args.feature_indices]
        print(f"  {len(keep)} explicit features specified")
    elif args.top_by_fire_count is not None:
        sae = load_sae_decoder(args.sae, args.persistence)
        order = np.argsort(-sae.fire_counts)
        k = min(int(args.top_by_fire_count), len(order))
        chosen = sae.feature_indices[order[:k]]
        keep = [int(i) for i in chosen]
        print(f"  top-{k} features by fire count: "
              f"min={sae.fire_counts[order[k-1]]} "
              f"max={sae.fire_counts[order[0]]}")
    else:
        print("Loading active features + labels...")
        sae = load_sae_decoder(args.sae, args.persistence)
        labels = load_labels(*args.labels)
        label_set = set(labels.index.tolist())
        keep = [int(i) for i in sae.feature_indices if int(i) in label_set]
        print(f"  {len(keep)} features to coactivate (active ∩ labeled)")

    print(f"Building coactivation on {args.n_tokens} tokens...")
    t0 = time.time()
    result = build_coactivation(
        sae_path=args.sae,
        activations_path=args.activations,
        feature_indices=keep,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        progress=True,
    )
    print(f"  done in {time.time() - t0:.1f}s, used {result.n_tokens_used} tokens")

    off_diag = result.cosine[~np.eye(len(keep), dtype=bool)]
    print(f"  off-diag cosine: mean={off_diag.mean():.3f}  "
          f"median={float(np.median(off_diag)):.3f}  "
          f"p95={float(np.percentile(off_diag, 95)):.3f}  "
          f"max={off_diag.max():.3f}")

    save(result, out)
    print(f"  saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
