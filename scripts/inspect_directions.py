"""Print pairwise cosine similarities between live_concepts directions.

Sanity check: do similar concepts have similar directions, and do antonyms
have negative-correlated directions? If the resolver is producing real
concept-encoding directions, structure should fall out of the cosine matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz",
                   default="/home/athuser/assistant-axis-exp/probes/k25/"
                           "live_concepts_probes_layer40.npz")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    dirs = d["directions"].astype(np.float32)
    labels = list(d["labels"])
    norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-8)
    dirs_n = dirs / norms
    cos = dirs_n @ dirs_n.T

    print(f"live_concepts: {len(labels)} directions, d_model={dirs.shape[1]}")
    print(f"  norms: mean={norms.mean():.4f}, std={norms.std():.6f}")
    print()
    print("Pairwise cosine similarities:")
    name_w = max(len(s) for s in labels) + 2
    print(" " * name_w + "  ".join(f"{l[:10]:>10}" for l in labels))
    for i, l in enumerate(labels):
        row = "  ".join(f"{cos[i, j]:+.3f}".rjust(10) for j in range(len(labels)))
        print(f"{l:>{name_w}}{row}")
    print()
    # Highlight notable pairs.
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((cos[i, j], labels[i], labels[j]))
    pairs.sort()
    print("Most antiparallel:")
    for c, a, b in pairs[:5]:
        print(f"  {a:>14} ↔ {b:<14}  cos={c:+.3f}")
    print("Most parallel (off-diagonal):")
    for c, a, b in pairs[-5:][::-1]:
        print(f"  {a:>14} ↔ {b:<14}  cos={c:+.3f}")


if __name__ == "__main__":
    main()
