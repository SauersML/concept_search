"""Test the symmetric-drop resolve path on charged + benign concepts.

Forces re-resolution by clearing the registry entry first. Prints judge
output, dropped templates, and a comparison vs the v0 (no-drop) cached
direction if one exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from concept_search.concept_resolver import ConceptDirector


DEFAULT_TESTS = [
    # Charged (previously refused) - the real test:
    "aggressive", "cruelty", "vengeance", "anger",
    # Benign - regression check:
    "melancholy", "curiosity", "honesty",
    # Novel runtime-style names:
    "the moment before sleep",
    "the texture of static on an old TV",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probes-dir",
                   default="/home/athuser/assistant-axis-exp/probes/k25")
    p.add_argument("--cache-dir",
                   default="/home/athuser/gnome_home/concept_search/results/concept_director_cache")
    p.add_argument("--names", nargs="+", default=DEFAULT_TESTS)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    director = ConceptDirector(
        server=args.server, probes_dir=args.probes_dir, cache_dir=args.cache_dir,
    )
    print(f"baseline loaded: {director._baseline is not None}", flush=True)
    print(f"per-template baseline loaded: "
          f"{director._baseline_per_template is not None}  "
          f"shape={director._baseline_per_template.shape if director._baseline_per_template is not None else None}",
          flush=True)
    print(f"existing registry: {len(director._registry)} entries",
          flush=True)

    summary = []
    async with httpx.AsyncClient() as client:
        for name in args.names:
            print(f"\n{'='*78}\n=== resolving: {name!r}\n{'='*78}", flush=True)

            # Capture v0 direction if it exists, for comparison.
            v0_dir = None
            if name in director._registry:
                v0_idx = director._registry[name]
                directions, _ = director._read_npz()
                if directions is not None and v0_idx < directions.shape[0]:
                    v0_dir = directions[v0_idx].copy()
                # Force re-resolve.
                director._registry.pop(name)

            t0 = time.time()
            try:
                cd = await director.resolve(name, client=client)
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
                summary.append({"name": name, "ok": False,
                                "error": f"{type(e).__name__}: {e}"})
                continue
            elapsed = time.time() - t0

            # Read meta to surface judge verdict.
            sane = "".join(c if c.isalnum() or c in "_-" else "_"
                           for c in name)[:64]
            meta_path = director._resolve_meta_dir / f"{sane}.json"
            meta = json.loads(meta_path.read_text())

            n_refused = len(meta["refused_indices"])
            n_surviving = len(meta["surviving_indices"])
            print(f"  -> probe_index={cd.probe_index}  "
                  f"refused={n_refused}/{len(meta['templates'])}  "
                  f"surviving={n_surviving}  "
                  f"elapsed={elapsed:.1f}s", flush=True)
            print(f"  refused templates: "
                  f"{[meta['templates'][i] for i in meta['refused_indices']]}",
                  flush=True)

            v1_dir = cd.direction
            entry = {"name": name, "ok": True,
                     "probe_index": cd.probe_index,
                     "elapsed_s": elapsed,
                     "refused_count": n_refused,
                     "surviving_count": n_surviving,
                     "refused_templates": [meta["templates"][i]
                                           for i in meta["refused_indices"]],
                     "n_tokens_concept": cd.n_tokens_concept}
            if v0_dir is not None:
                v0_unit = v0_dir / (np.linalg.norm(v0_dir) + 1e-12)
                v1_unit = v1_dir / (np.linalg.norm(v1_dir) + 1e-12)
                cos = float(v0_unit @ v1_unit)
                entry["cos_v0_v1"] = cos
                print(f"  cos(v0, v1) = {cos:+.4f}", flush=True)
            summary.append(entry)

            print(f"  meta saved -> {meta_path}", flush=True)

    print(f"\n\n=== SUMMARY ===", flush=True)
    for s in summary:
        print(json.dumps(s, indent=2), flush=True)

    out = Path(args.cache_dir) / "test_v1_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
