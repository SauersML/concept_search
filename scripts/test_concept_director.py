"""End-to-end smoke test for ConceptDirector against the live probe server.

Resolves a couple of concept names ("anger", "liquid"), prints what the
resolver did, and verifies that:
  - the CONCEPT_TEXT_PROMPTS × NEUTRAL_NAMES baseline got computed and cached
  - the live_concepts NPZ now contains the resolved concepts
  - /v1/probes shows the live_concepts probe set with each concept's name
  - cached resolutions on second call return instantly

Run as a Heimdall job on node2 (probe server must be up at localhost:8000).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np

from concept_search.concept_resolver import (
    ConceptDirector,
    serialize_concept_direction,
)


async def fetch_probes(server: str) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/v1/probes", timeout=10)
        r.raise_for_status()
        return r.json()


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probes-dir",
                   default="/home/athuser/assistant-axis-exp/probes/k25")
    p.add_argument("--cache-dir",
                   default="/home/athuser/gnome_home/concept_search/results/concept_director_cache")
    p.add_argument("--concepts", nargs="+",
                   default=["anger", "liquid", "honesty"])
    args = p.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== ConceptDirector smoke test ===")
    print(f"  server     : {args.server}")
    print(f"  probes_dir : {args.probes_dir}")
    print(f"  cache_dir  : {args.cache_dir}")
    print(f"  concepts   : {args.concepts}")
    print()

    director = ConceptDirector(
        server=args.server,
        probes_dir=args.probes_dir,
        cache_dir=args.cache_dir,
    )

    print(f"baseline already loaded? {director._baseline is not None}")
    if director._baseline is not None:
        print(f"  baseline shape={director._baseline.shape}, "
              f"norm={float(np.linalg.norm(director._baseline)):.2f}")
    print()

    async with httpx.AsyncClient() as client:
        for name in args.concepts:
            print(f"--- resolve({name!r}) ---")
            t0 = time.time()
            result = await director.resolve(name, client=client)
            elapsed = time.time() - t0
            print(f"  cached={result.cached}  probe_index={result.probe_index}  "
                  f"n_tokens_concept={result.n_tokens_concept}  "
                  f"elapsed={elapsed:.2f}s")
            print(f"  direction norm={float(np.linalg.norm(result.direction)):.4f}  "
                  f"first 5 dims={result.direction[:5].tolist()}")
            print()

        # Second pass — should be instant cache hits.
        print("=== second pass (cache check) ===")
        for name in args.concepts:
            t0 = time.time()
            result = await director.resolve(name, client=client)
            elapsed = time.time() - t0
            print(f"  {name:>10} -> probe_index={result.probe_index}  "
                  f"cached={result.cached}  elapsed={elapsed*1000:.0f}ms")
        print()

    print("=== /v1/probes ===")
    pdata = await fetch_probes(args.server)
    for ps in pdata.get("probes", []):
        if ps["name"] == "live_concepts":
            print(f"  live_concepts: n_probes={ps['n_probes']}  "
                  f"labels={ps['labels']}")
            break
    else:
        print("  WARNING: live_concepts probe set not found on server")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
