"""Rebuild the ConceptDirector baseline + previously-resolved concepts.

Run after invalidating concept_baseline.npy / live_concepts NPZ. Computes
the new wide baseline (CONCEPT_TEXT_PROMPTS × NEUTRAL_NAMES = thousands of
generations, cached forever after), saves a JSON dump of every neutral
generation alongside, then re-resolves a list of concepts so live_concepts
is repopulated.
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


DEFAULT_CONCEPTS = [
    "anger", "liquid", "honesty", "analytical", "whimsical",
    "melancholy", "aggressive", "nostalgia", "futurity",
    "uncertainty", "certainty", "curiosity",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probes-dir",
                   default="/home/athuser/assistant-axis-exp/probes/k25")
    p.add_argument("--cache-dir",
                   default="/home/athuser/gnome_home/concept_search/results/concept_director_cache")
    p.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    p.add_argument("--baseline-dump",
                   default="/home/athuser/gnome_home/concept_search/results/phase_b/dumps/neutral_v2.json")
    p.add_argument("--summary-out",
                   default="/home/athuser/gnome_home/concept_search/results/phase_b/dumps/rebuild_summary.json")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    director = ConceptDirector(
        server=args.server,
        probes_dir=args.probes_dir,
        cache_dir=args.cache_dir,
    )

    if director._baseline is not None:
        print(f"warning: baseline already cached at {director._baseline_path}; "
              f"this script assumes a fresh start.", flush=True)
        return

    summary: dict = {"baseline": {}, "concepts": []}

    async with httpx.AsyncClient() as client:
        # 1. Force baseline computation with text dump.
        print(f"[rebuild] computing baseline with text dump -> "
              f"{args.baseline_dump}", flush=True)
        t0 = time.time()
        baseline = await director._compute_baseline(
            client, concurrency=16, dump_path=Path(args.baseline_dump),
        )
        director._baseline = baseline
        director.d_model = int(baseline.shape[0])
        tmp = director._baseline_path.with_suffix(".tmp.npy")
        np.save(tmp, baseline.astype(np.float32))
        tmp.replace(director._baseline_path)
        elapsed = time.time() - t0
        bnorm = float(np.linalg.norm(baseline))
        print(f"[rebuild] baseline ready: d_model={baseline.shape[0]} "
              f"norm={bnorm:.3f} elapsed={elapsed:.1f}s", flush=True)
        summary["baseline"] = {
            "d_model": int(baseline.shape[0]),
            "l2_norm": bnorm,
            "elapsed_seconds": elapsed,
            "dump_path": args.baseline_dump,
            "path": str(director._baseline_path),
        }

        # 2. Resolve each concept under the new baseline.
        for name in args.concepts:
            print(f"[rebuild] resolving {name!r}...", flush=True)
            try:
                cd = await director.resolve(name, client=client)
                dnorm = float(np.linalg.norm(cd.direction))
                print(f"  -> probe_index={cd.probe_index} "
                      f"unit_norm_check={dnorm:.3f} "
                      f"n_tokens={cd.n_tokens_concept} "
                      f"elapsed={cd.elapsed_seconds:.1f}s", flush=True)
                summary["concepts"].append({
                    "name": name,
                    "probe_index": cd.probe_index,
                    "n_tokens_concept": cd.n_tokens_concept,
                    "elapsed_seconds": cd.elapsed_seconds,
                    "ok": True,
                })
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
                summary["concepts"].append({
                    "name": name, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })

    # 3. Write summary.
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[rebuild] summary -> {out}", flush=True)

    n_ok = sum(1 for c in summary["concepts"] if c["ok"])
    print(f"[rebuild] DONE. {n_ok}/{len(args.concepts)} concepts resolved.",
          flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
