"""Build per-template baseline residual means from an existing neutral text dump.

Reads neutral_v2.json (produced by the rebuild script's text dump), encodes
every text via /v1/encode, groups results by template, averages, and writes
shape-[n_templates, d_model] array to disk. Used by the symmetric-drop
resolve path: when a concept refuses on template t, that t's baseline mean
is dropped along with the concept's t-gen, so subtraction stays clean.

No new generation; just encoding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from concept_search.concept_resolver import (
    CONCEPT_TEXT_PROMPTS,
    ConceptDirector,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probes-dir",
                   default="/home/athuser/assistant-axis-exp/probes/k25")
    p.add_argument("--cache-dir",
                   default="/home/athuser/gnome_home/concept_search/results/concept_director_cache")
    p.add_argument("--neutral-json",
                   default="/home/athuser/gnome_home/concept_search/results/phase_b/dumps/neutral_v2.json")
    p.add_argument("--concurrency", type=int, default=16)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    director = ConceptDirector(
        server=args.server,
        probes_dir=args.probes_dir,
        cache_dir=args.cache_dir,
    )

    rows = json.loads(Path(args.neutral_json).read_text())
    print(f"loaded {len(rows)} entries from {args.neutral_json}", flush=True)

    template_to_idx = {t: i for i, t in enumerate(CONCEPT_TEXT_PROMPTS)}
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    completed = 0

    async def encode_one(row: dict) -> tuple[int, np.ndarray | None]:
        nonlocal completed
        text = row.get("text") or ""
        tmpl = row.get("template")
        if not text or tmpl not in template_to_idx:
            return -1, None
        async with sem:
            try:
                r = await director._encode(client, text)
            except Exception as e:
                print(f"  encode fail: {type(e).__name__}: {e}", flush=True)
                return -1, None
        completed += 1
        if completed % 100 == 0:
            print(f"  encoded {completed}/{len(rows)}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
        if r.shape[0] == 0:
            return -1, None
        return template_to_idx[tmpl], r.mean(axis=0).astype(np.float32)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[encode_one(r) for r in rows])

    by_template: dict[int, list[np.ndarray]] = defaultdict(list)
    for tmpl_idx, mean in results:
        if tmpl_idx >= 0 and mean is not None:
            by_template[tmpl_idx].append(mean)

    n_templates = len(CONCEPT_TEXT_PROMPTS)
    d_model = next(iter(by_template.values()))[0].shape[0]
    per_template = np.zeros((n_templates, d_model), dtype=np.float32)
    counts = []
    for i in range(n_templates):
        chunk = by_template.get(i, [])
        counts.append(len(chunk))
        if not chunk:
            raise RuntimeError(f"no encodings for template {i}: "
                               f"{CONCEPT_TEXT_PROMPTS[i]!r}")
        per_template[i] = np.stack(chunk).mean(axis=0)

    out_path = Path(args.cache_dir) / "concept_baseline_per_template.npy"
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, per_template)
    tmp.replace(out_path)

    print(f"\nsaved per-template baseline -> {out_path}", flush=True)
    print(f"shape={per_template.shape}  d_model={d_model}", flush=True)
    print(f"per-template encoded counts: {counts}", flush=True)
    for i, c in enumerate(counts):
        norm = float(np.linalg.norm(per_template[i]))
        print(f"  [{i}] {CONCEPT_TEXT_PROMPTS[i]!r:60s} n={c:3d} norm={norm:.3f}",
              flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
