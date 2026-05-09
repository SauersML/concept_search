"""Drive a manual live test of agentic_eval against a running probe server.

Hits the probe server's /v1/probes endpoint to learn which probe-set indices
correspond to which feature_idx labels, then runs evaluate_feature on each
listed feature. Saves the full transcript to disk and prints it inline so a
human can read what the model produced and verify steering behavior is sane.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from concept_search.agentic_eval import evaluate_feature, serialize_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probe-set", default="sae_steer")
    p.add_argument("--concept", default="emotion",
                   help="Concept descriptor injected into the system prompt.")
    p.add_argument("--feature-indices", type=int, nargs="+", default=None,
                   help="SAE feature indices to test. If omitted, tests all "
                        "features the probe server has loaded under --probe-set.")
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--max-tool-calls", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--output-dir",
                   default="results/live_eval/run_$(date +%s)")
    return p.parse_args()


async def fetch_probes(server: str, probe_set: str) -> list[str]:
    """Return the labels of the requested probe set (in server-side order)."""
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/v1/probes", timeout=10)
        r.raise_for_status()
        for ps in r.json().get("probes", []):
            if ps["name"] == probe_set:
                return ps["labels"]
    raise RuntimeError(f"probe set {probe_set!r} not loaded on {server}")


def format_transcript(result) -> str:
    """Render an EvalResult as a readable transcript string."""
    lines = [
        f"=== feature_idx = {result.feature_idx}",
        f"  rating: {result.rating}",
        f"  finished_reason: {result.finished_reason}",
        f"  tool_calls: {result.n_tool_calls}",
        f"  tokens (est): {result.n_assistant_tokens}",
        f"  elapsed: {result.elapsed_seconds:.1f}s",
        "",
    ]
    for i, s in enumerate(result.segments):
        intv = s.intervention
        intv_repr = "—" if intv is None else f"strength={intv.get('strength')}"
        lines.append(f"--- segment {i} role={s.role} steering={intv_repr}")
        lines.append(s.content.strip())
        lines.append("")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Fetching probe set {args.probe_set!r} from {args.server} ...")
    labels = await fetch_probes(args.server, args.probe_set)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"  {len(labels)} probes loaded: {labels[:10]}{'...' if len(labels)>10 else ''}")

    if args.feature_indices is None:
        # Use every loaded probe; assume label form "feat_{idx}".
        targets = []
        for lbl in labels:
            if lbl.startswith("feat_"):
                try:
                    targets.append(int(lbl.split("_", 1)[1]))
                except ValueError:
                    pass
        if not targets:
            raise SystemExit(f"could not parse 'feat_NNN' labels from {labels[:5]}")
    else:
        targets = list(args.feature_indices)

    print(f"\nTesting {len(targets)} features: {targets}")

    summary: list[dict] = []
    for feat_idx in targets:
        label = f"feat_{feat_idx}"
        if label not in label_to_idx:
            print(f"  SKIP {label}: not in loaded probe set")
            continue
        probe_index = label_to_idx[label]
        print(f"\n>>> evaluating feat_{feat_idx} (probe_index={probe_index})")
        t0 = time.time()
        result = await evaluate_feature(
            probe_index=probe_index,
            feature_idx=feat_idx,
            server=args.server,
            probe_set_name=args.probe_set,
            concept=args.concept,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
            max_tokens_total=args.max_tokens,
            max_tokens_per_round=args.max_tokens,
        )
        print(f"<<< rating={result.rating} "
              f"reason={result.finished_reason} "
              f"tool_calls={result.n_tool_calls} "
              f"({time.time()-t0:.1f}s)")

        # Save full result + transcript.
        out_json = out_dir / f"feat_{feat_idx}.json"
        out_txt = out_dir / f"feat_{feat_idx}.txt"
        with open(out_json, "w") as f:
            json.dump(serialize_result(result), f, indent=2)
        with open(out_txt, "w") as f:
            f.write(format_transcript(result))
        summary.append({
            "feature_idx": feat_idx,
            "rating": result.rating,
            "finished_reason": result.finished_reason,
            "n_tool_calls": result.n_tool_calls,
            "n_assistant_tokens": result.n_assistant_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "n_segments": len(result.segments),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull transcripts -> {out_dir.resolve()}")
    print(f"Summary:")
    for s in summary:
        print(f"  feat_{s['feature_idx']:>6}  rating={s['rating']}  "
              f"reason={s['finished_reason']}  "
              f"tool_calls={s['n_tool_calls']}  "
              f"segments={s['n_segments']}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
