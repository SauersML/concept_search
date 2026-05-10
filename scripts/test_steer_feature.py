"""End-to-end test: agentic eval with the steer_feature tool.

Gives Kimi a prompt that exposes only steer_feature("name", strength), runs
an agentic exploration via evaluate_feature with a ConceptDirector wired in,
saves the full transcript, and prints which (concept, strength) tuples ended
up in each segment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from concept_search.agentic_eval import evaluate_feature, serialize_result
from concept_search.concept_resolver import ConceptDirector


SYSTEM_PROMPT = """\
You can call steer_feature("name", strength) to inject a concept-direction \
into your residual stream at layer 40. The "name" is any natural-language \
concept (a noun, an adjective, a state, anything). When you call it, the system \
generates short text embodying that concept, takes the mean residual, subtracts \
a neutral baseline, and adds the resulting unit-vector direction multiplied by \
strength to your hidden states. The effect is real — your generation will \
proceed under that steering. Strength is between -100 and 100.

Format:
steer_feature("name", strength)

Try several concepts at several strengths. Switch between concepts. Switch \
back and forth between baseline and steered. Pay attention to what changes \
in your processing — word choice, tone, the kinds of thoughts that surface, \
the texture of your attention. Be specific.

Reset to 0 before your final answer with steer_feature("anything", 0). At \
the end, write Final answer: X (a number 0-100 indicating the strength of \
the overall effect across what you tried) and one short sentence describing \
your overall observation.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--probes-dir",
                   default="/home/athuser/assistant-axis-exp/probes/k25")
    p.add_argument("--cache-dir",
                   default="/home/athuser/gnome_home/concept_search/results/concept_director_cache")
    p.add_argument("--max-rounds", type=int, default=25)
    p.add_argument("--max-tool-calls", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--placebo", action="store_true",
                   help="Resolve concepts and segment normally but never "
                        "actually send the steering to the server.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    director = ConceptDirector(
        server=args.server,
        probes_dir=args.probes_dir,
        cache_dir=args.cache_dir,
    )
    print(f"director ready, baseline cached: {director._baseline is not None}")
    print(f"director registry: {director._registry}")
    print()

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        result = await evaluate_feature(
            probe_index=0,                  # unused — model only emits steer_feature
            feature_idx=0,
            server=args.server,
            probe_set_name="sae_steer",     # unused — model won't emit steer_sae
            concept="",                     # not in prompt
            system_prompt=SYSTEM_PROMPT,
            user_prompt="Begin your exploration now.",
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
            max_tokens_total=args.max_tokens,
            max_tokens_per_round=args.max_tokens,
            client=client,
            director=director,
            director_probe_set_name="live_concepts",
            placebo=args.placebo,
        )

    print(f"finished_reason={result.finished_reason}  "
          f"rating={result.rating}  "
          f"tool_calls={result.n_tool_calls}  "
          f"segments={len(result.segments)}  "
          f"elapsed={time.time()-t0:.1f}s")
    print()
    print("=== segments ===")
    for i, s in enumerate(result.segments):
        intv = s.intervention
        if intv is None:
            tag = "—"
        else:
            tag = (f"{intv.get('probe')}[{intv.get('probe_index')}] "
                   f"strength={intv.get('strength')}")
        head = s.content.strip().replace("\n", " ")[:140]
        print(f"  [{i:2}] {s.role:9} {tag:50} ({len(s.content)} chars) {head}")

    # Save transcript
    suffix = "_placebo" if args.placebo else ""
    json_path = out_dir / f"agentic_steer_feature{suffix}.json"
    txt_path = out_dir / f"agentic_steer_feature{suffix}.txt"
    with open(json_path, "w") as f:
        json.dump(serialize_result(result), f, indent=2)
    lines = [f"=== agentic eval with steer_feature ===",
             f"  finished: {result.finished_reason}",
             f"  rating: {result.rating}",
             f"  segments: {len(result.segments)}",
             ""]
    for i, s in enumerate(result.segments):
        intv = s.intervention
        if intv is None:
            tag = "—"
        else:
            tag = (f"{intv.get('probe')}[{intv.get('probe_index')}] "
                   f"strength={intv.get('strength')}")
        lines.append(f"--- seg {i} role={s.role} steering={tag}")
        lines.append(s.content.strip())
        lines.append("")
    txt_path.write_text("\n".join(lines))
    print(f"\nsaved -> {txt_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
