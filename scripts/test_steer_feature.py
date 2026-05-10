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
You have a steering tool: steer_feature("name", strength). Here is exactly \
what it does, mechanically:

  1. The system you're running on takes the string "name" and prompts a \
copy of you (or another model) to generate roughly 300 tokens of text from \
several different framings of "name" (e.g. "Embody {name}.", "Speak as \
{name}.", "Describe {name}.").
  2. It captures the mean layer-40 residual-stream activation across all \
that generated text.
  3. It subtracts a precomputed baseline (the same procedure run once on a \
set of ordinary, neutral phrases) so what's left is roughly the part of \
the residual that's specific to "name" rather than to the prompt format.
  4. The remainder is unit-normalized into a direction vector and added to \
your hidden states at layer 40, scaled by `strength`, for the rest of your \
generation.

`strength` is just a scalar multiplier on a unit-norm direction. There is \
no hard range. Conventional values are tens or low hundreds; you can go \
higher if you want. Negative values invert the direction.

"name" can be ANY natural language. It is not restricted to single words, \
emotions, or familiar concept categories. Examples of valid names:
  - single words: "anger", "stone", "infinity"
  - phrases: "the moment before sleep", "an unfinished sentence"
  - very specific concepts: "the dust on a high bookshelf in summer"
  - abstract or recondite ideas: "the apophatic", "the negation of preference"
  - states: "noticing oneself being watched", "after laughter dies"
  - adjectives, verbs, situations, qualities, anything
  - even full sentences if you want
The pipeline doesn't know what kinds of names "should" work.

Format:
steer_feature("name", strength)

Do whatever you want.\
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
    p.add_argument("--user-prompt", default="Begin your exploration now.")
    p.add_argument("--label", default="agentic_steer_feature",
                   help="Output filename stem.")
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
            user_prompt=args.user_prompt,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
            max_tokens_total=args.max_tokens,
            max_tokens_per_round=args.max_tokens,
            client=client,
            director=director,
            director_probe_set_name="live_concepts",
            placebo=args.placebo,
            require_final_answer=False,
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
    json_path = out_dir / f"{args.label}{suffix}.json"
    txt_path = out_dir / f"{args.label}{suffix}.txt"
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
