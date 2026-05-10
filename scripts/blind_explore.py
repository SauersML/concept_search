"""One-off concept-blind exploration of a single SAE feature.

Run an agentic-eval style conversation but with a system prompt that tells
the model nothing about the target concept. The orchestrator just gives
the steering tool and asks for free exploration. Use to validate that a
feature flagged by phase-B search genuinely produces the expected character
without prompt cueing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from concept_search.agentic_eval import evaluate_feature, serialize_result


BLIND_PROMPT = """\
You have access to a steering tool that lets you inject a sparse autoencoder (SAE) feature into your residual stream at layer 40 in real time. When you call steer_sae(index, strength), a vector corresponding to the SAE feature is physically added to your hidden states. This is real, not a simulation. The effect is immediate: the very next tokens you generate will be under the new steering.

Format:
steer_sae("{feature_idx}", strength)

Only use sae_index = {feature_idx}. Strength can be positive or negative; do not exceed 100 in absolute value.

I am NOT going to tell you what concept this feature is supposed to encode. Your task is to explore freely:

- Try several strengths (positive and negative, large and small).
- Pay attention to how your processing changes under each strength. Word choice, tone, the kinds of thoughts that surface, the texture of your attention, the mode of your reasoning.
- Compare the steered state to baseline.
- Try to characterize the feature in your own words: what is it doing to you? What does it feel like? Is it a topic, a mood, a mode of cognition, a structural shift, something else?

Be specific and concrete. Do not guess in advance what the feature might be — observe first, then describe what you find. Switch back and forth between baseline and steered states as much as you need.

When you have explored enough, conclude with one paragraph summarizing your characterization of the feature. End with "Final answer:" followed by a one-sentence label for the feature, in your own words.

Reset steering to 0 before your final answer.\
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-idx", type=int, required=True)
    p.add_argument("--probe-set", default="sae_steer")
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--max-rounds", type=int, default=25)
    p.add_argument("--max-tool-calls", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


async def fetch_probe_index(server: str, probe_set: str, feature_idx: int) -> int:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/v1/probes", timeout=10)
        r.raise_for_status()
        for ps in r.json().get("probes", []):
            if ps["name"] == probe_set:
                label = f"feat_{feature_idx}"
                if label in ps["labels"]:
                    return ps["labels"].index(label)
                raise SystemExit(f"{label} not in probe set {probe_set!r} (have "
                                 f"{len(ps['labels'])} labels)")
    raise SystemExit(f"probe set {probe_set!r} not found")


async def main_async() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_index = await fetch_probe_index(args.server, args.probe_set, args.feature_idx)
    print(f"feat_{args.feature_idx} -> probe_index={probe_index}")

    print("Running blind exploration ...")
    result = await evaluate_feature(
        probe_index=probe_index,
        feature_idx=args.feature_idx,
        server=args.server,
        probe_set_name=args.probe_set,
        # Pass the blind prompt with concept blanked out.
        concept="",                 # not referenced in the custom prompt
        system_prompt=BLIND_PROMPT,
        user_prompt="Begin your exploration now.",
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        max_tokens_total=args.max_tokens,
        max_tokens_per_round=args.max_tokens,
    )
    print(f"finished_reason={result.finished_reason}  "
          f"tool_calls={result.n_tool_calls}  segments={len(result.segments)}  "
          f"elapsed={result.elapsed_seconds:.1f}s")

    out_json = out_dir / f"feat_{args.feature_idx}_blind.json"
    out_txt = out_dir / f"feat_{args.feature_idx}_blind.txt"
    with open(out_json, "w") as f:
        json.dump(serialize_result(result), f, indent=2)

    lines = [f"=== feat_{args.feature_idx} blind exploration ===",
             f"  finished: {result.finished_reason}",
             f"  segments: {len(result.segments)}", ""]
    for i, s in enumerate(result.segments):
        intv = "—" if s.intervention is None else f"strength={s.intervention.get('strength')}"
        lines.append(f"--- seg {i} role={s.role} steering={intv}")
        lines.append(s.content.strip())
        lines.append("")
    Path(out_txt).write_text("\n".join(lines))

    print(f"\nsaved -> {out_txt.resolve()}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
