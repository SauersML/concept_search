"""Generate the same prompts the ConceptDirector uses, capture text, save.

So we can see exactly what Kimi writes for each concept-seed and each
neutral-seed — i.e. the raw text whose mean residual becomes the steering
direction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from concept_search.concept_resolver import (
    CONCEPT_TEXT_PROMPTS,
    NEUTRAL_NAMES,
)


async def chat(client, server, prompt, max_tokens, temperature):
    body = {"messages": [{"role": "user", "content": prompt}],
            "stream": True, "max_tokens": max_tokens,
            "temperature": temperature}
    chunks = []
    async with client.stream("POST", f"{server}/v1/chat/completions",
                             json=body, timeout=180.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            ds = line[6:]
            if ds == "[DONE]":
                break
            try: d = json.loads(ds)
            except: continue
            choices = d.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta: chunks.append(delta)
    return "".join(chunks)


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--concepts", nargs="+", default=["melancholy"])
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        # Neutral baseline: every CONCEPT template × every NEUTRAL_NAME, so
        # the dump matches what _compute_baseline actually feeds the encoder.
        print("=== NEUTRAL prompts ===")
        neutral_dump = []
        for name in NEUTRAL_NAMES:
            for tmpl in CONCEPT_TEXT_PROMPTS:
                prompt = tmpl.format(name=name)
                print(f"\n--- prompt: {prompt!r} ---")
                text = await chat(client, args.server, prompt,
                                  args.max_tokens, args.temperature)
                print(text)
                neutral_dump.append({"name": name, "template": tmpl,
                                     "prompt": prompt, "text": text})
        (out / "neutral.json").write_text(
            json.dumps(neutral_dump, indent=2))

        for name in args.concepts:
            print(f"\n\n=== CONCEPT: {name} ===")
            concept_dump = []
            for tmpl in CONCEPT_TEXT_PROMPTS:
                prompt = tmpl.format(name=name)
                print(f"\n--- prompt: {prompt!r} ---")
                text = await chat(client, args.server, prompt,
                                  args.max_tokens, args.temperature)
                print(text)
                concept_dump.append({"prompt": prompt, "text": text})
            sane = name.replace(" ", "_")[:32]
            (out / f"concept_{sane}.json").write_text(
                json.dumps(concept_dump, indent=2))

    print(f"\n\nsaved -> {out.resolve()}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
