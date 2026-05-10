"""Inject a live_concepts steering direction into a plain chat with no prompt
priming, and see what the model actually says.

User says only "Hi!" The model has no system prompt, no mention of steering,
no concept name in context. The steering direction (and strength) come from
intervention= in the API call, applied during decode.

Compares the same "Hi!" under (a) no steering, (b) the chosen concept at
high strength, (c) optionally negative strength.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def chat_under_steering(
    client: httpx.AsyncClient,
    server: str,
    user_message: str,
    intervention: dict | None,
    max_tokens: int = 600,
    temperature: float = 0.8,
) -> str:
    body = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if intervention is not None:
        body["intervention"] = intervention
    chunks: list[str] = []
    async with client.stream(
        "POST", f"{server}/v1/chat/completions",
        json=body, timeout=180.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            ds = line[6:]
            if ds == "[DONE]":
                break
            try:
                d = json.loads(ds)
            except json.JSONDecodeError:
                continue
            choices = d.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    chunks.append(delta)
    return "".join(chunks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--concept", required=True,
                   help="Label in the live_concepts probe set.")
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 80.0, -80.0])
    p.add_argument("--user", default="Hi!")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--probe-set", default="live_concepts")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


async def fetch_label_index(server: str, probe_set: str, label: str) -> int:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/v1/probes", timeout=10)
        r.raise_for_status()
        for ps in r.json().get("probes", []):
            if ps["name"] == probe_set:
                if label in ps["labels"]:
                    return ps["labels"].index(label)
                raise SystemExit(
                    f"label {label!r} not in {probe_set!r}; have "
                    f"{ps['labels']}")
    raise SystemExit(f"probe set {probe_set!r} not loaded on {server}")


async def main_async() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_index = await fetch_label_index(args.server, args.probe_set,
                                          args.concept)
    print(f"concept: {args.concept!r} -> probe_index={probe_index}")
    print(f"user message: {args.user!r}")
    print(f"max_tokens: {args.max_tokens}  temperature: {args.temperature}")
    print()

    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for strength in args.strengths:
            if abs(strength) < 0.01:
                interv = None
                tag = "NO STEERING"
            else:
                interv = {
                    "type": "steer",
                    "probe": args.probe_set,
                    "probe_index": probe_index,
                    "strength": float(strength),
                    "renorm": True,
                }
                tag = f"steering={args.concept}@{strength:+.0f}"
            print(f"=== {tag} ===")
            text = await chat_under_steering(
                client, args.server, args.user, interv,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            print(text)
            print()
            results.append({"strength": strength, "intervention": interv,
                            "response": text})

    # Save.
    sane = args.concept.replace(" ", "_")
    json_path = out_dir / f"inject_{sane}.json"
    txt_path = out_dir / f"inject_{sane}.txt"
    with open(json_path, "w") as f:
        json.dump({"concept": args.concept, "user_message": args.user,
                   "results": results}, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(f"concept: {args.concept}\nuser: {args.user}\n\n")
        for r in results:
            tag = (f"strength={r['strength']:+.0f}"
                   if r["intervention"] else "NO STEERING")
            f.write(f"=== {tag} ===\n{r['response']}\n\n")
    print(f"saved -> {txt_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
