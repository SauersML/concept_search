"""Phase-B: live GP-BO search for SAE features matching a target concept.

Combines:
  - the coactivation-RBF kernel + run_bo loop (Phase-A)
  - the K,V-faithful agentic eval orchestrator (evaluate_feature)

Each acquired feature is evaluated by the model itself: the orchestrator runs
an agentic conversation in which the model uses steer_sae(idx, strength) to
inject the SAE feature into its residual stream and reports a 0-100 rating
for concept-relatedness. Each segment carries the steering active when it
was generated, so the server's prefill recomputes K,V faithfully when the
model attends back across the conversation.

Outputs (per --output-dir):
    summary.json           per-feature ratings + posterior
    candidates.npz         posterior_mean/std + feature_indices
    transcripts/feat_NNN.json  full segment trace per evaluated feature
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import torch

from concept_search.agentic_eval import (
    DEFAULT_SYSTEM_PROMPT,
    evaluate_feature,
    serialize_result,
)
from concept_search.bo_loop import run_bo
from concept_search.coactivation import load as load_coactivation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-B live concept search.")
    p.add_argument("--concept", required=True,
                   help='Target concept (e.g. "liquid", "deception").')
    p.add_argument("--server", default="http://localhost:8000",
                   help="Probe server URL (the orchestrator hits this for "
                        "agentic eval; needs sae_steer probes loaded).")
    p.add_argument("--probe-set", default="sae_steer")
    p.add_argument("--coactivation",
                   default="results/coactivation_phase_b.npz",
                   help="Path to the saved CoactivationResult npz over the "
                        "candidate pool.")
    p.add_argument("--budget", type=int, default=30,
                   help="Total agentic evaluations (including seed).")
    p.add_argument("--seed-size", type=int, default=10)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--initial-lengthscale", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=12.0,
                   help="Phase-B observation-noise sigma; agentic ratings are "
                        "noisier than self-eval label TSVs.")
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--max-tool-calls", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--strategy", default="ucb",
                   choices=["ucb", "thompson", "random"])
    p.add_argument("--system-prompt-file", default=None,
                   help="Path to a custom system prompt template (uses "
                        "{concept} and {feature_idx} placeholders). Defaults "
                        "to the agentic_eval module's concept-agnostic prompt.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


async def fetch_probe_labels(server: str, probe_set: str) -> list[str]:
    """List the labels (in server-side order) for the requested probe set."""
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/v1/probes", timeout=10)
        r.raise_for_status()
        for ps in r.json().get("probes", []):
            if ps["name"] == probe_set:
                return ps["labels"]
    raise RuntimeError(f"probe set {probe_set!r} not loaded on {server}")


def parse_feature_idx(label: str) -> int | None:
    if not label.startswith("feat_"):
        return None
    try:
        return int(label.split("_", 1)[1])
    except ValueError:
        return None


def build_observe_fn(
    args: argparse.Namespace,
    candidate_feature_indices: np.ndarray,   # SAE feature idx per row
    probe_index_per_row: np.ndarray,         # server-side probe-set index per row
    transcripts_dir: Path,
    system_prompt: str,
    rng: np.random.Generator,
):
    """Returns (observe, default_var) where observe(row_idx) runs one agentic
    eval against the live probe server and returns (rating, observation_var).

    Failures (no_answer, stream_error, etc.) return the prior mean (50.0) with
    a wider variance so the GP doesn't lock onto them."""
    noise_var = float(args.noise_std) ** 2
    fail_var = (2 * float(args.noise_std)) ** 2
    fail_rating = 50.0
    client_holder = {}

    async def _eval_one(row_idx: int):
        if "client" not in client_holder:
            client_holder["client"] = httpx.AsyncClient()
        client = client_holder["client"]
        feature_idx = int(candidate_feature_indices[row_idx])
        probe_index = int(probe_index_per_row[row_idx])
        result = await evaluate_feature(
            probe_index=probe_index,
            feature_idx=feature_idx,
            server=args.server,
            probe_set_name=args.probe_set,
            concept=args.concept,
            system_prompt=system_prompt,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
            max_tokens_total=args.max_tokens,
            max_tokens_per_round=args.max_tokens,
            client=client,
        )
        # Persist transcript.
        out = transcripts_dir / f"feat_{feature_idx}.json"
        with open(out, "w") as f:
            json.dump(serialize_result(result), f, indent=2)
        return result

    def observe(row_idx: int) -> tuple[float, float]:
        result = asyncio.get_event_loop().run_until_complete(_eval_one(row_idx))
        if result.rating is None:
            print(f"  [observe] feat_{result.feature_idx} no_answer "
                  f"({result.finished_reason}); using fallback "
                  f"rating={fail_rating} var={fail_var}",
                  flush=True)
            return fail_rating, fail_var
        return float(result.rating), noise_var

    def cleanup():
        c = client_holder.pop("client", None)
        if c is not None:
            asyncio.get_event_loop().run_until_complete(c.aclose())

    return observe, noise_var, cleanup


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading coactivation matrix: {args.coactivation}")
    coact = load_coactivation(args.coactivation)
    angles = torch.from_numpy(coact.angles)
    n_features = angles.shape[0]
    candidate_feature_indices = coact.feature_indices.astype(np.int64)
    print(f"  {n_features} features in candidate pool")

    print(f"\nFetching probe set {args.probe_set!r} from {args.server} ...")
    labels = asyncio.get_event_loop().run_until_complete(
        fetch_probe_labels(args.server, args.probe_set))
    label_to_probe_idx: dict[int, int] = {}
    for i, lbl in enumerate(labels):
        feat = parse_feature_idx(lbl)
        if feat is not None:
            label_to_probe_idx[feat] = i
    print(f"  probe server has {len(label_to_probe_idx)} feat_NNN labels loaded")

    # Restrict candidate pool to features actually loaded on the server.
    keep_rows = [i for i, f in enumerate(candidate_feature_indices)
                 if int(f) in label_to_probe_idx]
    if len(keep_rows) < n_features:
        print(f"  pruning {n_features - len(keep_rows)} features that are not "
              f"loaded on the server; {len(keep_rows)} remain")
    keep = np.asarray(keep_rows, dtype=np.int64)
    angles = angles[keep][:, keep]
    candidate_feature_indices = candidate_feature_indices[keep]
    n_features = len(candidate_feature_indices)
    probe_index_per_row = np.array(
        [label_to_probe_idx[int(f)] for f in candidate_feature_indices],
        dtype=np.int64,
    )

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    print(f"  system_prompt: {len(system_prompt)} chars "
          f"(custom: {bool(args.system_prompt_file)})")

    rng = np.random.default_rng(args.rng_seed)
    seed_idx = rng.choice(n_features, size=args.seed_size, replace=False).tolist()
    candidate_idx_tensor = torch.arange(n_features, dtype=torch.long)

    observe, default_var, cleanup = build_observe_fn(
        args, candidate_feature_indices, probe_index_per_row,
        transcripts_dir, system_prompt, rng,
    )

    print(f"\n=== Running BO: concept={args.concept!r}  strategy={args.strategy}  "
          f"budget={args.budget}  seed_size={args.seed_size} ===")
    t0 = time.time()
    try:
        result = run_bo(
            angle_matrix=angles,
            candidate_idx=candidate_idx_tensor,
            observe=observe,
            seed_idx=seed_idx,
            budget=args.budget,
            strategy=args.strategy,
            beta=args.beta,
            initial_lengthscale=args.initial_lengthscale,
            rng=np.random.default_rng(args.rng_seed + 1000),
            homoscedastic_default_var=default_var,
        )
    finally:
        cleanup()
    elapsed = time.time() - t0

    # Rank candidates by posterior mean.
    rank = np.argsort(-result.posterior_mean)
    topk = []
    for r in rank[:30]:
        topk.append({
            "rank": int(np.where(rank == r)[0][0]) + 1,
            "feature_idx": int(candidate_feature_indices[r]),
            "posterior_mean": float(result.posterior_mean[r]),
            "posterior_std": float(result.posterior_std[r]),
            "evaluated": int(r) in set(result.observed_idx),
        })

    summary = {
        "args": vars(args),
        "concept": args.concept,
        "n_candidates": int(n_features),
        "n_evaluated": len(result.observed_idx),
        "elapsed_seconds": elapsed,
        "final_lengthscale": result.final_lengthscale,
        "observed": [
            {
                "row": int(i),
                "feature_idx": int(candidate_feature_indices[i]),
                "rating": float(m),
                "var": float(v),
            }
            for i, m, v in zip(result.observed_idx, result.observed_mean,
                               result.observed_var)
        ],
        "top30_by_posterior_mean": topk,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        out_dir / "candidates.npz",
        feature_indices=candidate_feature_indices,
        posterior_mean=result.posterior_mean.astype(np.float32),
        posterior_std=result.posterior_std.astype(np.float32),
        observed_rows=np.array(result.observed_idx, dtype=np.int64),
        observed_ratings=np.array(result.observed_mean, dtype=np.float32),
        final_lengthscale=np.float32(result.final_lengthscale),
    )

    print(f"\n=== Top 15 candidates by posterior mean ===")
    for t in topk[:15]:
        evaluated = "✓" if t["evaluated"] else " "
        print(f"  {evaluated}  rank={t['rank']:3d}  feat_{t['feature_idx']:>6d}  "
              f"posterior={t['posterior_mean']:.1f} ± {t['posterior_std']:.1f}")
    print(f"\n  output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
