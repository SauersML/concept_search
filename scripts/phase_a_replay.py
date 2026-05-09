"""Phase-A: GP-BO replay using the coactivation kernel.

Treats existing self-eval scores as ground-truth f(i) and runs BO against the
1000-label dataset under three strategies (UCB, Thompson, random) so the
kernel's value-add over random search is directly visible.

The kernel is angular-RBF on a coactivation angle matrix, built once by
scripts/build_coactivation.py from cached layer-40 activations and the SAE's
encoder. We dropped the decoder-direction kernel: TopK training pushes decoder
columns near-orthogonal independently of concept similarity, so that prior was
empty for this SAE. Coactivation measures concept similarity directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from concept_search.bo_loop import run_bo
from concept_search.coactivation import load as load_coactivation
from concept_search.data import load_labels
from concept_search.metrics import best_observed_curve, mean_top_k_score, recall_at_k


DEFAULT_LABELS = [
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-A coactivation-kernel replay.")
    p.add_argument("--coactivation",
                   default="results/coactivation_k25_labeled.npz",
                   help="Path to the saved CoactivationResult npz.")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seed-size", type=int, default=15)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--initial-lengthscale", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategies", nargs="+",
                   default=["ucb", "thompson", "random"])
    p.add_argument("--output-dir", default="results/phase_a")
    return p.parse_args()


def make_observe(scores: np.ndarray, noise_std: float, rng: np.random.Generator):
    noise_var = float(noise_std) ** 2

    def observe(i: int) -> tuple[float, float]:
        truth = float(scores[i])
        noisy = truth + float(rng.normal(0.0, noise_std))
        return float(np.clip(noisy, 0.0, 100.0)), noise_var

    return observe, noise_var


def metrics_for_run(result, truth_scores: np.ndarray, truth_top20: set[int],
                    seed_size: int) -> dict:
    posterior_rank = np.argsort(-result.posterior_mean).tolist()
    rho, _ = spearmanr(result.posterior_mean, truth_scores)
    boc = best_observed_curve(result.observed_mean)
    return {
        "strategy": result.strategy,
        "elapsed": result.elapsed_seconds,
        "spearman_rho": float(rho if rho is not None else float("nan")),
        "recall@20_of_top20": recall_at_k(truth_top20, posterior_rank, 20),
        "recall@50_of_top20": recall_at_k(truth_top20, posterior_rank, 50),
        "recall@100_of_top20": recall_at_k(truth_top20, posterior_rank, 100),
        "best_observed_at_25": float(boc[min(25, len(boc) - 1)]),
        "best_observed_at_50": float(boc[min(50, len(boc) - 1)]),
        "best_observed_at_100": float(boc[min(100, len(boc) - 1)]),
        "mean_seed_score": float(np.mean(result.observed_mean[:seed_size])),
        "mean_acquired_score": float(np.mean(result.observed_mean[seed_size:])),
        "top20_mean_under_posterior": mean_top_k_score(
            result.posterior_mean, truth_scores, 20),
        "final_lengthscale": result.final_lengthscale,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading coactivation matrix: {args.coactivation}")
    coact = load_coactivation(args.coactivation)
    angles = torch.from_numpy(coact.angles)
    n_features = angles.shape[0]
    print(f"  {n_features} features, n_tokens_used={coact.n_tokens_used}")

    off_diag = angles[~torch.eye(n_features, dtype=torch.bool)]
    print(f"  off-diag angle: mean={off_diag.mean().item():.3f} rad "
          f"({np.degrees(off_diag.mean().item()):.1f}°), "
          f"min={off_diag.min().item():.3f}, "
          f"p5={float(np.percentile(off_diag.numpy(), 5)):.3f}")

    print("Loading labels...")
    labels = load_labels(*args.labels)
    truth_scores = np.array(
        [float(labels.loc[int(idx), "score"]) for idx in coact.feature_indices],
        dtype=np.float32,
    )
    truth_top20 = set(np.argsort(-truth_scores)[:20].tolist())
    print(f"  true top-20 score range: "
          f"[{truth_scores[np.argsort(-truth_scores)[:20]].min():.0f}, "
          f"{truth_scores[np.argsort(-truth_scores)[:20]].max():.0f}]")

    candidate_idx = torch.arange(n_features, dtype=torch.long)

    seed_rng = np.random.default_rng(args.seed)
    seed_idx = seed_rng.choice(n_features, size=args.seed_size, replace=False).tolist()

    runs: dict[str, dict] = {}
    boc_by_strategy: dict[str, list[float]] = {}
    overall_t0 = time.time()
    for strategy in args.strategies:
        print(f"\n=== strategy={strategy} ===")
        obs_rng = np.random.default_rng(args.seed)
        observe, noise_var = make_observe(truth_scores, args.noise_std, obs_rng)
        t0 = time.time()
        result = run_bo(
            angle_matrix=angles,
            candidate_idx=candidate_idx,
            observe=observe,
            seed_idx=seed_idx,
            budget=args.budget,
            strategy=strategy,
            beta=args.beta,
            initial_lengthscale=args.initial_lengthscale,
            rng=np.random.default_rng(args.seed + 1000),
            homoscedastic_default_var=noise_var,
        )
        m = metrics_for_run(result, truth_scores, truth_top20, args.seed_size)
        print(
            f"  spearman={m['spearman_rho']:+.3f}  "
            f"recall@20={m['recall@20_of_top20']:.2f}  "
            f"recall@50={m['recall@50_of_top20']:.2f}  "
            f"best@25={m['best_observed_at_25']:.0f}  "
            f"best@50={m['best_observed_at_50']:.0f}  "
            f"best@100={m['best_observed_at_100']:.0f}  "
            f"top20_post_mean={m['top20_mean_under_posterior']:.1f}  "
            f"ls={m['final_lengthscale']:.3f}  "
            f"({time.time() - t0:.1f}s)"
        )
        runs[strategy] = m
        boc_by_strategy[strategy] = best_observed_curve(result.observed_mean).tolist()

        np.savez(
            out_dir / f"{strategy}.npz",
            observed_idx=np.array(result.observed_idx, dtype=np.int64),
            observed_mean=np.array(result.observed_mean, dtype=np.float32),
            observed_var=np.array(result.observed_var, dtype=np.float32),
            posterior_mean=result.posterior_mean.astype(np.float32),
            posterior_std=result.posterior_std.astype(np.float32),
            truth_scores=truth_scores,
            feature_indices=coact.feature_indices,
            final_lengthscale=np.float32(result.final_lengthscale),
            elapsed=np.float32(result.elapsed_seconds),
        )

    summary = {
        "args": vars(args),
        "n_features": int(n_features),
        "true_top20_min": float(truth_scores[np.argsort(-truth_scores)[:20]].min()),
        "true_top20_max": float(truth_scores[np.argsort(-truth_scores)[:20]].max()),
        "true_top20_mean": float(truth_scores[np.argsort(-truth_scores)[:20]].mean()),
        "elapsed_total": time.time() - overall_t0,
        "by_strategy": {
            s: {"metrics": runs[s], "best_observed_curve": boc_by_strategy[s]}
            for s in args.strategies
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    cols = ["spearman_rho", "recall@20_of_top20", "recall@50_of_top20",
            "best_observed_at_25", "best_observed_at_50", "best_observed_at_100",
            "top20_mean_under_posterior"]
    header = ["strategy"] + cols
    widths = [max(len(c), 11) for c in header]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for s in args.strategies:
        m = runs[s]
        row = [s] + [f"{m[c]:.3f}" for c in cols]
        print("  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)))
    print(f"\n  output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
