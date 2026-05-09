"""Phase-A: replay GP-BO against the existing 1000-label dataset.

Treats the existing self-eval scores as ground-truth f(i) and runs the BO loop
under several strategies (UCB, Thompson, random search) for direct comparison.
The random-search baseline is the right zero point for this kind of evaluation:
it shows what we'd expect from no kernel structure at all.

Outputs (per --output-dir):
    summary.json                 aggregate metrics across replicates / strategies
    rep{NN}_{strategy}.npz       observation history + final posterior per run
    config.json                  reproducible argument record
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
from concept_search.data import load_labels, load_sae_decoder, restrict_to_labeled
from concept_search.kernel import precompute_angles
from concept_search.metrics import best_observed_curve, mean_top_k_score, recall_at_k


DEFAULT_LABELS = [
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
]
DEFAULT_SAE = "/models/sae/k25-145M-16x-k64.pt"
DEFAULT_PERSISTENCE = (
    "/home/athuser/assistant-axis-exp/results/persistence_final/k25/"
    "sae_persistence_arrays.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-A replay (see module docstring).")
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--persistence", default=DEFAULT_PERSISTENCE)
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seed-size", type=int, default=15)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--initial-lengthscale", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=8.0,
                   help="Phase-A homoscedastic obs noise (0–100 scale).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-replicates", type=int, default=5)
    p.add_argument("--strategies", nargs="+",
                   default=["ucb", "thompson", "random"])
    p.add_argument("--output-dir", default="results/phase_a")
    return p.parse_args()


def make_observe(scores: np.ndarray, noise_std: float, rng: np.random.Generator):
    noise_var = float(noise_std) ** 2

    def observe(i: int) -> tuple[float, float]:
        truth = float(scores[i])
        noisy = truth + float(rng.normal(0.0, noise_std))
        noisy = float(np.clip(noisy, 0.0, 100.0))
        return noisy, noise_var

    return observe, noise_var


def metrics_for_run(
    result,
    truth_scores: np.ndarray,
    truth_top20: set[int],
    seed_size: int,
) -> dict:
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
        "best_observed": float(max(result.observed_mean)),
        "best_observed_at_25": float(boc[min(25, len(boc) - 1)]),
        "best_observed_at_50": float(boc[min(50, len(boc) - 1)]),
        "best_observed_at_100": float(boc[min(100, len(boc) - 1)]),
        "mean_seed_score": float(np.mean(result.observed_mean[:seed_size])),
        "mean_acquired_score": float(np.mean(result.observed_mean[seed_size:])),
        "top20_mean_under_posterior": mean_top_k_score(
            result.posterior_mean, truth_scores, 20),
        "final_lengthscale": result.final_lengthscale,
    }


def aggregate(per_run_metrics: list[dict]) -> dict:
    """Median + IQR over replicates of each numeric metric."""
    out: dict[str, dict[str, float]] = {}
    keys = [k for k in per_run_metrics[0].keys() if k != "strategy"]
    for k in keys:
        vals = np.array([m[k] for m in per_run_metrics], dtype=np.float64)
        out[k] = {
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading SAE decoder + persistence...")
    sae = load_sae_decoder(args.sae, args.persistence)
    print(f"  active features: {sae.decoder.shape[0]}")

    print("Loading labeled features...")
    labels = load_labels(*args.labels)
    print(f"  labeled features: {len(labels)}")

    sae_lab, scores = restrict_to_labeled(sae, labels)
    print(f"  intersect (active ∩ labeled): {sae_lab.decoder.shape[0]}")

    print("Precomputing angle matrix...")
    angle = precompute_angles(sae_lab.decoder)
    off_diag = angle[~torch.eye(angle.shape[0], dtype=torch.bool)]
    print(f"  angle matrix: {tuple(angle.shape)}, "
          f"off-diag mean={off_diag.mean().item():.3f} rad "
          f"({np.degrees(off_diag.mean().item()):.1f}°), "
          f"min={off_diag.min().item():.3f}, max={angle.max().item():.3f}")

    candidate_idx = torch.arange(sae_lab.decoder.shape[0], dtype=torch.long)
    truth_top20 = set(np.argsort(-scores)[:20].tolist())
    print(f"  true top-20 score range: "
          f"[{scores[np.argsort(-scores)[:20]].min():.0f}, "
          f"{scores[np.argsort(-scores)[:20]].max():.0f}]")

    # Per-strategy storage.
    runs_by_strategy: dict[str, list[dict]] = {s: [] for s in args.strategies}
    boc_by_strategy: dict[str, list[np.ndarray]] = {s: [] for s in args.strategies}

    overall_t0 = time.time()
    for rep in range(args.n_replicates):
        # Each replicate uses the same RNG seed across strategies for a fair
        # comparison (same random seed-features, same observation-noise draws).
        rep_seed = args.seed + rep
        seed_rng = np.random.default_rng(rep_seed)
        seed_idx = seed_rng.choice(
            sae_lab.decoder.shape[0], size=args.seed_size, replace=False
        ).tolist()

        for strategy in args.strategies:
            print(f"\n=== Replicate {rep} | strategy={strategy} | seed={rep_seed} ===")
            obs_rng = np.random.default_rng(rep_seed)
            observe, noise_var = make_observe(scores, args.noise_std, obs_rng)
            t0 = time.time()
            result = run_bo(
                angle_matrix=angle,
                candidate_idx=candidate_idx,
                observe=observe,
                seed_idx=seed_idx,
                budget=args.budget,
                strategy=strategy,
                beta=args.beta,
                initial_lengthscale=args.initial_lengthscale,
                rng=np.random.default_rng(rep_seed + 1000),
                homoscedastic_default_var=noise_var,
            )
            m = metrics_for_run(result, scores, truth_top20, args.seed_size)
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
            runs_by_strategy[strategy].append(m)
            boc_by_strategy[strategy].append(best_observed_curve(result.observed_mean))

            np.savez(
                out_dir / f"rep{rep:02d}_{strategy}.npz",
                observed_idx=np.array(result.observed_idx, dtype=np.int64),
                observed_mean=np.array(result.observed_mean, dtype=np.float32),
                observed_var=np.array(result.observed_var, dtype=np.float32),
                posterior_mean=result.posterior_mean.astype(np.float32),
                posterior_std=result.posterior_std.astype(np.float32),
                truth_scores=scores,
                feature_indices=sae_lab.feature_indices,
                final_lengthscale=np.float32(result.final_lengthscale),
                elapsed=np.float32(result.elapsed_seconds),
                rep_seed=np.int64(rep_seed),
            )

    # Aggregate + write summary.
    summary = {
        "args": vars(args),
        "n_active": int(sae.decoder.shape[0]),
        "n_intersect": int(sae_lab.decoder.shape[0]),
        "true_top20_min": float(scores[np.argsort(-scores)[:20]].min()),
        "true_top20_max": float(scores[np.argsort(-scores)[:20]].max()),
        "true_top20_mean": float(scores[np.argsort(-scores)[:20]].mean()),
        "elapsed_total": time.time() - overall_t0,
        "by_strategy": {
            s: {
                "aggregate": aggregate(runs_by_strategy[s]),
                "best_observed_curve_median": np.median(
                    np.stack(boc_by_strategy[s]), axis=0
                ).tolist(),
                "best_observed_curve_p25": np.percentile(
                    np.stack(boc_by_strategy[s]), 25, axis=0
                ).tolist(),
                "best_observed_curve_p75": np.percentile(
                    np.stack(boc_by_strategy[s]), 75, axis=0
                ).tolist(),
                "per_run": runs_by_strategy[s],
            }
            for s in args.strategies
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary table.
    print("\n=== Summary (medians across replicates) ===")
    cols = ["spearman_rho", "recall@20_of_top20", "recall@50_of_top20",
            "best_observed_at_25", "best_observed_at_50", "best_observed_at_100",
            "top20_mean_under_posterior"]
    header = ["strategy"] + cols
    widths = [max(len(c), 11) for c in header]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for s in args.strategies:
        agg = summary["by_strategy"][s]["aggregate"]
        row = [s] + [f"{agg[c]['median']:.3f}" for c in cols]
        print("  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)))
    print(f"\n  output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
