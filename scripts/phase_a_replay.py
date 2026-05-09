"""Phase-A: replay GP-BO against the existing 1000-label dataset.

Treats the existing self-eval scores as ground-truth f(i) and runs the GP-BO
loop with a budget of 100 observations against that finite candidate pool.
Reports recall@K of the top-K features, Spearman rho between posterior mean
and ground truth, and the seed-vs-acquired score gap.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sae", default="/models/sae/k25-145M-16x-k64.pt")
    p.add_argument("--persistence",
                   default="/home/athuser/assistant-axis-exp/results/persistence_final/k25/sae_persistence_arrays.npz")
    p.add_argument("--labels", nargs="+",
                   default=[
                       "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
                       "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
                   ])
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seed-size", type=int, default=15)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--initial-lengthscale", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=8.0,
                   help="Phase-A homoscedastic noise (0–100 scale, std).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/phase_a")
    p.add_argument("--n-replicates", type=int, default=3,
                   help="Repeat the BO loop N times with different RNG seeds.")
    return p.parse_args()


def recall_at_k(true_top: set[int], recovered: list[int], k: int) -> float:
    return len(true_top & set(recovered[:k])) / max(len(true_top), 1)


def evaluate_replicate(
    angle_matrix: torch.Tensor,
    candidate_idx: torch.Tensor,
    truth_scores: np.ndarray,
    truth_top20: set[int],
    args: argparse.Namespace,
    rep_seed: int,
) -> dict:
    """One BO-replicate. Returns metrics + the run's posterior."""
    rng = np.random.default_rng(rep_seed)
    N = candidate_idx.numel()

    seed_idx = rng.choice(N, size=args.seed_size, replace=False).tolist()
    noise_var = float(args.noise_std) ** 2

    def observe(i: int) -> tuple[float, float]:
        # Phase-A: ground-truth label + Gaussian observation noise.
        truth = float(truth_scores[i])
        noisy = truth + float(rng.normal(0.0, args.noise_std))
        noisy = max(0.0, min(100.0, noisy))
        return noisy, noise_var

    t0 = time.time()
    result = run_bo(
        angle_matrix=angle_matrix,
        candidate_idx=candidate_idx,
        observe=observe,
        seed_idx=seed_idx,
        budget=args.budget,
        beta=args.beta,
        initial_lengthscale=args.initial_lengthscale,
        rng=rng,
        homoscedastic_default_var=noise_var,
    )

    posterior_rank = np.argsort(-result.posterior_mean).tolist()  # best first
    rho, _ = spearmanr(result.posterior_mean, truth_scores)

    metrics = {
        "rep_seed": rep_seed,
        "elapsed": time.time() - t0,
        "spearman_rho": float(rho if rho is not None else float("nan")),
        "recall_at_20_of_top20": recall_at_k(truth_top20, posterior_rank, 20),
        "recall_at_50_of_top20": recall_at_k(truth_top20, posterior_rank, 50),
        "recall_at_100_of_top20": recall_at_k(truth_top20, posterior_rank, 100),
        "best_observed_score": float(max(result.observed_mean)),
        "mean_observed_score": float(np.mean(result.observed_mean)),
        "mean_seed_score": float(np.mean(result.observed_mean[: args.seed_size])),
        "mean_acquired_score": float(np.mean(result.observed_mean[args.seed_size:])),
        "final_lengthscale": result.final_lengthscale,
    }
    return {"metrics": metrics, "result": result, "posterior_rank": posterior_rank}


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  angle matrix: {tuple(angle.shape)}, "
          f"min={angle.min().item():.3f}, max={angle.max().item():.3f}, "
          f"mean={angle.mean().item():.3f}")

    candidate_idx = torch.arange(sae_lab.decoder.shape[0], dtype=torch.long)
    truth_top20 = set(np.argsort(-scores)[:20].tolist())

    all_metrics = []
    for rep in range(args.n_replicates):
        print(f"\n=== Replicate {rep} (seed={args.seed + rep}) ===")
        run = evaluate_replicate(
            angle, candidate_idx, scores, truth_top20, args,
            rep_seed=args.seed + rep,
        )
        m = run["metrics"]
        print(f"  spearman_rho={m['spearman_rho']:.3f}  "
              f"recall@20={m['recall_at_20_of_top20']:.2f}  "
              f"recall@50={m['recall_at_50_of_top20']:.2f}  "
              f"recall@100={m['recall_at_100_of_top20']:.2f}")
        print(f"  best_observed={m['best_observed_score']:.1f}  "
              f"seed_mean={m['mean_seed_score']:.1f}  "
              f"acquired_mean={m['mean_acquired_score']:.1f}  "
              f"lengthscale={m['final_lengthscale']:.3f}  "
              f"elapsed={m['elapsed']:.1f}s")
        all_metrics.append(m)

        # Persist per-rep posterior + observation history.
        np.savez(out_dir / f"rep{rep:02d}.npz",
                 observed_idx=np.array(run["result"].observed_idx, dtype=np.int64),
                 observed_mean=np.array(run["result"].observed_mean, dtype=np.float32),
                 observed_var=np.array(run["result"].observed_var, dtype=np.float32),
                 posterior_mean=run["result"].posterior_mean.astype(np.float32),
                 posterior_std=run["result"].posterior_std.astype(np.float32),
                 truth_scores=scores,
                 feature_indices=sae_lab.feature_indices,
                 final_lengthscale=np.float32(run["result"].final_lengthscale),
                 elapsed=np.float32(run["result"].elapsed_seconds))

    summary = {
        "args": vars(args),
        "n_active": int(sae.decoder.shape[0]),
        "n_labeled": int(len(labels)),
        "n_intersect": int(sae_lab.decoder.shape[0]),
        "replicates": all_metrics,
        "median_spearman_rho": float(np.median([m["spearman_rho"] for m in all_metrics])),
        "median_recall_at_20": float(np.median([m["recall_at_20_of_top20"] for m in all_metrics])),
        "median_recall_at_50": float(np.median([m["recall_at_50_of_top20"] for m in all_metrics])),
        "median_best_observed": float(np.median([m["best_observed_score"] for m in all_metrics])),
        "true_top20_max": float(scores[np.argsort(-scores)[:20]].max()),
        "true_top20_min": float(scores[np.argsort(-scores)[:20]].min()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"  median spearman_rho    : {summary['median_spearman_rho']:.3f}")
    print(f"  median recall@20 of T20: {summary['median_recall_at_20']:.2f}")
    print(f"  median recall@50 of T20: {summary['median_recall_at_50']:.2f}")
    print(f"  median best observed   : {summary['median_best_observed']:.1f}")
    print(f"  true T20 score range   : [{summary['true_top20_min']:.0f}, "
          f"{summary['true_top20_max']:.0f}]")
    print(f"  output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
