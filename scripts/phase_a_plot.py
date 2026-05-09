"""Plot Phase-A results: best-observed-by-step curve per strategy + posterior scatter.

Reads summary.json + the per-strategy npz files written by phase_a_replay.py
and saves two PNGs into the same directory:
  best_observed_curves.png   best-observed-vs-step per strategy
  posterior_vs_truth.png     scatter of posterior mean vs true score, faceted
                             by strategy. True top-20 highlighted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "figure.dpi": 160,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=Path)
    return p.parse_args()


def plot_best_observed_curves(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    strategies = list(summary["by_strategy"].keys())
    colors = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    for color, s in zip(colors, strategies):
        curve = np.array(summary["by_strategy"][s]["best_observed_curve"])
        x = np.arange(1, len(curve) + 1)
        ax.plot(x, curve, color=color, label=s)

    true_top = summary["true_top20_max"]
    ax.axhline(true_top, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best observed score")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_posterior_vs_truth(input_dir: Path, summary: dict, out_path: Path) -> None:
    strategies = list(summary["by_strategy"].keys())
    fig, axes = plt.subplots(
        1, len(strategies),
        figsize=(4.2 * len(strategies), 4.4),
        sharex=True, sharey=True,
    )
    if len(strategies) == 1:
        axes = [axes]

    for ax, s in zip(axes, strategies):
        npz = input_dir / f"{s}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        truth = d["truth_scores"]
        post = d["posterior_mean"]

        top20 = set(np.argsort(-truth)[:20].tolist())
        is_top = np.array([i in top20 for i in range(len(truth))])

        ax.scatter(truth[~is_top], post[~is_top], s=8, color="0.7", alpha=0.5)
        ax.scatter(truth[is_top], post[is_top], s=24, color="C3", alpha=0.9)
        ax.set_xlabel("true score")
        if ax is axes[0]:
            ax.set_ylabel("posterior mean")
        ax.set_title(s)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.input_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    plot_best_observed_curves(summary, args.input_dir / "best_observed_curves.png")
    plot_posterior_vs_truth(
        args.input_dir, summary, args.input_dir / "posterior_vs_truth.png"
    )
    print(f"plots -> {args.input_dir.resolve()}/")


if __name__ == "__main__":
    main()
