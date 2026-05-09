"""Bayesian-optimization driver: seed → loop[acquire, observe, update] → report.

This module is intentionally agnostic of *how* observations are produced —
Phase-A passes in a function that looks up labels from a TSV; Phase-B passes
in the agentic-eval function that runs the model under steering. The BO loop
is the same.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from concept_search.acquisition import pick_next
from concept_search.gp import fit_gp, make_gp


# An observe-function takes a feature ID (in the candidate pool's coordinate
# system: an int in [0, N)) and returns (mean_rating, observation_variance).
ObserveFn = Callable[[int], tuple[float, float]]


@dataclass
class BOResult:
    observed_idx: list[int]
    observed_mean: list[float]
    observed_var: list[float]
    posterior_mean: np.ndarray   # [N], over all candidates at end
    posterior_std: np.ndarray    # [N]
    final_lengthscale: float
    strategy: str
    elapsed_seconds: float


def _build_model(
    observed_idx: list[int],
    observed_mean: list[float],
    observed_var: list[float],
    angle_matrix: torch.Tensor,
    initial_lengthscale: float,
):
    train_idx = torch.tensor(observed_idx, dtype=torch.long)
    train_y = torch.tensor(observed_mean, dtype=torch.float64)
    train_yvar = torch.tensor(observed_var, dtype=torch.float64)
    use_fixed = bool((train_yvar > 0).all().item())
    return make_gp(
        train_idx=train_idx,
        train_y=train_y,
        train_yvar=train_yvar if use_fixed else None,
        angle_matrix=angle_matrix,
        initial_lengthscale=initial_lengthscale,
    )


def run_bo(
    angle_matrix: torch.Tensor,           # [N, N]
    candidate_idx: torch.Tensor,          # [M], long, IDs into [0, N)
    observe: ObserveFn,
    seed_idx: list[int],
    *,
    budget: int = 100,
    strategy: str = "ucb",                # "ucb" | "thompson" | "random"
    beta: float = 2.0,
    refit_every: int = 10,
    initial_lengthscale: float = 0.5,
    rng: np.random.Generator | None = None,
    homoscedastic_default_var: float | None = None,
    verbose: bool = False,
) -> BOResult:
    """Standard GP-BO loop on a finite candidate pool.

    seed_idx: the IDs at which to make initial observations (e.g. ~15 random
    candidates) before the GP starts driving acquisition. Always evaluated
    even under strategy="random" — the random baseline corresponds to using
    the seed loop for the entire budget, ignoring the GP.
    budget: total observations including the seed.
    refit_every: re-learn lengthscale every this many new observations.
    homoscedastic_default_var: if observe() returns var=0 we substitute this.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    observed_idx: list[int] = []
    observed_mean: list[float] = []
    observed_var: list[float] = []
    seen: set[int] = set()
    t0 = time.time()

    def _observe(i: int) -> None:
        m, v = observe(i)
        if v <= 0.0 and homoscedastic_default_var is not None:
            v = homoscedastic_default_var
        observed_idx.append(i)
        observed_mean.append(float(m))
        observed_var.append(float(v))
        seen.add(i)

    # Seed
    for i in seed_idx[:budget]:
        _observe(int(i))

    if verbose:
        print(f"  [bo_loop] seed done ({len(observed_idx)} obs); strategy={strategy}")

    # Acquire-observe-update
    while len(observed_idx) < budget:
        if strategy == "random":
            next_i = pick_next(
                model=None, candidate_idx=candidate_idx,
                observed_idx=seen, strategy="random", rng=rng,
            )
        else:
            model = _build_model(
                observed_idx, observed_mean, observed_var,
                angle_matrix, initial_lengthscale,
            )
            # Re-fit hyperparameters periodically.
            n_acquired = len(observed_idx) - len(seed_idx)
            if n_acquired % refit_every == 0:
                try:
                    fit_gp(model)
                except Exception as e:
                    print(f"  [bo_loop] fit_gp failed: {type(e).__name__}: {e}")
            next_i = pick_next(
                model=model, candidate_idx=candidate_idx,
                observed_idx=seen, strategy=strategy,
                beta=beta, rng=rng,
            )
        _observe(next_i)

    # Final posterior — even under random search we fit a GP at the end so we
    # have a comparable posterior for diagnostics.
    final_model = _build_model(
        observed_idx, observed_mean, observed_var,
        angle_matrix, initial_lengthscale,
    )
    try:
        fit_gp(final_model)
    except Exception as e:
        print(f"  [bo_loop] final fit_gp failed: {type(e).__name__}: {e}")

    final_model.eval()
    X_all = candidate_idx.unsqueeze(-1).double()
    with torch.no_grad():
        post = final_model.posterior(X_all)
        mean_all = post.mean.squeeze(-1).cpu().numpy()
        std_all = post.variance.clamp_min(0.0).sqrt().squeeze(-1).cpu().numpy()

    try:
        final_ls = float(
            final_model.covar_module.lengthscale.detach().cpu().squeeze().item()
        )
    except Exception:
        final_ls = float("nan")

    return BOResult(
        observed_idx=observed_idx,
        observed_mean=observed_mean,
        observed_var=observed_var,
        posterior_mean=mean_all,
        posterior_std=std_all,
        final_lengthscale=final_ls,
        strategy=strategy,
        elapsed_seconds=time.time() - t0,
    )
