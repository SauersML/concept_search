"""Acquisition functions over a finite candidate pool.

We don't optimize over a continuous space — we score every un-evaluated feature
ID and pick by argmax. With a few-thousand-candidate pool this is trivially
fast and avoids the gradient-based BoTorch optimizer that doesn't help on
discrete inputs.

Three acquisition strategies:
    - ucb       : argmax_i mu(i) + sqrt(beta) * sigma(i)
    - thompson  : draw one posterior sample over all candidates, argmax it
    - random    : pure random search (baseline; ignores the GP)
"""

from __future__ import annotations

import math

import numpy as np
import torch


def ucb_scores(
    model,
    candidate_idx: torch.Tensor,   # [M] long
    beta: float = 2.0,
) -> torch.Tensor:
    """UCB(x) = mu(x) + sqrt(beta) * sigma(x), evaluated at every candidate."""
    X = candidate_idx.unsqueeze(-1).double()
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1).clamp_min(0.0)
    return mean + math.sqrt(beta) * var.sqrt()


def thompson_scores(
    model,
    candidate_idx: torch.Tensor,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Single-draw Thompson sample over candidates.

    GPyTorchPosterior.rsample takes no `generator` argument, so we seed the
    global torch RNG locally for reproducibility.
    """
    X = candidate_idx.unsqueeze(-1).double()
    model.eval()
    with torch.no_grad():
        if seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                posterior = model.posterior(X)
                sample = posterior.rsample(torch.Size([1]))
        else:
            posterior = model.posterior(X)
            sample = posterior.rsample(torch.Size([1]))
    return sample.squeeze(0).squeeze(-1)


def _argmax_unobserved(
    scores: torch.Tensor,
    candidate_idx: torch.Tensor,
    observed_idx: set[int],
) -> int:
    mask = torch.tensor([int(i) not in observed_idx for i in candidate_idx.tolist()])
    masked = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    return int(candidate_idx[masked.argmax()].item())


def pick_next(
    model,
    candidate_idx: torch.Tensor,
    observed_idx: set[int],
    *,
    strategy: str = "ucb",
    beta: float = 2.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Return one candidate ID to evaluate next, excluding already-observed."""
    if rng is None:
        rng = np.random.default_rng()

    if strategy == "random":
        unobs = [int(i) for i in candidate_idx.tolist() if int(i) not in observed_idx]
        if not unobs:
            raise RuntimeError("no unobserved candidates remain")
        return int(rng.choice(unobs))

    if strategy == "ucb":
        scores = ucb_scores(model, candidate_idx, beta=beta)
    elif strategy == "thompson":
        # Derive a torch seed from the numpy rng so successive picks differ
        # while remaining reproducible against `rng`.
        seed = int(rng.integers(0, 2**31 - 1))
        scores = thompson_scores(model, candidate_idx, seed=seed)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return _argmax_unobserved(scores, candidate_idx, observed_idx)
