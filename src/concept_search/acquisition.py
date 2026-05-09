"""UCB acquisition over a finite candidate pool.

We don't optimize over a continuous space — we score every un-evaluated feature
ID and pick the argmax. With a few-thousand candidate pool this is trivially
fast and avoids the gradient-based BoTorch optimizer that doesn't help on
discrete inputs.
"""

from __future__ import annotations

import math

import torch


def ucb_scores(
    model,
    candidate_idx: torch.Tensor,   # [M] long
    beta: float = 2.0,
) -> torch.Tensor:
    """UCB(x) = mu(x) + sqrt(beta) * sigma(x), evaluated at every candidate."""
    X = candidate_idx.unsqueeze(-1).float()
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1).clamp_min(0.0)
    return mean + math.sqrt(beta) * var.sqrt()


def pick_next(
    model,
    candidate_idx: torch.Tensor,
    observed_idx: set[int],
    beta: float = 2.0,
) -> int:
    """Return the candidate ID with the highest UCB, excluding observed ones."""
    scores = ucb_scores(model, candidate_idx, beta=beta)
    mask = torch.tensor([int(i) not in observed_idx for i in candidate_idx.tolist()])
    scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    pick = candidate_idx[scores.argmax()].item()
    return int(pick)
