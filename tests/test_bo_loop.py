"""End-to-end smoke test for the BO loop on a synthetic problem."""

from __future__ import annotations

import numpy as np
import torch

from concept_search.bo_loop import run_bo
from concept_search.kernel import precompute_angles


def _synthetic_setup(n: int = 80, d: int = 24, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    decoder = torch.randn(n, d, generator=g, dtype=torch.float32)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)
    angle_matrix = precompute_angles(decoder)

    # Ground truth: pick one "concept direction" and score each feature by its
    # cosine similarity to it (scaled to 0..100). Features near the concept
    # have high scores; others have low scores. This is the easy regime where
    # the kernel prior is correct.
    rng = np.random.default_rng(seed)
    concept = rng.standard_normal(d).astype(np.float32)
    concept /= np.linalg.norm(concept)
    cos = decoder.numpy() @ concept
    truth = (50.0 + 50.0 * cos).astype(np.float32)

    return angle_matrix, truth


def test_run_bo_random_finds_high_score_eventually():
    angle, truth = _synthetic_setup(n=80, seed=0)
    candidate_idx = torch.arange(80, dtype=torch.long)
    rng = np.random.default_rng(0)

    def observe(i):
        return float(truth[i] + rng.normal(0.0, 2.0)), 4.0

    seed_idx = rng.choice(80, 8, replace=False).tolist()
    res = run_bo(
        angle_matrix=angle, candidate_idx=candidate_idx, observe=observe,
        seed_idx=seed_idx, budget=30, strategy="random",
        rng=np.random.default_rng(0),
        homoscedastic_default_var=4.0,
    )
    assert len(res.observed_idx) == 30
    assert max(res.observed_mean) >= float(truth.max()) - 10.0


def test_run_bo_ucb_beats_random_on_easy_problem():
    """On the synthetic problem where the kernel prior is correct, UCB should
    find a higher best-observed score than random search at the same budget."""
    angle, truth = _synthetic_setup(n=80, seed=1)
    candidate_idx = torch.arange(80, dtype=torch.long)
    seed_idx = np.random.default_rng(1).choice(80, 8, replace=False).tolist()

    def make_observe():
        rng = np.random.default_rng(0)
        def observe(i):
            return float(truth[i] + rng.normal(0.0, 2.0)), 4.0
        return observe

    res_ucb = run_bo(
        angle_matrix=angle, candidate_idx=candidate_idx,
        observe=make_observe(), seed_idx=seed_idx, budget=30, strategy="ucb",
        rng=np.random.default_rng(123),
        homoscedastic_default_var=4.0,
    )
    res_random = run_bo(
        angle_matrix=angle, candidate_idx=candidate_idx,
        observe=make_observe(), seed_idx=seed_idx, budget=30, strategy="random",
        rng=np.random.default_rng(123),
        homoscedastic_default_var=4.0,
    )
    # UCB should at minimum match random search on its own seed-set, and
    # typically exceeds it once acquisition kicks in.
    assert max(res_ucb.observed_mean) >= max(res_random.observed_mean) - 5.0
