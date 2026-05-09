"""GP wrapper: BoTorch SingleTaskGP with our angular-RBF kernel and per-observation noise.

Inputs are integer feature IDs (cast to float for BoTorch's tensor pipeline) into
a fixed candidate pool of size N. Targets are the (mean) ratings on a 0–100
scale. Per-observation noise variance is provided externally — homoscedastic
in Phase-A, heteroscedastic in Phase-B.
"""

from __future__ import annotations

import gpytorch
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

from concept_search.kernel import AngularRBFKernel


def make_gp(
    train_idx: torch.Tensor,         # [n] long
    train_y: torch.Tensor,            # [n] float, 0..100
    train_yvar: torch.Tensor | None,  # [n] float (variance) or None
    angle_matrix: torch.Tensor,       # [N, N] precomputed angles (radians)
    initial_lengthscale: float = 0.5,
) -> SingleTaskGP:
    """Build a GP with AngularRBFKernel.

    If `train_yvar` is None we use a SingleTaskGP with a learned homoscedastic
    likelihood. Otherwise FixedNoiseGP with the given per-point variances.
    """
    X = train_idx.unsqueeze(-1).float()
    Y = train_y.unsqueeze(-1).float()

    kernel = AngularRBFKernel(angle_matrix=angle_matrix)
    kernel.lengthscale = torch.tensor([initial_lengthscale])

    mean = ConstantMean()
    outcome_tf = Standardize(m=1)

    if train_yvar is None:
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-4),
        )
        model = SingleTaskGP(
            train_X=X, train_Y=Y,
            covar_module=kernel,
            mean_module=mean,
            likelihood=likelihood,
            outcome_transform=outcome_tf,
        )
    else:
        # Modern BoTorch unified FixedNoiseGP into SingleTaskGP via train_Yvar.
        Yvar = train_yvar.unsqueeze(-1).float().clamp_min(1e-4)
        model = SingleTaskGP(
            train_X=X, train_Y=Y, train_Yvar=Yvar,
            covar_module=kernel,
            mean_module=mean,
            outcome_transform=outcome_tf,
        )

    return model


def fit_gp(model: SingleTaskGP) -> None:
    """Fit GP hyperparameters by exact marginal log-likelihood."""
    from botorch.fit import fit_gpytorch_mll

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
