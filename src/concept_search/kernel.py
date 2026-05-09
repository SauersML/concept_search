"""Angular-RBF kernel on unit-norm SAE decoder columns.

K(i, j) = exp(-theta_ij^2 / (2 * lengthscale^2)),
theta_ij = arccos(clamp(<W_dec[:, i], W_dec[:, j]>, -1, 1)).

Inputs to the kernel are integer feature indices into a fixed candidate set,
not the decoder vectors themselves — that lets us precompute the angle matrix
once and skip the Gram math at every BO step.
"""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel


def precompute_angles(decoder: torch.Tensor) -> torch.Tensor:
    """Pairwise angles between unit-norm decoder rows.

    decoder: [N, d_model], rows are unit-norm.
    returns: [N, N] of theta_ij in radians, on the same device as `decoder`.
    """
    cos = (decoder @ decoder.T).clamp(-1.0, 1.0)
    return torch.arccos(cos)


class AngularRBFKernel(Kernel):
    """RBF kernel on a precomputed angle matrix indexed by integer feature IDs.

    The "input" passed at .forward() is a tensor of integer indices into the
    angle matrix. This avoids materializing the full Gram matrix and makes the
    BO loop cheap — every observed/candidate point is just an int lookup.
    """

    has_lengthscale = True

    def __init__(self, angle_matrix: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        if angle_matrix.dim() != 2 or angle_matrix.shape[0] != angle_matrix.shape[1]:
            raise ValueError("angle_matrix must be square 2D tensor")
        self.register_buffer("angle_matrix", angle_matrix)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        # Inputs are floats in BoTorch's convention; round and cast to long.
        i1 = x1.squeeze(-1).round().long()
        i2 = x2.squeeze(-1).round().long()
        if diag:
            theta = self.angle_matrix[i1, i1]
            theta_sq = theta.pow(2)
            ls_sq = self.lengthscale.squeeze().pow(2)
            return torch.exp(-theta_sq / (2.0 * ls_sq))

        theta = self.angle_matrix[i1.unsqueeze(-1), i2.unsqueeze(-2)]  # [..., n1, n2]
        theta_sq = theta.pow(2)
        ls_sq = self.lengthscale.view(*([1] * (theta_sq.dim() - 2)), 1, 1).pow(2)
        return torch.exp(-theta_sq / (2.0 * ls_sq))
