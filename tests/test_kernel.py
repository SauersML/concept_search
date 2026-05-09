"""Sanity tests for the angular-RBF kernel and angle precomputation."""

from __future__ import annotations

import math

import torch

from concept_search.kernel import AngularRBFKernel, precompute_angles


def _random_unit_decoder(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def test_angles_diagonal_zero():
    decoder = _random_unit_decoder(20, 64)
    angles = precompute_angles(decoder)
    assert angles.shape == (20, 20)
    diag = angles.diagonal()
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-4)


def test_angles_symmetric():
    decoder = _random_unit_decoder(15, 32, seed=1)
    angles = precompute_angles(decoder)
    assert torch.allclose(angles, angles.T, atol=1e-5)


def test_angles_range():
    decoder = _random_unit_decoder(10, 16, seed=2)
    angles = precompute_angles(decoder)
    assert (angles >= 0).all()
    assert (angles <= math.pi + 1e-4).all()


def test_kernel_psd():
    # Build a small angle matrix from real unit vectors and check the kernel
    # matrix is symmetric PSD at a couple of lengthscales.
    decoder = _random_unit_decoder(30, 24, seed=3)
    angles = precompute_angles(decoder)
    for ls in (0.1, 0.5, 1.0):
        kern = AngularRBFKernel(angle_matrix=angles)
        kern.lengthscale = torch.tensor([ls])
        idx = torch.arange(30).unsqueeze(-1).float()
        K = kern.forward(idx, idx).to_dense()
        assert K.shape == (30, 30)
        assert torch.allclose(K, K.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(30))
        assert (eigvals > -1e-5).all(), f"non-PSD at ls={ls}: min eig {eigvals.min()}"
        diag = K.diagonal()
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_kernel_index_lookup():
    decoder = _random_unit_decoder(12, 16, seed=4)
    angles = precompute_angles(decoder)
    kern = AngularRBFKernel(angle_matrix=angles)
    kern.lengthscale = torch.tensor([0.3])

    i1 = torch.tensor([0, 5]).unsqueeze(-1).float()
    i2 = torch.tensor([5, 7, 11]).unsqueeze(-1).float()
    K = kern.forward(i1, i2).to_dense()
    assert K.shape == (2, 3)

    expected = torch.exp(-angles[[0, 5]][:, [5, 7, 11]].pow(2) / (2 * 0.3 ** 2))
    assert torch.allclose(K, expected, atol=1e-5)
