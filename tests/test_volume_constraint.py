"""Tests for VolumeConstraint: a volume-preserving density projection.

The constraint shifts the density field by a single constant, clipped to
[0, 1], so the mean density equals a target volume fraction. This is the
classic SIMP volume control expressed as a projection onto the constant-volume
set, with a clip-mask vector Jacobian product so the optimizer can chain
gradients through it.
"""

import numpy as np

from morphos.field import Field
from morphos.manufacturing.constraints import VolumeConstraint


def test_project_hits_target_volume_fraction():
    rng = np.random.default_rng(0)
    rho = rng.uniform(size=(8, 8, 8))
    for target in (0.2, 0.4, 0.6, 0.8):
        c = VolumeConstraint(target)
        out = c.project(Field(rho, spacing=1.0))
        assert np.isclose(out.values.mean(), target, atol=1e-6)
        assert out.values.min() >= 0.0 and out.values.max() <= 1.0


def test_project_handles_already_on_target():
    rho = np.full((4, 4, 4), 0.4)
    out = VolumeConstraint(0.4).project(Field(rho, spacing=1.0))
    assert np.isclose(out.values.mean(), 0.4, atol=1e-9)


def test_vjp_is_clip_mask_of_the_shift():
    rng = np.random.default_rng(1)
    rho = rng.uniform(size=(6, 6))
    c = VolumeConstraint(0.4)
    grad = rng.normal(size=rho.shape)
    field = Field(rho, spacing=1.0)
    # The projection is clip(rho + shift, 0, 1); its Jacobian is the identity
    # on unsaturated voxels and zero where the clip is active.
    shift = c._shift(rho)
    shifted = rho + shift
    mask = (shifted > 0.0) & (shifted < 1.0)
    expected = grad * mask
    assert np.allclose(c.vjp(field, grad), expected)


def test_vjp_matches_finite_difference_on_frozen_shift():
    # With the shift frozen at its value at x0, the projection is a plain clip,
    # which is differentiable away from the clip boundaries. A central
    # difference on the unsaturated voxels must match the vjp mask exactly.
    rng = np.random.default_rng(2)
    rho = 0.3 + 0.4 * rng.uniform(size=(5, 5))  # well inside (0, 1)
    c = VolumeConstraint(0.4)
    shift = c._shift(rho)
    g = lambda x: np.clip(x + shift, 0.0, 1.0)
    eps = 1e-6
    seed = rng.normal(size=rho.shape)
    field = Field(rho, spacing=1.0)
    jvp_fd = (g(rho + eps * seed) - g(rho - eps * seed)) / (2 * eps)
    # <vjp(grad), seed> == <grad, J seed> for any grad (adjoint identity)
    grad = rng.normal(size=rho.shape)
    lhs = np.sum(c.vjp(field, grad) * seed)
    rhs = np.sum(grad * jvp_fd)
    assert np.isclose(lhs, rhs, rtol=1e-5, atol=1e-7)


def test_report_records_volume_fraction():
    rho = np.full((4, 4, 4), 0.4)
    rep = VolumeConstraint(0.5).report(Field(rho, spacing=1.0))
    assert np.isclose(rep["volume_fraction"], 0.4)
    assert np.isclose(rep["target_volume_fraction"], 0.5)


def test_rejects_degenerate_targets():
    import pytest

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            VolumeConstraint(bad)
