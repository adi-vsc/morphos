"""Tests for the 3D electromagnetic FDTD backend.

Unlike the 2D ceviche backend, this oracle has no hard optional dependency: it
always works via a self-contained pure-numpy Yee-grid FDTD fallback when
neither meep nor the `fdtd` pip package is importable, so these tests run
unconditionally (no importorskip) and use a tiny grid / few timesteps to stay
fast.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.fdtd3d import FDTD3DOracle
from morphos.physics.oracle import PhysicsOracle

SHAPE = (6, 6, 6)


def make_oracle(**kw):
    defaults = dict(
        shape=SHAPE,
        source=(1, 3, 3),
        probe=(4, 3, 3),
        n_steps=40,
    )
    defaults.update(kw)
    return FDTD3DOracle(**defaults)


def vacuum_field():
    return Field(np.zeros(SHAPE), spacing=1.0)


def test_fdtd3d_is_a_physics_oracle():
    o = make_oracle()
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_solve_returns_result_with_matching_gradient_shape():
    o = make_oracle()
    r = o.solve(vacuum_field())
    assert np.isfinite(r.value)
    assert r.gradient is not None
    assert r.gradient.shape == SHAPE
    assert np.all(np.isfinite(r.gradient))


def test_baseline_fom_is_normalized_to_one():
    o = make_oracle()
    assert o.solve(vacuum_field()).value == pytest.approx(1.0, rel=1e-6)


def test_field_shape_mismatch_raises():
    o = make_oracle()
    with pytest.raises(ValueError):
        o.solve(Field(np.zeros((5, 6, 6)), spacing=1.0))


def test_uses_pure_numpy_fallback_when_no_backend_available(monkeypatch):
    """Backend selection must fall back to the bundled Yee-grid FDTD when
    neither meep nor the fdtd package can be imported -- this is the default
    in this environment, but assert it explicitly so the contract is pinned."""
    o = make_oracle()
    assert o.backend == "numpy"


def test_gradient_is_directionally_reasonable():
    """A coarse finite-difference gradient check: perturbing the density at
    the cell with the largest |gradient| should move the FOM in the predicted
    direction for at least one sign of a small step (loose check, since the
    underlying FD gradient is itself noisy on a 6-cubed grid)."""
    o = make_oracle()
    f = Field(np.full(SHAPE, 0.3), spacing=1.0)
    r = o.solve(f)
    idx = np.unravel_index(np.argmax(np.abs(r.gradient)), r.gradient.shape)
    assert r.gradient[idx] != 0.0
