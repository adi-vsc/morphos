"""Tests for the electromagnetic (FDFD) backend.

These skip unless the optional electromagnetic dependency is installed, so core
CI stays fast and dependency-light. When ceviche is present they exercise the
real solve, the normalized figure of merit, the adjoint gradient, and an
end-to-end inverse-design run through the engine.
"""

import numpy as np
import pytest

pytest.importorskip("ceviche")

from morphos.field import Field  # noqa: E402
from morphos.physics.ceviche_em import CevicheEMOracle  # noqa: E402
from morphos.physics.oracle import PhysicsOracle  # noqa: E402
from morphos.objective.objective import MaximizeValue  # noqa: E402
from morphos.optimize.topopt import TopologyOptimizer  # noqa: E402
from morphos.spec import DesignSpec  # noqa: E402
from morphos.engine import Engine  # noqa: E402


SHAPE = (48, 48)


def make_oracle():
    return CevicheEMOracle(
        shape=SHAPE,
        source=(10, 24),
        probe=(38, 24),
        wavelength=1550e-9,
        dl=40e-9,
        npml=8,
        eps_max=12.25,
    )


def vacuum_field():
    return Field(np.zeros(SHAPE), spacing=40e-9)


def test_em_oracle_is_a_physics_oracle():
    o = make_oracle()
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_solve_returns_result_with_matching_gradient_shape():
    o = make_oracle()
    r = o.solve(vacuum_field())
    assert np.isfinite(r.value)
    assert r.gradient.shape == SHAPE


def test_baseline_fom_is_normalized_to_one():
    o = make_oracle()
    # density zero is vacuum, which defines the reference, so the FOM is ~1
    assert o.solve(vacuum_field()).value == pytest.approx(1.0, rel=1e-6)


def test_adjoint_agrees_with_finite_difference_at_top_cell():
    o = make_oracle()
    f = Field(np.full(SHAPE, 0.3), spacing=40e-9)
    grad = o.solve(f).gradient
    idx = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
    base = o.solve(f).value
    eps = 1e-2
    fp = f.copy()
    fp.values[idx] += eps
    fd = (o.solve(fp).value - base) / eps
    assert fd == pytest.approx(grad[idx], rel=0.3)


def test_engine_inverse_design_improves_focus():
    o = make_oracle()
    spec = DesignSpec(
        initial=vacuum_field(),
        oracle=o,
        objective=MaximizeValue(),
        optimizer=TopologyOptimizer(step_size=5.0, max_iter=20, bounds=(0.0, 1.0)),
        name="em-focusing",
    )
    res = Engine().run(spec)
    assert res.used_finite_differences is False
    assert res.history[-1] > res.history[0]
    assert res.figure_of_merit > 1.0  # beats the vacuum baseline
