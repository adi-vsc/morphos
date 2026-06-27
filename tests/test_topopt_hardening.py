"""Hardening gates for the topology optimizer.

Two robustness properties the engine lacked:

1. Non-finite guard. A diverging solve (NaN / Inf figure of merit or gradient)
   used to propagate silently into the design through the ascent step and the
   bound clip, returning a garbage Field that *looked* like a valid result. The
   optimizer must instead fail loud with a clear error naming the iteration.

2. Optional monotonic line search. Fixed-step ascent overshoots on a non-convex
   (or simply ill-scaled) problem: a too-large step can step *past* the optimum
   and lower the figure of merit. With backtracking enabled the per-iteration
   figure of merit must never decrease.

3. Composite adjoint gate. The optimizer chains the objective gradient back
   through the constraint projection, the Heaviside projection, and the density
   filter. Each stage is FD-gated on its own elsewhere; this locks the *whole*
   chain (filter -> Heaviside -> constraint -> oracle) as one composite gradient
   against a central finite difference, so a future change to any one stage that
   breaks the end-to-end sensitivity is caught.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.topopt import TopologyOptimizer, _evaluate

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fd_gate import fd_gate  # noqa: E402


class _NaNOracle(PhysicsOracle):
    """Returns a non-finite figure of merit and gradient, simulating a solve
    that diverged (singular system, blown-up Krylov residual, etc.)."""

    provides_gradient = True

    def solve(self, field):
        bad = np.full_like(field.values, np.nan)
        return PhysicsResult(value=float("nan"), gradient=bad)


# --- 1. Non-finite guard --------------------------------------------------


def test_nonfinite_fom_raises_clear_error():
    opt = TopologyOptimizer(step_size=0.1, max_iter=10)
    with pytest.raises(RuntimeError, match="non-finite"):
        opt.run(Field(np.zeros((3, 3)), spacing=1.0), _NaNOracle(), MaximizeValue())


def test_finite_problem_is_unaffected_by_the_guard():
    """The guard must not change behavior on a normal finite problem."""
    target = np.array([[0.2, -0.4], [0.9, 0.1]])
    opt = TopologyOptimizer(step_size=0.2, max_iter=2000, tol=1e-12)
    res = opt.run(Field(np.zeros((2, 2)), spacing=1.0), AnalyticOracle(target), MaximizeValue())
    assert res.fom == pytest.approx(0.0, abs=1e-6)


# --- 2. Optional monotonic line search ------------------------------------


def test_backtracking_keeps_fom_monotone_under_an_oversized_step():
    """With an oversized fixed step, plain ascent overshoots (the FOM history
    dips at least once); enabling backtracking must keep it non-decreasing."""
    target = np.array([[1.0, -1.0, 0.5]])
    init = Field(np.zeros((1, 3)), spacing=1.0)

    # An oversized step makes fixed-step ascent oscillate across the optimum:
    # the FOM history repeatedly dips instead of climbing monotonically.
    plain = TopologyOptimizer(step_size=1.5, max_iter=40, tol=0.0)
    res_plain = plain.run(init.copy(), AnalyticOracle(target), MaximizeValue())
    dips = np.diff(res_plain.history)
    assert np.any(dips < -1e-9), "expected the oversized plain step to overshoot"

    guarded = TopologyOptimizer(step_size=1.5, max_iter=40, tol=0.0, line_search=True)
    res_guarded = guarded.run(init.copy(), AnalyticOracle(target), MaximizeValue())
    dips_g = np.diff(res_guarded.history)
    assert np.all(dips_g >= -1e-9), "backtracking must keep the FOM monotone"
    # And it must still make real progress toward the optimum (fom -> 0).
    assert res_guarded.fom > res_plain.history[0]


# --- 3. Composite end-to-end adjoint gate ---------------------------------


def test_full_design_chain_gradient_passes_fd_gate():
    """filter -> Heaviside -> constraint -> oracle, as one composite gradient."""
    from morphos.manufacturing.constraints import MinFeatureSize

    rng = np.random.default_rng(3)
    shape = (5, 6)
    x0 = rng.uniform(0.2, 0.8, shape)
    target = rng.uniform(-1.0, 1.0, shape)

    oracle = AnalyticOracle(target=target)
    obj = MaximizeValue()
    constraint = MinFeatureSize(radius=1)
    beta = 6.0

    opt = TopologyOptimizer(filter_radius=1, beta_start=beta, beta_end=beta)

    def f(flat_x):
        field = Field(flat_x.reshape(shape), spacing=1.0)
        return opt._fom(field, oracle, obj, constraint, beta)

    x_field = Field(x0, spacing=1.0)
    design, filtered = opt._design_chain(x_field, beta, constraint)
    ov = _evaluate(design, oracle, obj)
    grad_raw = opt._chain_vjp(x_field, filtered, beta, ov.gradient, constraint)

    fd_gate(f, grad_raw, x0, rel=1e-4)
