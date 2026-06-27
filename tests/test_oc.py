"""Tests for OCOptimizer: Optimality-Criteria topology optimizer.

Written BEFORE the implementation (TDD London School). All tests must fail
until src/morphos/optimize/oc.py is created.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.elasticity import ElasticityOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.topopt import TopologyOptimizer


# ---------------------------------------------------------------------------
# Shared test fixture helpers
# ---------------------------------------------------------------------------

def _cantilever_setup(shape=(6, 12), volume_fraction=0.4):
    """Build a standard 2-D cantilever and return (oracle, initial_field, objective).

    Matches the fixture used in test_optimize.py and test_elasticity.py.
    Fixed left edge, downward point load at the bottom-right corner.
    """
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads, penalty=3.0)
    initial = Field(np.full(shape, volume_fraction, dtype=float), spacing=1.0)
    return oracle, initial, MaximizeValue()


def _grayness(rho: np.ndarray) -> float:
    """Moll grayness indicator: 0 = perfectly crisp 0/1 design, 1 = max gray."""
    return float(np.mean(4.0 * rho * (1.0 - rho)))


# ---------------------------------------------------------------------------
# Test 1: volume fraction enforcement
# ---------------------------------------------------------------------------

def test_oc_hits_target_volume_fraction():
    """After optimization, mean density must be within 1e-2 of volume_fraction."""
    from morphos.optimize.oc import OCOptimizer

    volume_fraction = 0.4
    oracle, initial, objective = _cantilever_setup(volume_fraction=volume_fraction)
    opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=20,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    mean_density = float(np.mean(result.field.values))
    assert abs(mean_density - volume_fraction) < 1e-2, (
        f"mean density {mean_density:.4f} not within 1e-2 of {volume_fraction}"
    )


# ---------------------------------------------------------------------------
# Test 2: OC beats fixed-step gradient ascent at equal compute budget
# ---------------------------------------------------------------------------

def test_oc_beats_fixed_step_ascent_on_compliance():
    """OC must reach higher fom (lower compliance) than gradient ascent
    with VolumeConstraint at equal max_iter, across a sweep of step sizes.

    Both optimizers enforce volume_fraction=0.4. The sweep ensures we do not
    advantage OC by picking a bad step for TopologyOptimizer.
    """
    from morphos.optimize.oc import OCOptimizer
    from morphos.manufacturing.constraints import VolumeConstraint

    volume_fraction = 0.4
    shape = (6, 12)
    max_iter = 80

    oracle_oc, initial, objective = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    oc_opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=max_iter,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    res_oc = oc_opt.run(initial.copy(), oracle_oc, objective)

    # Find the best TopologyOptimizer result over a sweep of step sizes.
    # VolumeConstraint makes the comparison fair: both have the same volume budget.
    volume_constraint = VolumeConstraint(volume_fraction)
    best_topopt_fom = -np.inf
    for step in [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        oracle_top, initial_top, obj_top = _cantilever_setup(
            shape=shape, volume_fraction=volume_fraction
        )
        top_opt = TopologyOptimizer(
            step_size=step,
            max_iter=max_iter,
            tol=1e-12,
            bounds=(1e-3, 1.0),
            p_start=1.0,
            p_end=3.0,
            p_ramp_fraction=0.5,
        )
        res_top = top_opt.run(
            initial_top.copy(), oracle_top, obj_top, constraint=volume_constraint
        )
        best_topopt_fom = max(best_topopt_fom, res_top.fom)

    assert res_oc.fom >= best_topopt_fom, (
        f"OC fom {res_oc.fom:.6f} should be >= best topopt fom {best_topopt_fom:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: OC produces crisper (less gray) designs
# ---------------------------------------------------------------------------

def test_oc_produces_crisper_design():
    """OC must drive densities toward 0/1 more aggressively than fixed-step
    ascent at equal max_iter, measured by the Moll grayness indicator."""
    from morphos.optimize.oc import OCOptimizer
    from morphos.manufacturing.constraints import VolumeConstraint

    volume_fraction = 0.4
    shape = (6, 12)
    max_iter = 40

    oracle_oc, initial, objective = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    oc_opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=max_iter,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    res_oc = oc_opt.run(initial.copy(), oracle_oc, objective)

    # TopologyOptimizer reference with VolumeConstraint for a fair comparison.
    oracle_top, initial_top, obj_top = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    volume_constraint = VolumeConstraint(volume_fraction)
    top_opt = TopologyOptimizer(
        step_size=1e-2,
        max_iter=max_iter,
        tol=1e-12,
        bounds=(1e-3, 1.0),
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
    )
    res_top = top_opt.run(
        initial_top.copy(), oracle_top, obj_top, constraint=volume_constraint
    )

    g_oc = _grayness(res_oc.field.values)
    g_top = _grayness(res_top.field.values)
    assert g_oc < g_top, (
        f"OC grayness {g_oc:.4f} should be < topopt grayness {g_top:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: raises ValueError when gradient is unavailable
# ---------------------------------------------------------------------------

def test_oc_requires_gradient():
    """OCOptimizer must raise ValueError when the objective gradient is None.

    OC's multiplicative update is derived from the analytic sensitivity; it
    cannot fall back to finite differences like TopologyOptimizer can.
    """
    from morphos.optimize.oc import OCOptimizer
    from morphos.physics.oracle import PhysicsOracle, PhysicsResult

    class NoGradOracle(PhysicsOracle):
        provides_gradient = False

        def solve(self, field):
            return PhysicsResult(value=-1.0, gradient=None)

    oracle = NoGradOracle()
    initial = Field(np.full((4, 8), 0.4), spacing=1.0)
    opt = OCOptimizer(volume_fraction=0.4, max_iter=5)
    with pytest.raises(ValueError, match="gradient"):
        opt.run(initial, oracle, MaximizeValue())


# ---------------------------------------------------------------------------
# Test 5: result.fom equals max(history), not the last iteration's fom
# ---------------------------------------------------------------------------

def test_oc_returns_best_not_last():
    """result.fom must equal the maximum fom seen in history.

    If fom is not monotone (e.g. due to p-continuation overshooting), the
    optimizer must still return the best design seen, not the last.
    """
    from morphos.optimize.oc import OCOptimizer

    volume_fraction = 0.4
    oracle, initial, objective = _cantilever_setup(volume_fraction=volume_fraction)
    opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=20,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    assert len(result.history) > 0
    assert abs(result.fom - max(result.history)) < 1e-10, (
        f"result.fom {result.fom} != max(history) {max(result.history)}"
    )
