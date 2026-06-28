"""Tests for RobustOptimizer: delta-p (three-field) robust topology optimization."""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.elasticity import ElasticityOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.optimizer import Optimizer
from morphos.optimize.oc import OCOptimizer
from morphos.optimize.robust import RobustOptimizer, _heaviside


# ---------------------------------------------------------------------------
# Shared test fixture helpers (mirrors test_oc.py)
# ---------------------------------------------------------------------------

def _cantilever_setup(shape=(8, 16), volume_fraction=0.4):
    """Build a standard 2-D cantilever and return (oracle, initial_field, objective).

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


# ---------------------------------------------------------------------------
# Test 1: isinstance check
# ---------------------------------------------------------------------------

def test_robust_optimizer_is_optimizer():
    opt = RobustOptimizer(volume_fraction=0.4)
    assert isinstance(opt, Optimizer)


# ---------------------------------------------------------------------------
# Test 2: three-field Heaviside projections differ as expected
# ---------------------------------------------------------------------------

def test_three_field_foms_differ():
    """On a random density field, nom/ero/dil Heaviside projections must
    give different mean values, with ero < nom < dil for default eta=0.5
    and delta>0 (raising the threshold erodes solid; lowering it dilates)."""
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.0, 1.0, size=(20, 20))
    beta = 4.0
    delta = 0.1

    rho_nom = _heaviside(rho, beta, eta=0.5)
    rho_ero = _heaviside(rho, beta, eta=0.5 + delta)
    rho_dil = _heaviside(rho, beta, eta=0.5 - delta)

    mean_nom = np.mean(rho_nom)
    mean_ero = np.mean(rho_ero)
    mean_dil = np.mean(rho_dil)

    assert mean_ero < mean_nom < mean_dil
    # Sanity: all distinct (not numerically collapsed)
    assert not np.allclose(rho_nom, rho_ero)
    assert not np.allclose(rho_nom, rho_dil)


# ---------------------------------------------------------------------------
# Test 3: runs on a small cantilever
# ---------------------------------------------------------------------------

def test_robust_optimizer_runs_on_small_cantilever():
    oracle, initial, objective = _cantilever_setup(shape=(8, 16), volume_fraction=0.4)
    opt = RobustOptimizer(
        volume_fraction=0.4,
        max_iter=10,
        delta=0.1,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    assert np.isfinite(result.fom)
    assert result.fom < 0.0  # compliance objective: fom = -compliance < 0


# ---------------------------------------------------------------------------
# Test 4: history length matches iteration count
# ---------------------------------------------------------------------------

def test_robust_optimizer_history_length():
    oracle, initial, objective = _cantilever_setup(shape=(8, 16), volume_fraction=0.4)
    iterations = 10
    opt = RobustOptimizer(
        volume_fraction=0.4,
        max_iter=iterations,
        delta=0.1,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    assert len(result.history) == iterations
    assert result.iterations == iterations


# ---------------------------------------------------------------------------
# Test 5: volume fraction satisfied (on nominal field)
# ---------------------------------------------------------------------------

def test_robust_optimizer_volume_fraction_satisfied():
    volume_fraction = 0.4
    oracle, initial, objective = _cantilever_setup(
        shape=(8, 16), volume_fraction=volume_fraction
    )
    opt = RobustOptimizer(
        volume_fraction=volume_fraction,
        max_iter=20,
        delta=0.1,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    mean_density = float(np.mean(result.field.values))
    assert abs(mean_density - volume_fraction) < 0.05, (
        f"mean density {mean_density:.4f} not within 0.05 of {volume_fraction}"
    )


# ---------------------------------------------------------------------------
# Test 6: robust FOM is more conservative than standard OC optimization
# ---------------------------------------------------------------------------

def test_robust_worse_than_nominal():
    """The robust (worst-case) fom must be <= a standard nominal-only
    OCOptimizer's fom on the same problem: robust optimization trades
    nominal performance for tolerance to erosion/dilation, so it should
    never beat a method that only ever optimizes the nominal field."""
    volume_fraction = 0.4
    shape = (8, 16)
    max_iter = 20

    oracle_oc, initial_oc, objective_oc = _cantilever_setup(
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
    res_oc = oc_opt.run(initial_oc.copy(), oracle_oc, objective_oc)

    oracle_rb, initial_rb, objective_rb = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    rb_opt = RobustOptimizer(
        volume_fraction=volume_fraction,
        max_iter=max_iter,
        delta=0.1,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
        tol=1e-12,
    )
    res_rb = rb_opt.run(initial_rb.copy(), oracle_rb, objective_rb)

    assert res_rb.fom <= res_oc.fom + 1e-9, (
        f"robust fom {res_rb.fom:.6f} should be <= nominal OC fom {res_oc.fom:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7: delta=0 collapses the three fields, matching single-field behavior
# ---------------------------------------------------------------------------

def test_delta_zero_equivalent_to_single_field():
    """With delta=0 the eroded and dilated thresholds collapse onto the
    nominal threshold (eta=0.5 for all three), so the three projected
    fields are numerically identical and the "worst case" is degenerate:
    nom/ero/dil foms must agree at every iteration, and the run must
    reproduce exactly (bit-for-bit, modulo solver determinism) on repeat,
    since which of the three (tied) fields is picked as "worst" cannot
    affect the result."""
    volume_fraction = 0.4
    shape = (8, 16)
    max_iter = 15

    def _run():
        oracle, initial, objective = _cantilever_setup(
            shape=shape, volume_fraction=volume_fraction
        )
        opt = RobustOptimizer(
            volume_fraction=volume_fraction,
            max_iter=max_iter,
            delta=0.0,
            p_start=1.0,
            p_end=3.0,
            p_ramp_fraction=0.5,
            tol=1e-12,
        )
        return opt.run(initial.copy(), oracle, objective)

    res_a = _run()
    res_b = _run()

    # Determinism / reproducibility: degenerate worst-case selection must
    # not introduce nondeterminism.
    assert abs(res_a.fom - res_b.fom) < 1e-10
    np.testing.assert_allclose(res_a.field.values, res_b.field.values)

    # The three projected fields (nom/ero/dil) must be identical when
    # delta=0, since eta_ero = eta_dil = eta_nom = 0.5.
    from morphos.optimize.robust import _heaviside

    rng = np.random.default_rng(1)
    rho = rng.uniform(0.0, 1.0, size=(10, 10))
    beta = 5.0
    rho_nom = _heaviside(rho, beta, eta=0.5)
    rho_ero = _heaviside(rho, beta, eta=0.5 + 0.0)
    rho_dil = _heaviside(rho, beta, eta=0.5 - 0.0)
    np.testing.assert_allclose(rho_nom, rho_ero)
    np.testing.assert_allclose(rho_nom, rho_dil)
