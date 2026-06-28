"""Tests for LevelSetOptimizer: Hamilton-Jacobi level-set topology optimizer.

Written against src/morphos/optimize/levelset.py.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.optimize.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Test 1: isinstance(Optimizer)
# ---------------------------------------------------------------------------

def test_levelset_optimizer_is_optimizer():
    from morphos.optimize.levelset import LevelSetOptimizer

    opt = LevelSetOptimizer(volume_fraction=0.4)
    assert isinstance(opt, Optimizer)


# ---------------------------------------------------------------------------
# Test 2: _density_vjp matches finite differences of _density_from_phi
# ---------------------------------------------------------------------------

def test_density_from_phi_and_vjp_consistent():
    from morphos.optimize.levelset import LevelSetOptimizer

    rng = np.random.default_rng(0)
    opt = LevelSetOptimizer(volume_fraction=0.4, eta=10.0)
    phi = rng.uniform(-0.5, 0.5, size=(3, 4))
    grad_rho = rng.normal(size=(3, 4))

    analytic = opt._density_vjp(phi, grad_rho)

    eps = 1e-6
    fd = np.zeros_like(phi)
    it = np.nditer(phi, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        phi_p = phi.copy()
        phi_p[idx] += eps
        phi_m = phi.copy()
        phi_m[idx] -= eps
        d_rho = (
            opt._density_from_phi(phi_p)[idx] - opt._density_from_phi(phi_m)[idx]
        ) / (2 * eps)
        fd[idx] = d_rho * grad_rho[idx]
        it.iternext()

    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: reinit gives an (approximate) signed distance function
# ---------------------------------------------------------------------------

def test_reinit_gives_signed_distance():
    from morphos.optimize.levelset import LevelSetOptimizer

    # Start from a non-SDF level set with a roughly circular interface so
    # there is a nontrivial zero level set to reinitialize around.
    ny, nx = 40, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2.0, nx / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    phi0 = (r - 12.0) * 3.0  # scaled, not a true SDF

    phi_sdf = LevelSetOptimizer._reinit(phi0)

    grads = np.gradient(phi_sdf)
    grad_mag = np.sqrt(grads[0] ** 2 + grads[1] ** 2)
    # Skip a thin border where one-sided differences are less accurate.
    interior = grad_mag[2:-2, 2:-2]
    assert 0.8 <= float(np.mean(interior)) <= 1.2


# ---------------------------------------------------------------------------
# Test 4: volume bisection enforces the target volume fraction
# ---------------------------------------------------------------------------

def test_volume_bisect_enforces_target():
    from morphos.optimize.levelset import LevelSetOptimizer

    rng = np.random.default_rng(1)
    opt = LevelSetOptimizer(volume_fraction=0.4, eta=10.0)
    phi = rng.uniform(-2.0, 2.0, size=(10, 12))

    target_vf = 0.4
    phi_shifted = opt._volume_bisect(phi, target_vf)
    rho = opt._density_from_phi(phi_shifted)
    assert abs(float(np.mean(rho)) - target_vf) < 1e-5


# ---------------------------------------------------------------------------
# Test 5: optimizer reduces compliance (raises fom) on a small cantilever
# ---------------------------------------------------------------------------

def test_levelset_optimizer_reduces_compliance_on_cantilever():
    from morphos.optimize.levelset import LevelSetOptimizer
    from morphos.physics.elasticity import ElasticityOracle
    from morphos.objective.objective import MaximizeValue, PhysicalBound

    def cantilever_bcs(shape, load=-1.0):
        ny, nx = shape
        nny, nnx = ny + 1, nx + 1
        fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
        loads = {(nnx - 1, nny - 1, "y"): load}
        return fixed, loads

    shape = (8, 16)
    fixed, loads = cantilever_bcs(shape)
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)
    objective = MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid"))
    initial = Field(np.full(shape, 0.4), spacing=1.0)
    opt = LevelSetOptimizer(
        volume_fraction=0.4, max_iter=20, dt=0.05, p_start=1.0, p_end=2.0
    )
    result = opt.run(initial, oracle, objective)

    assert result.history[-1] > result.history[0]
    assert abs(float(np.mean(result.field.values)) - 0.4) < 0.05


# ---------------------------------------------------------------------------
# Test 6: on_iteration callback fires with (iter, fom, delta, p)
# ---------------------------------------------------------------------------

def test_levelset_on_iteration_callback():
    from morphos.optimize.levelset import LevelSetOptimizer
    from morphos.physics.elasticity import ElasticityOracle
    from morphos.objective.objective import MaximizeValue

    def cantilever_bcs(shape, load=-1.0):
        ny, nx = shape
        nny, nnx = ny + 1, nx + 1
        fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
        loads = {(nnx - 1, nny - 1, "y"): load}
        return fixed, loads

    shape = (6, 8)
    fixed, loads = cantilever_bcs(shape)
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)
    objective = MaximizeValue()
    initial = Field(np.full(shape, 0.4), spacing=1.0)
    opt = LevelSetOptimizer(volume_fraction=0.4, max_iter=5, dt=0.05)

    calls = []

    def on_iteration(iteration, fom, delta, p):
        calls.append((iteration, fom, delta, p))

    opt.run(initial, oracle, objective, on_iteration=on_iteration)

    assert len(calls) > 0
    for iteration, fom, delta, p in calls:
        assert isinstance(iteration, int)
        assert np.isfinite(fom)
        assert delta >= 0.0
        assert p > 0.0
