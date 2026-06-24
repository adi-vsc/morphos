"""The iterative (conjugate-gradient) solve path for HeatConductionOracle.

The direct sparse LU solve does not scale: its fill-in blows up memory on large
3D-scale grids. The interior operator is symmetric positive-definite, so a
matrix-free preconditioned CG is a valid drop-in that scales. These tests pin
the iterative path to the direct path it replaces.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.heat import HeatConductionOracle


def central_fd_gradient(oracle, field, eps=1e-5):
    grad = np.zeros_like(field.values)
    it = np.nditer(field.values, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        fp = field.copy()
        fp.values[idx] += eps
        fm = field.copy()
        fm.values[idx] -= eps
        grad[idx] = (oracle.solve(fp).value - oracle.solve(fm).value) / (2 * eps)
        it.iternext()
    return grad


def test_unknown_solver_raises():
    with pytest.raises(ValueError):
        HeatConductionOracle(target=np.zeros((5, 5)), solver="nonsense")


def test_cg_temperature_matches_direct():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(9, 9)) * 0.1
    s = rng.normal(size=(9, 9))
    direct = HeatConductionOracle(target=target, solver="direct")
    cg = HeatConductionOracle(target=target, solver="cg")
    Td = direct.solve(Field(s, spacing=1.0)).aux["temperature"]
    Tc = cg.solve(Field(s, spacing=1.0)).aux["temperature"]
    assert np.allclose(Td, Tc, atol=1e-8)


def test_cg_value_matches_direct():
    rng = np.random.default_rng(8)
    target = rng.normal(size=(9, 9)) * 0.1
    s = rng.normal(size=(9, 9))
    vd = HeatConductionOracle(target=target, solver="direct").solve(Field(s, spacing=1.0)).value
    vc = HeatConductionOracle(target=target, solver="cg").solve(Field(s, spacing=1.0)).value
    assert vc == pytest.approx(vd, abs=1e-8)


def test_cg_adjoint_matches_finite_differences():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(6, 6)) * 0.1
    o = HeatConductionOracle(target=target, solver="cg")
    f = Field(rng.normal(size=(6, 6)) * 0.1, spacing=1.0)
    analytic = o.solve(f).gradient
    numeric = central_fd_gradient(o, f)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_cg_solution_satisfies_the_linear_system():
    o = HeatConductionOracle(target=np.zeros((9, 9)), solver="cg")
    rng = np.random.default_rng(10)
    s = rng.normal(size=(9, 9))
    r = o.solve(Field(s, spacing=1.0))
    residual = o.residual(s, r.aux["temperature"])
    assert np.max(np.abs(residual)) < 1e-6


def test_unknown_preconditioner_raises():
    with pytest.raises(ValueError):
        HeatConductionOracle(target=np.zeros((5, 5)), solver="cg", preconditioner="nope")


def test_amg_preconditioned_cg_matches_direct():
    pytest.importorskip("pyamg")
    rng = np.random.default_rng(11)
    target = rng.normal(size=(16, 16)) * 0.1
    s = rng.normal(size=(16, 16))
    direct = HeatConductionOracle(target=target, solver="direct")
    amg = HeatConductionOracle(target=target, solver="cg", preconditioner="amg")
    Td = direct.solve(Field(s, spacing=1.0)).aux["temperature"]
    Ta = amg.solve(Field(s, spacing=1.0)).aux["temperature"]
    assert np.allclose(Td, Ta, atol=1e-7)


# --- new "iterative"/"auto" solver path (AMG + CG via morphos.physics._linsolve) --

def test_iterative_solver_value_is_invalid_alongside_legacy_cg_value():
    # "iterative" and "auto" are new, distinct accepted values for `solver`,
    # alongside the legacy "direct"/"cg"; all four must be accepted.
    for mode in ("direct", "cg", "iterative", "auto"):
        HeatConductionOracle(target=np.zeros((5, 5)), solver=mode)


def test_direct_solve_reports_zero_residual_and_iterations():
    target = np.zeros((6, 6))
    o = HeatConductionOracle(target=target, solver="direct")
    rng = np.random.default_rng(20)
    s = rng.normal(size=(6, 6))
    r = o.solve(Field(s, spacing=1.0))
    assert r.solver_iterations == 0
    assert r.residual_norm == 0.0


def test_iterative_solver_matches_direct_temperature():
    pytest.importorskip("pyamg")
    rng = np.random.default_rng(21)
    target = rng.normal(size=(16, 16)) * 0.1
    s = rng.normal(size=(16, 16))
    direct = HeatConductionOracle(target=target, solver="direct")
    it = HeatConductionOracle(target=target, solver="iterative")
    Td = direct.solve(Field(s, spacing=1.0)).aux["temperature"]
    Ti = it.solve(Field(s, spacing=1.0)).aux["temperature"]
    assert np.allclose(Td, Ti, atol=1e-7)


def test_iterative_solver_reports_iterations_and_residual():
    pytest.importorskip("pyamg")
    rng = np.random.default_rng(22)
    target = rng.normal(size=(16, 16)) * 0.1
    s = rng.normal(size=(16, 16))
    o = HeatConductionOracle(target=target, solver="iterative")
    r = o.solve(Field(s, spacing=1.0))
    assert r.solver_iterations > 0
    assert r.residual_norm < 1e-6


def test_auto_solver_picks_direct_for_small_grid():
    target = np.zeros((9, 9))
    o = HeatConductionOracle(target=target, solver="auto")
    rng = np.random.default_rng(23)
    s = rng.normal(size=(9, 9))
    r = o.solve(Field(s, spacing=1.0))
    assert r.solver_iterations == 0


def test_auto_solver_small_grid_matches_explicit_direct():
    rng = np.random.default_rng(24)
    target = rng.normal(size=(9, 9)) * 0.1
    s = rng.normal(size=(9, 9))
    direct = HeatConductionOracle(target=target, solver="direct").solve(Field(s, spacing=1.0))
    auto = HeatConductionOracle(target=target, solver="auto").solve(Field(s, spacing=1.0))
    assert np.allclose(direct.aux["temperature"], auto.aux["temperature"], atol=1e-10)


def test_iterative_solver_amg_cache_is_reused_across_solves():
    pytest.importorskip("pyamg")
    target = np.zeros((16, 16))
    o = HeatConductionOracle(target=target, solver="iterative")
    rng = np.random.default_rng(25)
    o.solve(Field(rng.normal(size=(16, 16)), spacing=1.0))
    cache = o._amg_cache
    assert cache and cache[0].ml is not None
    first_ml = cache[0].ml
    # Same shape/spacing -> same sparsity pattern -> the hierarchy must be reused.
    o.solve(Field(rng.normal(size=(16, 16)), spacing=1.0))
    assert o._amg_cache[0].ml is first_ml
