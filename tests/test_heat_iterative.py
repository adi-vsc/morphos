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
