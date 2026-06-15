import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle


def fd_gradient(oracle, field, eps=1e-6):
    base = oracle.solve(field).value
    grad = np.zeros_like(field.values)
    it = np.nditer(field.values, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        perturbed = field.copy()
        perturbed.values[idx] += eps
        grad[idx] = (oracle.solve(perturbed).value - base) / eps
        it.iternext()
    return grad


def test_analytic_oracle_is_a_physics_oracle():
    o = AnalyticOracle(target=np.zeros((3, 3)))
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_solve_returns_physics_result():
    o = AnalyticOracle(target=np.zeros((2, 2)))
    f = Field(np.ones((2, 2)), spacing=1.0)
    r = o.solve(f)
    assert isinstance(r, PhysicsResult)
    # value = -sum((1-0)^2) = -4
    assert r.value == pytest.approx(-4.0)


def test_value_matches_weighted_sse():
    target = np.array([[1.0, 2.0]])
    o = AnalyticOracle(target=target, weight=2.0)
    f = Field(np.array([[1.0, 0.0]]), spacing=1.0)
    # -sum(2 * (f-t)^2) = -2*(0 + 4) = -8
    assert o.solve(f).value == pytest.approx(-8.0)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(3, 3))
    o = AnalyticOracle(target=target, weight=1.3)
    f = Field(rng.normal(size=(3, 3)), spacing=1.0)
    analytic = o.solve(f).gradient
    numeric = fd_gradient(o, f)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_target_shape_mismatch_raises():
    o = AnalyticOracle(target=np.zeros((2, 2)))
    f = Field(np.zeros((3, 3)), spacing=1.0)
    with pytest.raises(ValueError):
        o.solve(f)


def test_optimum_value_is_zero_at_target():
    target = np.array([[0.5, -0.5], [1.0, 2.0]])
    o = AnalyticOracle(target=target)
    f = Field(target.copy(), spacing=1.0)
    assert o.solve(f).value == pytest.approx(0.0)
