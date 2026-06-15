import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import TopologyOptimizer


class NoGradientOracle(PhysicsOracle):
    """Same response as AnalyticOracle but withholds the gradient."""

    provides_gradient = False

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def solve(self, field):
        diff = field.values - self.target
        return PhysicsResult(value=-float(np.sum(diff ** 2)), gradient=None)


def test_topology_optimizer_is_an_optimizer():
    assert isinstance(TopologyOptimizer(), Optimizer)


def test_recovers_target_with_gradient():
    target = np.array([[0.2, -0.4], [0.9, 0.1]])
    oracle = AnalyticOracle(target=target)
    obj = MaximizeValue()
    opt = TopologyOptimizer(step_size=0.2, max_iter=5000, tol=1e-12)
    res = opt.run(Field(np.zeros((2, 2)), spacing=1.0), oracle, obj)
    assert isinstance(res, OptimizeResult)
    assert res.fom == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.field.values, target, atol=1e-3)
    assert res.used_finite_differences is False


def test_objective_improves_monotonically_on_convex_problem():
    target = np.array([[1.0, 2.0]])
    opt = TopologyOptimizer(step_size=0.2, max_iter=200, tol=0.0)
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), AnalyticOracle(target), MaximizeValue()
    )
    assert res.history[-1] >= res.history[0]


def test_finite_difference_fallback_when_no_gradient():
    target = np.array([[0.3, -0.2]])
    opt = TopologyOptimizer(step_size=0.2, max_iter=4000, tol=1e-12)
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), NoGradientOracle(target), MaximizeValue()
    )
    assert res.used_finite_differences is True
    assert np.allclose(res.field.values, target, atol=1e-2)


def test_bounds_are_respected():
    target = np.array([[5.0, 5.0]])  # outside [0, 1]
    opt = TopologyOptimizer(step_size=0.2, max_iter=500, bounds=(0.0, 1.0))
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), AnalyticOracle(target), MaximizeValue()
    )
    assert np.all(res.field.values <= 1.0 + 1e-9)
    assert np.all(res.field.values >= 0.0 - 1e-9)
