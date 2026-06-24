"""Tests for weighted-sum multi-objective scalarisation and the Pareto archive."""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue
from morphos.objective.multi_objective import MultiObjective, ObjectiveVector
from morphos.optimize.topopt import TopologyOptimizer


def _two_target_problem(shape=(5, 5)):
    rng = np.random.default_rng(0)
    t1 = rng.uniform(size=shape)
    t2 = rng.uniform(size=shape)
    oracles = [AnalyticOracle(target=t1), AnalyticOracle(target=t2)]
    objectives = [MaximizeValue(), MaximizeValue()]
    return objectives, oracles, (t1, t2)


def test_two_objective_scalar_fom_is_weighted_sum():
    objectives, oracles, _ = _two_target_problem()
    mo = MultiObjective(objectives, oracles, weights=[0.25, 0.75])
    field = Field(np.zeros((5, 5)), spacing=1.0)
    ov = mo.evaluate_design(field)
    assert isinstance(ov, ObjectiveVector)
    assert ov.values.shape == (2,)
    assert ov.scalar_fom == pytest.approx(0.25 * ov.values[0] + 0.75 * ov.values[1])
    # scalar gradient is the weighted sum of the component gradients
    expected = 0.25 * ov.gradient[0] + 0.75 * ov.gradient[1]
    assert np.allclose(ov.scalar_grad, expected)


def test_pareto_archive_adds_nondominated():
    mo = MultiObjective(*_two_target_problem()[:2], weights=[0.5, 0.5])
    mo.update_pareto_archive(np.array([1.0, 0.0]))
    mo.update_pareto_archive(np.array([0.0, 1.0]))  # neither dominates the other
    assert len(mo.pareto_archive) == 2


def test_pareto_archive_prunes_dominated():
    mo = MultiObjective(*_two_target_problem()[:2], weights=[0.5, 0.5])
    mo.update_pareto_archive(np.array([1.0, 1.0]))
    mo.update_pareto_archive(np.array([2.0, 2.0]))  # dominates the first
    assert len(mo.pareto_archive) == 1
    assert np.allclose(mo.pareto_archive[0][1], [2.0, 2.0])


def test_is_dominated_semantics():
    assert MultiObjective.is_dominated(np.array([1.0, 1.0]), np.array([2.0, 1.0]))
    assert not MultiObjective.is_dominated(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
    assert not MultiObjective.is_dominated(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_optimizer_with_multi_objective_improves_scalar_fom():
    objectives, oracles, _ = _two_target_problem()
    mo = MultiObjective(objectives, oracles, weights=[0.5, 0.5])
    opt = TopologyOptimizer(step_size=0.3, max_iter=200, tol=1e-12)
    result = opt.run(Field(np.zeros((5, 5)), spacing=1.0), oracle=None, objective=mo)
    # the scalarised figure of merit must improve over the run
    assert result.history[-1] > result.history[0]
    assert result.fom >= max(result.history)


def test_pareto_archive_grows_with_different_weights():
    objectives, oracles, _ = _two_target_problem()
    mo = MultiObjective(objectives, oracles, weights=[0.9, 0.1])
    opt = TopologyOptimizer(step_size=0.3, max_iter=60, tol=1e-12)
    field = Field(np.zeros((5, 5)), spacing=1.0)
    opt.run(field, oracle=None, objective=mo)
    n_after_first = len(mo.pareto_archive)
    mo.update_weights([0.1, 0.9])
    opt.run(field, oracle=None, objective=mo)
    assert len(mo.pareto_archive) >= n_after_first >= 1
