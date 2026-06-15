import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.manufacturing.constraints import Connectivity
from morphos.spec import DesignSpec, DesignResult
from morphos.engine import Engine


def connected_target():
    t = np.zeros((5, 5))
    t[1:4, 1:4] = 1.0
    return t


def base_spec(target, bound=None, constraint=None):
    return DesignSpec(
        initial=Field(np.zeros(target.shape), spacing=1.0),
        oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(bound=bound),
        optimizer=TopologyOptimizer(step_size=0.2, max_iter=5000, tol=1e-12),
        constraint=constraint,
        name="analytic-target-match",
    )


def test_engine_recovers_target_and_reports_zero_margin():
    target = connected_target()
    spec = base_spec(target, bound=PhysicalBound(value=0.0, name="target-match"))
    res = Engine().run(spec)
    assert isinstance(res, DesignResult)
    assert res.figure_of_merit == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.field.values, target, atol=1e-3)
    assert res.converged is True
    assert res.margin == pytest.approx(0.0, abs=1e-6)
    # ceiling is zero, so a fraction of the ceiling is undefined
    assert res.attained_fraction is None


def test_engine_reports_manufacturability():
    target = connected_target()
    spec = base_spec(target, constraint=Connectivity(threshold=0.5))
    res = Engine().run(spec)
    assert res.manufacturability["connected"] is True
    assert res.manufacturability["num_components"] == 1


def test_engine_without_bound_has_none_margin():
    target = connected_target()
    res = Engine().run(base_spec(target))
    assert res.bound is None
    assert res.margin is None
    assert res.attained_fraction is None


def test_engine_margin_and_fraction_with_nonzero_ceiling():
    target = connected_target()
    spec = base_spec(target, bound=PhysicalBound(value=2.0, name="contrived"))
    res = Engine().run(spec)
    # fom converges to ~0, ceiling is 2.0
    assert res.margin == pytest.approx(2.0, abs=1e-5)
    assert res.attained_fraction == pytest.approx(0.0, abs=1e-5)
