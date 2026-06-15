import numpy as np
import pytest

from morphos.field import Field
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.parametric import ParametricOptimizer
from morphos.optimize.optimizer import OptimizeResult
from morphos.spec import ParametricSpec, DesignResult
from morphos.engine import Engine


def sphere_builder():
    kernel = VoxelKernel(grid_shape=(7, 7, 7), spacing=1.0)

    def build(params):
        return kernel.build(
            {"primitive": "sphere", "center": (3, 3, 3), "radius": float(params[0])}
        )

    return build


def target_at_radius(build, radius):
    return build(np.array([radius])).values


def test_parametric_optimizer_recovers_a_known_radius():
    build = sphere_builder()
    target = target_at_radius(build, 1.5)
    oracle = AnalyticOracle(target=target)
    opt = ParametricOptimizer(max_iter=2000, tol=1e-14)
    res = opt.run(
        initial_params=np.array([0.5]),
        build=build,
        oracle=oracle,
        objective=MaximizeValue(),
    )
    assert isinstance(res, OptimizeResult)
    assert res.params[0] == pytest.approx(1.5, abs=1e-3)
    assert res.used_finite_differences is True
    assert isinstance(res.field, Field)


def test_engine_runs_parametric_spec_and_reports_margin():
    build = sphere_builder()
    target = target_at_radius(build, 1.5)
    spec = ParametricSpec(
        initial_params=np.array([0.5]),
        build=build,
        oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="shape-match")),
        optimizer=ParametricOptimizer(max_iter=2000, tol=1e-14),
        name="sphere-radius-recovery",
    )
    res = Engine().run(spec)
    assert isinstance(res, DesignResult)
    assert res.figure_of_merit == pytest.approx(0.0, abs=1e-4)
    assert res.margin == pytest.approx(0.0, abs=1e-4)
    assert res.field.shape == (7, 7, 7)
