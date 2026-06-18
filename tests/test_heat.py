import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.heat import HeatConductionOracle
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.spec import DesignSpec
from morphos.engine import Engine


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


def test_heat_oracle_is_a_physics_oracle():
    o = HeatConductionOracle(target=np.zeros((5, 5)))
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_accepts_2d_and_3d_but_rejects_lower():
    # 2D and 3D grids are both supported now
    HeatConductionOracle(target=np.zeros((4, 4)))
    HeatConductionOracle(target=np.zeros((4, 4, 4)))
    # a 1D grid has no interior cross-section and is rejected
    with pytest.raises(ValueError):
        HeatConductionOracle(target=np.zeros(4))


def test_positive_source_makes_positive_interior_temperature():
    o = HeatConductionOracle(target=np.zeros((5, 5)))
    s = np.zeros((5, 5))
    s[2, 2] = 1.0
    r = o.solve(Field(s, spacing=1.0))
    T = r.aux["temperature"]
    assert T[2, 2] > 0.0
    # boundary is held at zero
    assert T[0, 0] == pytest.approx(0.0)


def test_solution_satisfies_the_linear_system():
    o = HeatConductionOracle(target=np.zeros((5, 5)))
    rng = np.random.default_rng(3)
    s = rng.normal(size=(5, 5))
    r = o.solve(Field(s, spacing=1.0))
    residual = o.residual(s, r.aux["temperature"])
    assert np.max(np.abs(residual)) < 1e-8


def test_adjoint_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(5, 5)) * 0.1
    o = HeatConductionOracle(target=target, weight=1.0)
    f = Field(rng.normal(size=(5, 5)) * 0.1, spacing=1.0)
    analytic = o.solve(f).gradient
    numeric = central_fd_gradient(o, f)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_engine_drives_real_pde_and_improves_with_adjoint():
    # Build a reachable target temperature from a known source.
    probe = HeatConductionOracle(target=np.zeros((5, 5)))
    s0 = np.zeros((5, 5))
    s0[2, 2] = 1.0
    s0[1, 3] = 0.5
    t_target = probe.solve(Field(s0, spacing=1.0)).aux["temperature"]

    spec = DesignSpec(
        initial=Field(np.zeros((5, 5)), spacing=1.0),
        oracle=HeatConductionOracle(target=t_target),
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="temp-match")),
        optimizer=TopologyOptimizer(step_size=0.05, max_iter=4000, tol=1e-14),
        name="heat-source-recovery",
    )
    res = Engine().run(spec)
    assert res.used_finite_differences is False  # adjoint was used
    assert res.history[-1] > res.history[0]      # objective improved
    assert res.figure_of_merit > -1e-3           # close to the ceiling of 0
