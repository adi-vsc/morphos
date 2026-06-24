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


def _stokes_parabola(x):
    y = x[1]
    Ly = y.max()
    return np.stack([4.0 * y * (Ly - y) / (Ly**2), np.zeros_like(y)])


def _coupled_heat_exchanger_spec(shape=(4, 8), n_outer=1):
    pytest.importorskip("skfem")
    from morphos.physics.stokes import StokesFlowOracle
    from morphos.physics.conjugate_heat import ConjugateHeatOracle
    from morphos.spec import CoupledSpec

    nely, nelx = shape
    nny, nnx = nely + 1, nelx + 1
    inlet = [iy * nnx for iy in range(nny)]

    stokes = StokesFlowOracle(
        shape=shape, inlet=("left", _stokes_parabola), noslip_edges=("top", "bottom")
    )
    cht = ConjugateHeatOracle(
        shape=shape,
        velocity=np.zeros((nny, nnx, 2)),  # placeholder, overwritten by passthrough
        source=np.ones(shape),
        fixed_nodes=inlet,
        fixed_values=[0.0] * len(inlet),
        k_solid=1.0, k_fluid=1.0, rho_cp=1.0,
    )
    opt = TopologyOptimizer(step_size=1e-4, max_iter=2, bounds=(0.0, 1.0))
    return CoupledSpec(
        stages=[
            (stokes, MaximizeValue(), None),
            (cht, MaximizeValue(), None),
        ],
        optimizer=opt,
        initial_field=Field(np.ones(shape), spacing=1.0),
        passthrough={0: ["velocity"]},
        n_outer=n_outer,
    ), stokes, cht


def test_coupled_engine_passthrough_forwards_velocity_field():
    from morphos.engine import CoupledEngine

    spec, stokes, cht = _coupled_heat_exchanger_spec()
    results = CoupledEngine().run(spec)
    assert len(results) == 2
    # The CHT oracle received the Stokes velocity field by passthrough.
    assert cht.velocity.shape == (spec.initial_field.values.shape[0] + 1,
                                  spec.initial_field.values.shape[1] + 1, 2)
    assert np.abs(cht.velocity).max() > 0.1  # nonzero flow forwarded
    # It equals the velocity Stokes solves on the final field.
    final_field = Field(results[0].field.values, spacing=1.0)
    stokes_vel = stokes.solve(final_field).aux["velocity"]
    assert np.allclose(cht.velocity, stokes_vel)


def test_coupled_engine_stokes_then_cht_reduces_thermal_objective():
    from morphos.engine import CoupledEngine

    spec, stokes, cht = _coupled_heat_exchanger_spec()
    results = CoupledEngine().run(spec)
    cht_result = results[1]
    mean_T_flow = -cht_result.figure_of_merit  # value = -mean temperature

    # No-flow baseline on the same final field: convection removes heat, so the
    # coupled (with-flow) mean temperature must be lower.
    final_field = Field(cht_result.field.values, spacing=1.0)
    cht.velocity = np.zeros_like(cht.velocity)
    cht._cache_h = None
    mean_T_noflow = -cht.solve(final_field).value
    assert mean_T_flow < mean_T_noflow


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
