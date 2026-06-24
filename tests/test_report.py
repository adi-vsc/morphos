import numpy as np
import pytest

from morphos.field import Field
from morphos.engine import Engine
from morphos.intent import CantileverIntent, ChannelIntent
from morphos.report import PerformanceReport, build_report
from morphos.spec import DesignResult


def test_elasticity_report_has_engineering_quantities():
    intent = CantileverIntent(span=10, height=5, load=-1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert isinstance(report, PerformanceReport)
    assert report.quantities["compliance"] > 0.0
    assert report.quantities["max_displacement"] > 0.0
    assert 0.0 < report.mass_fraction <= 1.0
    assert report.figure_of_merit == pytest.approx(res.figure_of_merit)


def test_darcy_report_has_engineering_quantities():
    intent = ChannelIntent(span=10, height=5, source=1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert report.quantities["dissipation"] > 0.0
    assert report.quantities["peak_pressure"] > 0.0


def _result_for(field):
    return DesignResult(field=field, figure_of_merit=0.0)


def test_heat_report_has_engineering_quantities():
    from morphos.physics.heat import HeatConductionOracle

    shape = (6, 6)
    rng = np.random.default_rng(0)
    oracle = HeatConductionOracle(target=rng.normal(size=shape) * 0.1)
    field = Field(0.5 + 0.5 * rng.uniform(size=shape), spacing=1.0)
    report = build_report(_result_for(field), oracle)
    q = report.quantities
    assert "peak_temperature_K" in q and "mean_temperature_K" in q
    assert q["peak_temperature_K"] >= q["mean_temperature_K"]
    assert q["temperature_range_K"] >= 0.0


def test_stokes_report_has_engineering_quantities():
    pytest.importorskip("skfem")
    from morphos.physics.stokes import StokesFlowOracle

    def parabola(x):
        y = x[1]
        Ly = y.max()
        return np.stack([4.0 * y * (Ly - y) / (Ly**2), np.zeros_like(y)])

    shape = (6, 8)
    oracle = StokesFlowOracle(
        shape=shape, inlet=("left", parabola), noslip_edges=("top", "bottom")
    )
    report = build_report(_result_for(Field(np.ones(shape), spacing=1.0)), oracle)
    q = report.quantities
    assert q["viscous_dissipation_W"] > 0.0
    assert q["peak_velocity_ms"] > 0.0
    assert q["pressure_drop_Pa"] >= 0.0


def test_thermoelastic_report_has_engineering_quantities():
    from morphos.physics.thermoelastic import ThermoElasticOracle

    shape = (4, 6)
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    oracle = ThermoElasticOracle(
        shape=shape,
        fixed_dofs=[(0, j, ax) for j in range(nny) for ax in ("x", "y")],
        loads={(nnx - 1, nny // 2, "y"): -1.0},
        fixed_temps=[(0, j) for j in range(nny)],
        heat_sources={(nnx - 1, nny - 1): 1.0},
        thermal_expansion=1.0,
    )
    report = build_report(_result_for(Field(np.ones(shape), spacing=1.0)), oracle)
    q = report.quantities
    assert q["max_displacement"] > 0.0
    assert "peak_temperature_K" in q and "objective" in q


def test_modal_report_has_engineering_quantities():
    from morphos.physics.modal import ModalOracle

    shape = (7, 7)
    oracle = ModalOracle(shape=shape, mode=0)
    report = build_report(_result_for(Field(np.ones(shape), spacing=1.0)), oracle)
    q = report.quantities
    assert q["eigenvalue"] > 0.0
    assert q["fundamental_freq_Hz"] == pytest.approx(np.sqrt(q["eigenvalue"]) / (2 * np.pi))


def test_report_dispatch_raises_for_unknown_oracle():
    from morphos.physics.oracle import PhysicsOracle, PhysicsResult

    class MysteryOracle(PhysicsOracle):
        def solve(self, field):
            return PhysicsResult(value=0.0, gradient=None, aux={})

    with pytest.raises(TypeError):
        build_report(_result_for(Field(np.ones((4, 4)), spacing=1.0)), MysteryOracle())


def test_report_carries_margin_when_bound_is_known():
    intent = CantileverIntent(span=10, height=5, load=-1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert report.margin == pytest.approx(res.margin)
    assert report.attained_fraction == res.attained_fraction
