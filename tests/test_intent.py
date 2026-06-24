import numpy as np
import pytest

from morphos.spec import DesignSpec
from morphos.engine import Engine
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.darcy import DarcyFlowOracle
from morphos.intent import CantileverIntent, ChannelIntent


def test_cantilever_intent_builds_a_design_spec():
    intent = CantileverIntent(span=8, height=4, load=-1.0, volume_fraction=0.4)
    spec = intent.build()
    assert isinstance(spec, DesignSpec)
    assert isinstance(spec.oracle, ElasticityOracle)
    assert spec.oracle.shape == (4, 8)
    assert spec.initial.values.shape == (4, 8)
    assert np.allclose(spec.initial.values, 0.4)


def test_cantilever_intent_runs_end_to_end_and_reduces_compliance():
    intent = CantileverIntent(span=20, height=10, load=-1.0, volume_fraction=0.4, max_iter=60)
    res = Engine().run(intent.build())
    assert res.history[-1] > res.history[0]
    assert res.figure_of_merit <= 0.0


def test_cantilever_intent_rejects_bad_volume_fraction():
    with pytest.raises(ValueError):
        CantileverIntent(span=8, height=4, load=-1.0, volume_fraction=1.5)


def test_channel_intent_builds_a_design_spec():
    intent = ChannelIntent(span=8, height=4, source=1.0, volume_fraction=0.4)
    spec = intent.build()
    assert isinstance(spec, DesignSpec)
    assert isinstance(spec.oracle, DarcyFlowOracle)
    assert spec.oracle.shape == (4, 8)


def test_channel_intent_runs_end_to_end_and_reduces_dissipation():
    intent = ChannelIntent(span=20, height=10, source=1.0, volume_fraction=0.4, max_iter=60)
    res = Engine().run(intent.build())
    assert res.history[-1] > res.history[0]
    assert res.figure_of_merit <= 0.0


def test_channel_intent_rejects_bad_volume_fraction():
    with pytest.raises(ValueError):
        ChannelIntent(span=8, height=4, source=1.0, volume_fraction=0.0)


# --- new intents (GAP 5) ---------------------------------------------------

def test_thermal_sink_intent_runs_and_reduces_mean_temperature():
    from morphos.intent import ThermalSinkIntent

    intent = ThermalSinkIntent(
        nx=8, ny=8, heat_source_W_m3=1.0, k_solid=2.0, sink_edge="left",
        volume_fraction=0.5, step_size=0.2, max_iter=40,
    )
    spec = intent.build()
    res = Engine().run(spec)
    # value = -mean temperature, maximised -> mean temperature falls
    assert res.history[-1] > res.history[0]


def test_stokes_channel_intent_runs_and_reduces_dissipation():
    pytest.importorskip("skfem")
    from morphos.intent import StokesBrinkmanChannelIntent

    intent = StokesBrinkmanChannelIntent(
        nx=8, ny=6, mu=1.0, u_max=1.0, volume_fraction=0.5,
    )
    spec = intent.build()
    init_diss = spec.oracle.solve(spec.initial).aux["dissipation"]
    res = Engine().run(spec)
    final_diss = spec.oracle.solve(res.field).aux["dissipation"]
    assert final_diss <= init_diss


def test_thermoelastic_intent_runs_end_to_end():
    from morphos.intent import ThermoElasticIntent

    intent = ThermoElasticIntent(nx=8, ny=6, load=-1.0, alpha_cte=1.0,
                                 volume_fraction=0.5, max_iter=40)
    spec = intent.build()
    res = Engine().run(spec)
    assert res.history[-1] >= res.history[0]
    assert res.figure_of_merit <= 0.0


def test_heat_exchanger_intent_builds_coupled_spec_and_runs():
    pytest.importorskip("skfem")
    from morphos.intent import HeatExchangerIntent
    from morphos.engine import CoupledEngine
    from morphos.spec import CoupledSpec

    intent = HeatExchangerIntent(nx=8, ny=4, u_max=1.0, heat_source_W_m3=1.0)
    spec = intent.build()
    assert isinstance(spec, CoupledSpec)
    results = CoupledEngine().run(spec)
    assert len(results) == 2
    # stage 1 (CHT) received the Stokes velocity by passthrough
    assert np.abs(spec.stages[1][0].velocity).max() > 0.1
