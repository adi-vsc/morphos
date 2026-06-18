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
