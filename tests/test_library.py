"""Tests for the component library registry and channel templates."""

import numpy as np
import pytest

from morphos.field import Field
from morphos.library import (
    ComponentLibrary,
    ComponentSpec,
    default_library,
    straight_channel,
)


def _grid(shape=(10, 12)):
    return Field(np.zeros(shape), spacing=1.0)


def test_straight_channel_is_open_stripe_in_solid():
    f = straight_channel({"width_voxels": 4, "orientation": "x"}, _grid((10, 12)))
    vals = f.values
    assert vals.shape == (10, 12)
    # centre rows are open fluid (1), edges are solid (0)
    assert vals[5, :].mean() > 0.9
    assert vals[0, :].mean() < 0.1
    assert int(vals.sum()) == 4 * 12  # 4 open rows across full width


def test_straight_channel_orientation_y():
    f = straight_channel({"width_voxels": 2, "orientation": "y"}, _grid((8, 8)))
    assert f.values[:, 4].mean() > 0.9
    assert f.values[:, 0].mean() < 0.1


def test_library_register_and_build():
    lib = ComponentLibrary()
    lib.register("strip", straight_channel, oracle_types=["StokesFlowOracle"])
    assert "strip" in lib.names()
    f = lib.build("strip", {"width_voxels": 3, "orientation": "x"}, _grid())
    assert isinstance(f, Field)
    assert f.values.max() == 1.0


def test_library_rejects_duplicate_registration():
    lib = ComponentLibrary()
    lib.register("strip", straight_channel)
    with pytest.raises(ValueError):
        lib.register("strip", straight_channel)


def test_library_build_unknown_raises():
    with pytest.raises(KeyError):
        ComponentLibrary().build("nope", {}, _grid())


def test_list_compatible_filters_by_oracle_type():
    lib = default_library()
    flow = lib.list_compatible("StokesFlowOracle")
    assert "straight_channel" in flow and "serpentine_channel" in flow
    # an oracle the channels are not tagged for sees none of them
    assert lib.list_compatible("ElasticityOracle") == []


def test_default_library_spec_carries_metadata():
    lib = default_library()
    spec = lib.spec("straight_channel")
    assert isinstance(spec, ComponentSpec)
    assert "pressure_drop_Pa" in spec.expected_performance
