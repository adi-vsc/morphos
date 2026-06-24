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
from morphos.manufacturing.constraints import Connectivity


def _grid(shape=(10, 12)):
    return Field(np.zeros(shape), spacing=1.0)


def _grid3d(shape=(20, 20, 20)):
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
    # the channels are flow-only, so they are absent for a structural oracle
    structural_only = lib.list_compatible("ElasticityOracle")
    assert "straight_channel" not in structural_only
    assert "serpentine_channel" not in structural_only
    # an oracle nothing is tagged for sees no components at all
    assert lib.list_compatible("ModalOracle") == []


def test_default_library_spec_carries_metadata():
    lib = default_library()
    spec = lib.spec("straight_channel")
    assert isinstance(spec, ComponentSpec)
    assert "pressure_drop_Pa" in spec.expected_performance


# ---------------------------------------------------------------------------
# Step 8: fins, manifolds, structural ribs
# ---------------------------------------------------------------------------

from morphos.library.fins import corrugated_fin, pin_fin_array, plate_fin_array
from morphos.library.manifolds import tree_manifold, y_manifold
from morphos.library.structural import cross_rib, honeycomb_rib, i_beam_rib


def test_new_components_registered_in_default_library():
    lib = default_library()
    for name in (
        "pin_fin_array", "plate_fin_array", "corrugated_fin",
        "y_manifold", "tree_manifold",
        "cross_rib", "i_beam_rib", "honeycomb_rib",
    ):
        assert name in lib.names()


def test_list_compatible_filters_by_oracle_type_includes_new_components():
    lib = default_library()
    heat = lib.list_compatible("ConjugateHeatOracle")
    assert {"pin_fin_array", "plate_fin_array", "corrugated_fin"} <= set(heat)

    flow = lib.list_compatible("DarcyFlowOracle")
    assert {"y_manifold", "tree_manifold"} <= set(flow)

    structural = lib.list_compatible("ElasticityOracle")
    assert {"cross_rib", "i_beam_rib", "honeycomb_rib"} <= set(structural)


def test_pin_fin_array_is_solid_stripe():
    f = pin_fin_array({"pitch_voxels": 4, "height_voxels": 6, "thickness_voxels": 1}, _grid((20, 20)))
    assert isinstance(f, Field)
    assert f.shape == (20, 20)
    assert f.values.max() == pytest.approx(1.0)
    assert f.values.min() == pytest.approx(0.0)


def test_pin_fin_array_produces_expected_fin_count():
    pitch = 5
    grid = _grid((20, 20))
    f = pin_fin_array({"pitch_voxels": pitch, "height_voxels": 6, "thickness_voxels": 1}, grid)
    # count solid stripes along the base row (y=0): one fin per pitch
    base_row = f.values[0, :]
    # rising edges (0 -> 1) count number of separate fins
    rising = np.sum((base_row[1:] > 0.5) & (base_row[:-1] <= 0.5))
    if base_row[0] > 0.5:
        rising += 1
    expected = grid.shape[1] // pitch
    assert rising == pytest.approx(expected, abs=1)


def test_plate_fin_array_is_solid_stripe():
    f = plate_fin_array({"pitch_voxels": 4, "thickness_voxels": 1}, _grid((20, 20)))
    assert isinstance(f, Field)
    assert f.values.max() == pytest.approx(1.0)
    assert set(np.unique(f.values)) <= {0.0, 1.0}


def test_corrugated_fin_is_nonempty_and_bounded():
    f = corrugated_fin({"pitch_voxels": 6, "amplitude_voxels": 3, "thickness_voxels": 1}, _grid((24, 24)))
    assert isinstance(f, Field)
    assert f.values.shape == (24, 24)
    assert f.values.min() >= 0.0
    assert f.values.max() <= 1.0
    assert f.values.sum() > 0.0


def test_y_manifold_is_valid_nonempty_field():
    grid = _grid3d((24, 24, 24))
    f = y_manifold({"inlet_radius": 3.0, "outlet_radius": 2.0}, grid)
    assert isinstance(f, Field)
    assert f.shape == grid.shape
    assert (f.values < 0).any()  # has solid/open interior somewhere


def test_tree_manifold_is_connected():
    grid = _grid3d((32, 32, 32))
    f = tree_manifold({"depth": 2, "inlet_radius": 3.0}, grid)
    assert isinstance(f, Field)
    report = Connectivity(threshold=0.0).report(f.like(-f.values))  # solid where SDF < 0
    assert report["connected"]


def test_cross_rib_is_density_field_in_range():
    f = cross_rib({"thickness_voxels": 2}, _grid((20, 20)))
    assert isinstance(f, Field)
    assert f.values.min() >= 0.0
    assert f.values.max() <= 1.0
    assert f.values.sum() > 0.0


def test_i_beam_rib_is_density_field_in_range():
    f = i_beam_rib({"flange_voxels": 3, "web_voxels": 2}, _grid((20, 20)))
    assert isinstance(f, Field)
    assert f.values.min() >= 0.0
    assert f.values.max() <= 1.0
    assert f.values.sum() > 0.0


def test_honeycomb_rib_volume_fraction_is_sane():
    f = honeycomb_rib({"cell_size_voxels": 6, "wall_voxels": 1}, _grid((40, 40)))
    assert isinstance(f, Field)
    fraction = f.values.mean()
    # a honeycomb wall pattern should be a clear minority of the area but not empty
    assert 0.05 < fraction < 0.6
