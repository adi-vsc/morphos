import numpy as np
import pytest

from morphos.field import Field
from morphos.geometry.kernel import GeometryKernel
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.geometry.picogk import PicoGKKernel


def make_kernel():
    return VoxelKernel(grid_shape=(5, 5, 5), spacing=1.0)


def test_voxel_kernel_is_a_geometry_kernel():
    assert isinstance(make_kernel(), GeometryKernel)


def test_sphere_sdf_known_values():
    k = make_kernel()
    f = k.build({"primitive": "sphere", "center": (2, 2, 2), "radius": 1.0})
    assert isinstance(f, Field)
    # center of the sphere: signed distance is minus the radius
    assert f.values[2, 2, 2] == pytest.approx(-1.0)
    # two cells away along x: distance 2, minus radius 1, equals +1
    assert f.values[4, 2, 2] == pytest.approx(1.0)


def test_sphere_sign_convention_negative_inside():
    k = make_kernel()
    f = k.build({"primitive": "sphere", "center": (2, 2, 2), "radius": 1.5})
    inside = f.values < 0
    # the center cell is inside
    assert inside[2, 2, 2]
    # a far corner is outside
    assert not inside[0, 0, 0]


def test_box_sdf_inside_outside():
    k = make_kernel()
    f = k.build(
        {"primitive": "box", "center": (2, 2, 2), "half_extent": (1.0, 1.0, 1.0)}
    )
    assert f.values[2, 2, 2] < 0  # interior
    assert f.values[0, 0, 0] > 0  # exterior corner


def test_slab_sdf_along_axis():
    k = make_kernel()
    f = k.build({"primitive": "slab", "axis": 0, "lo": 1.0, "hi": 3.0})
    # index 2 (coord 2) is inside [1, 3]
    assert f.values[2, 0, 0] < 0
    # index 0 (coord 0) is below lo
    assert f.values[0, 0, 0] > 0


def test_unknown_primitive_raises():
    k = make_kernel()
    with pytest.raises(ValueError):
        k.build({"primitive": "torus"})


def test_field_grid_matches_kernel():
    k = VoxelKernel(grid_shape=(3, 4), spacing=(0.5, 2.0))
    f = k.build({"primitive": "sphere", "center": (0.5, 2.0), "radius": 0.5})
    assert f.shape == (3, 4)
    assert f.spacing == (0.5, 2.0)


def test_picogk_backend_is_explicit_not_silent():
    k = PicoGKKernel(grid_shape=(4, 4, 4), spacing=1.0)
    assert isinstance(k, GeometryKernel)
    # explicit failure (unknown primitive, or no native runtime), never silent wrong geometry
    with pytest.raises((ValueError, RuntimeError)):
        k.build({"primitive": "torus"})
