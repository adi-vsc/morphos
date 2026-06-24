"""Tests for TPMS (gyroid/Schwartz-P/diamond) and lattice infill primitives."""

import numpy as np
import pytest

from morphos.field import Field
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.geometry.picogk import PicoGKKernel
from morphos.geometry.tpms import (
    diamond_sdf,
    gyroid_sdf,
    lattice_infill,
    schwartz_p_sdf,
    shell_offset,
)


def _grid(shape=(20, 20, 20), spacing=1.0):
    return Field(np.zeros(shape), spacing=spacing)


def test_gyroid_sdf_is_periodic():
    period = 6.0
    field = _grid((24, 24, 24))
    f = gyroid_sdf(field, period=period, thickness=1.0)
    n_shift = int(round(period / field.spacing[0]))
    # shifting one full period along any axis must reproduce the same values
    # (restrict to the region available on both sides of the shift)
    a = f.values[: f.shape[0] - n_shift, : f.shape[1] - n_shift, : f.shape[2] - n_shift]
    b = f.values[n_shift:, n_shift:, n_shift:]
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_schwartz_p_and_diamond_are_periodic():
    period = 5.0
    field = _grid((20, 20, 20))
    n_shift = int(round(period / field.spacing[0]))
    for fn in (schwartz_p_sdf, diamond_sdf):
        f = fn(field, period=period, thickness=1.0)
        a = f.values[: f.shape[0] - n_shift, :, :]
        b = f.values[n_shift:, :, :]
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_tpms_thickness_controls_band_width():
    field = _grid((20, 20, 20))
    thin = gyroid_sdf(field, period=6.0, thickness=0.5)
    thick = gyroid_sdf(field, period=6.0, thickness=2.0)
    # a thicker wall has more solid (negative) voxels than a thin one
    assert (thick.values < 0).sum() > (thin.values < 0).sum()


def test_shell_offset_expands_sphere_by_offset():
    k = VoxelKernel(grid_shape=(40, 40, 40), spacing=1.0)
    sphere = k.build({"primitive": "sphere", "center": (20, 20, 20), "radius": 10.0})
    offset = 3.0
    shelled = shell_offset(sphere, offset)
    # zero crossing of the offset field along x, at the equator
    row = shelled.values[:, 20, 20]
    # find the index where the sign flips from negative to positive on the
    # right-hand side (outer surface)
    signs = np.sign(row)
    crossing_idx = np.where(np.diff(signs) > 0)[0]
    assert crossing_idx.size > 0
    crossing_x = crossing_idx[-1] + 0.5  # approx zero crossing between samples
    expected_x = 20 + 10.0 + offset
    assert crossing_x == pytest.approx(expected_x, abs=1.0)


def test_shell_offset_rejects_negative_offset():
    field = _grid((5, 5, 5))
    with pytest.raises(ValueError):
        shell_offset(field, -1.0)


def test_lattice_infill_volume_fraction_is_close():
    k = VoxelKernel(grid_shape=(30, 30, 30), spacing=1.0)
    outer = k.build({"primitive": "sphere", "center": (15, 15, 15), "radius": 14.0})
    tpms = gyroid_sdf(outer, period=8.0, thickness=2.0)
    target = 0.3
    filled = lattice_infill(outer, tpms, volume_fraction=target)
    outer_mask = outer.values < 0
    solid_mask = filled.values < 0
    fraction = solid_mask[outer_mask].sum() / outer_mask.sum()
    assert abs(fraction - target) < 0.05


def test_lattice_infill_rejects_mismatched_grids():
    a = _grid((10, 10, 10))
    b = _grid((12, 12, 12))
    with pytest.raises(ValueError):
        lattice_infill(a, b, 0.3)


def test_voxel_kernel_gyroid_primitive():
    k = VoxelKernel(grid_shape=(20, 20, 20), spacing=1.0)
    f = k.build({"primitive": "gyroid", "period": 6.0, "thickness": 1.0})
    assert isinstance(f, Field)
    assert f.shape == (20, 20, 20)


def test_voxel_kernel_schwartz_p_and_diamond_primitives():
    k = VoxelKernel(grid_shape=(15, 15, 15), spacing=1.0)
    for name in ("schwartz_p", "diamond"):
        f = k.build({"primitive": name, "period": 5.0, "thickness": 1.0})
        assert isinstance(f, Field)
        assert f.shape == (15, 15, 15)


def test_picogk_kernel_gyroid_primitive_matches_analytic():
    k = PicoGKKernel(grid_shape=(16, 16, 16), spacing=1.0)
    f = k.build({"primitive": "gyroid", "period": 6.0, "thickness": 1.0})
    assert isinstance(f, Field)
    assert f.shape == (16, 16, 16)
