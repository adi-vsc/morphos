"""Unit tests for resampling a PicoGK voxel block onto a different grid spacing,
with no native dependency.

PicoGK builds at whatever voxel size its ``Library`` instance was created
with; the kernel grid the engine actually wants may ask for a different
(but still isotropic) spacing. ``resample_volume`` trilinearly resamples the
solid block PicoGK returned (still in PicoGK's own voxel units) onto the
kernel's requested spacing, so ``PicoGKKernel.build`` no longer needs the two
to match exactly.

This synthesises a solid signed-distance block from an analytic sphere (no
DLL needed), so the resampling math is verified everywhere, including CI
without PicoGK.
"""

import numpy as np
import pytest

from morphos.geometry import _picogk_native as pk


def _analytic_sphere_block(shape, center, radius):
    idx = np.indices(shape).astype(float)
    return np.sqrt(sum((idx[a] - center[a]) ** 2 for a in range(len(shape)))) - radius


def test_resample_volume_preserves_shape_scaling():
    # PicoGK voxel = 0.5mm, kernel wants 0.25mm: each PicoGK voxel becomes 2.
    volume = np.zeros((10, 12, 14), dtype=float)
    out = pk.resample_volume(volume, zoom_factor=2.0)
    assert out.shape == (20, 24, 28)


def test_resample_volume_downsamples_too():
    volume = np.zeros((20, 20, 20), dtype=float)
    out = pk.resample_volume(volume, zoom_factor=0.5)
    assert out.shape == (10, 10, 10)


def test_resample_volume_matches_analytic_sphere_after_upsampling():
    shape = (30, 30, 30)
    center = (15.0, 15.0, 15.0)
    radius = 9.0
    block = _analytic_sphere_block(shape, center, radius)

    zoom_factor = 2.0  # PicoGK voxel 0.5mm -> kernel wants 0.25mm
    resampled = pk.resample_volume(block, zoom_factor)

    # resample_volume only changes array density (more samples of the same
    # PicoGK-voxel-unit field); output index i sits at input-space position
    # i / zoom_factor, so the analytic sphere's center/radius stay in PicoGK
    # voxel units while indices are descaled to compare.
    rs_shape = resampled.shape
    idx = np.indices(rs_shape).astype(float) / zoom_factor
    expected = np.sqrt(sum((idx[a] - center[a]) ** 2 for a in range(3))) - radius

    # away from the surface, sign must match; values should track closely too
    away = np.abs(expected) > 2.0
    assert np.array_equal(resampled[away] < 0, expected[away] < 0)
    assert np.allclose(resampled[away], expected[away], atol=1.0)


def test_resample_volume_is_a_no_op_at_zoom_factor_one():
    volume = np.random.default_rng(0).normal(size=(8, 8, 8))
    out = pk.resample_volume(volume, zoom_factor=1.0)
    assert np.allclose(out, volume)


def test_resample_origin_scales_with_zoom_factor():
    # The PicoGK active-block origin is in PicoGK voxel units; placing the
    # resampled block on the kernel grid needs that origin in kernel-voxel
    # units, i.e. scaled by the same zoom factor as the block itself.
    origin = (4, -2, 7)
    zoom_factor = 2.0
    assert pk.resample_origin(origin, zoom_factor) == (8, -4, 14)


def test_resample_origin_rounds_to_nearest_voxel():
    origin = (3, 1, 0)
    zoom_factor = 0.5
    # 3*0.5=1.5 rounds to 2 (round-half-to-even is fine; just must be an int
    # within 1 voxel of the exact scaled position)
    out = pk.resample_origin(origin, zoom_factor)
    assert all(isinstance(o, int) for o in out)
    exact = (1.5, 0.5, 0.0)
    assert all(abs(o - e) <= 1 for o, e in zip(out, exact))


def test_resampled_solid_volume_scales_with_voxel_count_change():
    """End-to-end-ish check: resampling a solid mask preserves physical volume.

    A solid sphere (interior < 0), resampled from PicoGK's native grid onto a
    grid with smaller (kernel-requested) voxels, should report the same
    physical volume once each grid's own voxel size (in mm) is accounted for
    -- i.e. ``voxel_count * voxel_size_mm**3`` is conserved up to the surface's
    interpolation error. PicoGK's voxel size shrinks to the kernel's voxel size
    by the same ``zoom_factor`` resample_volume used to grow the array.
    """
    shape = (40, 40, 40)
    center = (20.0, 20.0, 20.0)
    radius = 12.0
    picogk_voxel_mm = 0.5
    block = _analytic_sphere_block(shape, center, radius)

    zoom_factor = 2.0  # kernel wants a finer grid: half PicoGK's voxel size
    kernel_voxel_mm = picogk_voxel_mm / zoom_factor
    fine = pk.resample_volume(block, zoom_factor)

    coarse_volume_mm3 = np.sum(block < 0) * picogk_voxel_mm**3
    fine_volume_mm3 = np.sum(fine < 0) * kernel_voxel_mm**3
    analytic_volume_mm3 = (4.0 / 3.0) * np.pi * (radius * picogk_voxel_mm) ** 3

    assert coarse_volume_mm3 == pytest.approx(analytic_volume_mm3, rel=0.05)
    assert fine_volume_mm3 == pytest.approx(analytic_volume_mm3, rel=0.05)
