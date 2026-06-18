"""Spike-level checks for the native PicoGK ctypes binding.

These confirm the interop that the spike proved: Python can drive the native
PicoGK runtime (no .NET layer) and pull a coherent narrow-band signed-distance
field into numpy. They assert only what is verified -- the library loads, a
sphere voxel field is created, its bounding box is geometrically sane, and the
returned field is a narrow band with both signs clamped at +/- the background.

They do NOT assert a correct solid mask: PicoGK stores a narrow band, so the
deep interior is inactive and reads as the (positive) background value. Turning
the band into a filled solid (flood fill / bIsInside / BlackWhite slice mode) is
the open item for a real PicoGKKernel and is intentionally not claimed here.

The whole module is skipped when the native library cannot be loaded (non-Windows
CI, missing DLL), so the suite stays green everywhere.
"""

import numpy as np
import pytest

from morphos.geometry import _picogk_native as pk

pytestmark = pytest.mark.skipif(
    not pk.picogk_available(), reason="native PicoGK runtime not loadable on this host"
)


def test_reports_a_version_string():
    assert pk.version()  # non-empty, e.g. "26.2.0"


def test_sphere_bounding_box_matches_radius_and_voxel_size():
    res = pk.sphere_sdf_volume(center_mm=(10, 10, 10), radius_mm=5.0, voxel_mm=0.5)
    nz, ny, nx = res.volume.shape
    # a radius-5mm sphere at 0.5mm voxels spans ~20 voxels plus a few of band
    for n in (nx, ny, nz):
        assert 20 <= n <= 30


def test_field_is_a_narrow_band_with_both_signs():
    res = pk.sphere_sdf_volume(center_mm=(10, 10, 10), radius_mm=5.0, voxel_mm=0.5)
    sd = res.volume * res.voxel_mm
    assert sd.min() < 0.0  # interior band is negative
    assert sd.max() > 0.0  # exterior band is positive
    # values are clamped to the +/- background half-band (in mm)
    assert np.abs(sd).max() == pytest.approx(res.background * res.voxel_mm, rel=1e-5)


def test_surface_voxels_sit_near_the_requested_radius_on_average():
    center = (10.0, 10.0, 10.0)
    R = 5.0
    res = pk.sphere_sdf_volume(center_mm=center, radius_mm=R, voxel_mm=0.5)
    sd = res.volume * res.voxel_mm
    zi, yi, xi = np.where(np.abs(sd) < 0.12)
    ox, oy, oz = res.origin
    wx = (ox + xi) * res.voxel_mm
    wy = (oy + yi) * res.voxel_mm
    wz = (oz + zi) * res.voxel_mm
    dist = np.sqrt((wx - center[0]) ** 2 + (wy - center[1]) ** 2 + (wz - center[2]) ** 2)
    # the mean surface radius is correct even though individual band voxels scatter
    assert dist.mean() == pytest.approx(R, abs=0.5)
