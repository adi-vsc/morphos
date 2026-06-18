"""Unit test for the narrow-band -> solid fill, with no native dependency.

PicoGK stores only a narrow band around the surface; the deep interior is
inactive and reads as the positive background value, indistinguishable by value
from the deep exterior. ``solidify`` recovers a sign-correct solid by flooding
the exterior from the grid boundary: positive voxels reachable from outside are
exterior, everything the negative band encloses is solid.

This synthesises a PicoGK-style band from an analytic sphere (no DLL needed) so
the fill logic is verified everywhere it runs, including CI without PicoGK.
"""

import numpy as np

from morphos.geometry import _picogk_native as pk


def make_pico_style_band(shape, center, radius, background):
    """Voxel-unit signed distance stored PicoGK-style: only the |sd|<=bg band is
    real; every other voxel (deep interior AND deep exterior) reads +background.
    """
    idx = np.indices(shape).astype(float)
    sd = np.sqrt(sum((idx[a] - center[a]) ** 2 for a in range(len(shape)))) - radius
    band = np.where(np.abs(sd) <= background, sd, background)
    return band, sd


def test_solidify_recovers_a_sign_correct_solid():
    shape = (30, 30, 30)
    center = (15.0, 14.0, 16.0)
    radius = 9.0
    background = 1.5
    band, sd = make_pico_style_band(shape, center, radius, background)

    out = pk.solidify(band, background)

    got_solid = out < 0.0
    true_solid = sd < 0.0
    # away from the one-voxel-thick surface band, the sign must be exact
    away = np.abs(sd) > 1.0
    assert np.array_equal(got_solid[away], true_solid[away])


def test_solidify_fills_the_interior_that_the_raw_band_misses():
    shape = (30, 30, 30)
    center = (15.0, 15.0, 15.0)
    radius = 9.0
    background = 1.5
    band, sd = make_pico_style_band(shape, center, radius, background)

    raw_inside = int(np.sum(band < 0.0))         # only the negative band shell
    filled_inside = int(np.sum(pk.solidify(band, background) < 0.0))
    true_inside = int(np.sum(sd < 0.0))

    assert raw_inside < true_inside               # the raw band undercounts
    assert abs(filled_inside - true_inside) <= 0.02 * true_inside


def test_solidify_preserves_band_values():
    shape = (24, 24, 24)
    center = (12.0, 12.0, 12.0)
    radius = 7.0
    background = 1.5
    band, sd = make_pico_style_band(shape, center, radius, background)

    out = pk.solidify(band, background)
    inband = np.abs(sd) <= background
    # near-surface distances are physical and must be left untouched
    assert np.allclose(out[inband], band[inband])
