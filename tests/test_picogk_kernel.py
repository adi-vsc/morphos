"""PicoGKKernel: the production geometry backend, now wired and verified.

These run only where the native PicoGK runtime loads (skip-guarded elsewhere).
They assert what the spike's open item demanded: the kernel produces a *solid*
signed-distance Field whose sign matches an analytic sphere everywhere, not just
the ~75%-correct raw narrow band. An off-center sphere is used on purpose so an
axis-swap or origin-offset bug cannot hide behind the sphere's symmetry.
"""

import numpy as np
import pytest

from morphos.geometry import _picogk_native as pk
from morphos.geometry.picogk import PicoGKKernel

pytestmark = pytest.mark.skipif(
    not pk.picogk_available(), reason="native PicoGK runtime not loadable on this host"
)


def _analytic_sphere_sdf(kernel, center, radius):
    grids = kernel.coordinate_grids()
    return np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, center))) - radius


def test_build_returns_a_field_on_the_kernel_grid():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    field = kernel.build({"primitive": "sphere", "center": (10, 10, 10), "radius": 6.0})
    assert field.shape == grid_shape
    assert field.spacing == (spacing, spacing, spacing)


def test_build_sphere_is_a_sign_correct_solid():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (11.0, 9.0, 12.0)   # off-center: breaks the sphere's symmetry
    radius = 6.0
    field = kernel.build({"primitive": "sphere", "center": center, "radius": radius})

    sd = _analytic_sphere_sdf(kernel, center, radius)
    got_solid = field.values < 0.0
    true_solid = sd < 0.0
    # away from the surface band the solid mask must match the analytic sphere
    away = np.abs(sd) > spacing
    agreement = float(np.mean(got_solid[away] == true_solid[away]))
    assert agreement > 0.999


def test_build_surface_sits_near_the_requested_radius():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (10.0, 10.0, 10.0)
    radius = 6.0
    field = kernel.build({"primitive": "sphere", "center": center, "radius": radius})

    grids = kernel.coordinate_grids()
    near = np.abs(field.values) < 0.5 * spacing
    dist = np.sqrt(sum((g[near] - c) ** 2 for g, c in zip(grids, center)))
    assert dist.mean() == pytest.approx(radius, abs=spacing)


def test_unknown_primitive_raises():
    kernel = PicoGKKernel((20, 20, 20), 0.5)
    with pytest.raises(ValueError):
        kernel.build({"primitive": "torus"})
