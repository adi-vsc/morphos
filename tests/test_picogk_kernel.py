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


def _analytic_box_sdf(kernel, center, size):
    """Signed distance to an axis-aligned box (exact, via the standard box SDF)."""
    grids = kernel.coordinate_grids()
    half = [s / 2.0 for s in size]
    q = [np.abs(g - c) - h for g, c, h in zip(grids, center, half)]
    outside = np.sqrt(sum(np.maximum(qi, 0.0) ** 2 for qi in q))
    inside = np.minimum(np.maximum(q[0], np.maximum(q[1], q[2])), 0.0)
    return outside + inside


def test_build_box_is_a_sign_correct_solid():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (11.0, 9.0, 12.0)  # off-center: breaks symmetry
    size = (8.0, 10.0, 6.0)
    field = kernel.build({"primitive": "box", "center": center, "size": size})

    sd = _analytic_box_sdf(kernel, center, size)
    got_solid = field.values < 0.0
    true_solid = sd < 0.0
    away = np.abs(sd) > spacing
    agreement = float(np.mean(got_solid[away] == true_solid[away]))
    assert agreement > 0.999


def _analytic_cylinder_sdf(kernel, center, axis, radius, height):
    """Signed distance to a finite cylinder along an arbitrary axis (exact)."""
    grids = kernel.coordinate_grids()
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    center = np.asarray(center, dtype=float)

    rel = [g - c for g, c in zip(grids, center)]
    axial = sum(r * a for r, a in zip(rel, axis))
    radial_vec = [r - axial * a for r, a in zip(rel, axis)]
    radial = np.sqrt(sum(rv**2 for rv in radial_vec))

    d_radial = radial - radius
    d_axial = np.abs(axial) - height / 2.0
    outside = np.sqrt(np.maximum(d_radial, 0.0) ** 2 + np.maximum(d_axial, 0.0) ** 2)
    inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
    return outside + inside


def test_build_cylinder_is_a_sign_correct_solid():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (10.0, 10.0, 10.0)
    axis = (0.0, 0.0, 1.0)
    radius = 6.0
    height = 12.0
    field = kernel.build(
        {
            "primitive": "cylinder",
            "center": center,
            "axis": axis,
            "radius": radius,
            "height": height,
        }
    )

    sd = _analytic_cylinder_sdf(kernel, center, axis, radius, height)
    got_solid = field.values < 0.0
    true_solid = sd < 0.0
    away = np.abs(sd) > spacing
    agreement = float(np.mean(got_solid[away] == true_solid[away]))
    assert agreement > 0.999


def test_build_resamples_when_picogk_voxel_size_differs_from_kernel_spacing():
    # Kernel grid wants 0.5mm voxels; PicoGK is asked to build at a different
    # (finer) native voxel size, exercising the resample_volume/_origin path
    # in PicoGKKernel.build instead of the exact-match fast path.
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (11.0, 9.0, 12.0)
    radius = 6.0
    field = kernel.build(
        {"primitive": "sphere", "center": center, "radius": radius}, picogk_voxel_mm=0.25
    )

    assert field.shape == grid_shape
    assert field.spacing == (spacing, spacing, spacing)

    sd = _analytic_sphere_sdf(kernel, center, radius)
    got_solid = field.values < 0.0
    true_solid = sd < 0.0
    away = np.abs(sd) > spacing
    agreement = float(np.mean(got_solid[away] == true_solid[away]))
    assert agreement > 0.99


def test_build_resampled_volume_matches_analytic_volume():
    grid_shape = (40, 40, 40)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (10.0, 10.0, 10.0)
    radius = 6.0
    field = kernel.build(
        {"primitive": "sphere", "center": center, "radius": radius}, picogk_voxel_mm=0.25
    )

    solid_voxels = int(np.sum(field.values < 0.0))
    got_volume = solid_voxels * (spacing**3)
    analytic_volume = (4.0 / 3.0) * np.pi * radius**3
    assert got_volume == pytest.approx(analytic_volume, rel=0.05)


def test_build_cylinder_along_tilted_axis_is_a_sign_correct_solid():
    grid_shape = (44, 44, 44)
    spacing = 0.5
    kernel = PicoGKKernel(grid_shape, spacing)
    center = (11.0, 10.0, 12.0)
    axis = (1.0, 0.0, 1.0)  # not axis-aligned: exercises the basis-rotation path
    radius = 5.0
    height = 10.0
    field = kernel.build(
        {
            "primitive": "cylinder",
            "center": center,
            "axis": axis,
            "radius": radius,
            "height": height,
        }
    )

    sd = _analytic_cylinder_sdf(kernel, center, axis, radius, height)
    got_solid = field.values < 0.0
    true_solid = sd < 0.0
    away = np.abs(sd) > spacing
    agreement = float(np.mean(got_solid[away] == true_solid[away]))
    assert agreement > 0.995
