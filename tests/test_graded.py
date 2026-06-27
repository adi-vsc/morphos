"""Graded, physics-coupled TPMS lattices.

The existing ``lattice_infill`` thresholds the gyroid scalar with one global
constant, so the lattice has a uniform volume fraction everywhere. A real
load-bearing part wants the opposite: material concentrated where the physics
demands it and thinned where it does not, exactly the way trabecular bone grades
its density to the local stress field. This module grades the TPMS threshold by a
per-voxel *demand* field, and the demand field can be read straight off a physics
oracle's sensitivity (for SIMP compliance the sensitivity magnitude is the
element strain-energy density), which couples the geometry-generation leg to the
physics-optimization leg.

Solid convention: a voxel is solid where the returned SDF value <= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from morphos.field import Field
from morphos.geometry.tpms import gyroid_sdf
from morphos.geometry.graded import (
    demand_to_volume_fraction,
    graded_lattice,
    physics_demand_field,
)


def _solid_fraction(values: np.ndarray) -> float:
    return float(np.mean(values <= 0.0))


def _box(shape, spacing=1.0) -> Field:
    """A solid box envelope: negative (solid) everywhere on the grid."""
    return Field(np.full(shape, -1.0), spacing=spacing)


# --- demand -> volume fraction map ----------------------------------------


def test_demand_to_vf_is_monotone_and_bounded():
    demand = np.array([0.0, 1.0, 2.0, 4.0])
    vf = demand_to_volume_fraction(demand, vf_min=0.2, vf_max=0.6)
    assert np.all(vf >= 0.2 - 1e-12) and np.all(vf <= 0.6 + 1e-12)
    assert np.all(np.diff(vf) >= -1e-12), "more demand must never mean less material"
    assert vf[0] == pytest.approx(0.2)
    assert vf[-1] == pytest.approx(0.6)


def test_demand_to_vf_constant_field_is_safe():
    vf = demand_to_volume_fraction(np.full((4, 4), 3.0), vf_min=0.2, vf_max=0.6)
    assert np.all(np.isfinite(vf))
    assert np.all((vf >= 0.2 - 1e-9) & (vf <= 0.6 + 1e-9))


# --- graded lattice -------------------------------------------------------


def test_graded_lattice_puts_more_material_where_demand_is_high():
    shape = (24, 24, 24)
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=6.0, thickness=0.0)
    outer = _box(shape)

    # High demand in the first half along x, low in the second half.
    demand = np.zeros(shape)
    demand[: shape[0] // 2] = 1.0

    lat = graded_lattice(outer, tpms, demand, vf_min=0.2, vf_max=0.6)

    hi = _solid_fraction(lat.values[: shape[0] // 2])
    lo = _solid_fraction(lat.values[shape[0] // 2 :])
    assert hi > lo + 0.1, f"expected graded material (hi={hi:.3f}, lo={lo:.3f})"


def test_graded_lattice_global_fraction_tracks_mean_target():
    shape = (24, 24, 24)
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=6.0, thickness=0.0)
    outer = _box(shape)

    # A smooth ramp of demand across the domain.
    ramp = np.linspace(0.0, 1.0, shape[0])[:, None, None] * np.ones(shape)
    vf_min, vf_max = 0.25, 0.55
    lat = graded_lattice(outer, tpms, ramp, vf_min=vf_min, vf_max=vf_max)

    target_mean = demand_to_volume_fraction(ramp, vf_min, vf_max).mean()
    assert _solid_fraction(lat.values) == pytest.approx(target_mean, abs=0.07)


def test_graded_lattice_stays_within_envelope():
    shape = (20, 20, 20)
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=5.0, thickness=0.0)

    # Envelope solid only in a central sub-box; void (positive) elsewhere.
    outer_vals = np.full(shape, 1.0)
    outer_vals[5:15, 5:15, 5:15] = -1.0
    outer = Field(outer_vals, spacing=1.0)

    demand = np.ones(shape)
    lat = graded_lattice(outer, tpms, demand, vf_min=0.3, vf_max=0.6)

    outside = outer_vals > 0.0
    assert np.all(lat.values[outside] > 0.0), "no solid may escape the envelope"


# --- physics coupling: grade to an oracle's strain-energy field -----------


def test_lattice_grades_to_structural_strain_energy():
    """The headline coupling: a cantilever's strain-energy field (read straight
    off the elasticity oracle's sensitivity) drives the lattice grading, so the
    high-stress root carries more material than the low-stress free corner."""
    from morphos.physics.elasticity import ElasticityOracle
    from morphos.objective.objective import MaximizeValue

    ny, nx = 12, 24
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = ElasticityOracle(shape=(ny, nx), fixed_dofs=fixed, loads=loads, penalty=1.0)

    seed = Field(np.full((ny, nx), 1.0), spacing=1.0)  # full-solid probe
    result = oracle.solve(seed)
    demand2d = physics_demand_field(result.gradient)  # |sensitivity| = strain energy

    # Extrude the 2D demand into a thin 3D slab and grade a gyroid lattice on it
    # (the same 2D -> 3D extrusion morphos uses to mesh planar designs).
    nz = 8
    shape = (nz, ny, nx)
    demand3d = np.broadcast_to(demand2d, shape).copy()
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=5.0, thickness=0.0)
    outer = _box(shape)

    lat = graded_lattice(outer, tpms, demand3d, vf_min=0.2, vf_max=0.7)

    # Root = clamped left edge (high strain energy); tip-far corner = low.
    root = _solid_fraction(lat.values[:, :, :4])
    far = _solid_fraction(lat.values[:, : ny // 2, nx - 4 :])
    assert root > far + 0.05, f"material should follow load (root={root:.3f}, far={far:.3f})"


def _mesh_is_closed(faces: np.ndarray) -> bool:
    """A triangle mesh is closed (watertight) iff every undirected edge is
    shared by exactly two triangles."""
    from collections import Counter

    edges = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(min(u, v), max(u, v))] += 1
    return all(count == 2 for count in edges.values())


def test_graded_lattice_meshes_watertight():
    """The graded lattice is watertight by construction: its signed-distance
    zero level set, meshed by the export pipeline's zero-padded marching cubes,
    is a closed manifold surface. (A smooth band around the level set is meshed,
    not a hard binary mask, which is exactly how the SDF defines the surface and
    avoids the marching-cubes binary ambiguity.)"""
    pytest.importorskip("skimage")
    from morphos.manufacturing.export import _iso_surface

    shape = (16, 16, 16)
    grid = Field(np.zeros(shape), spacing=1.0)
    tpms = gyroid_sdf(grid, period=6.0, thickness=0.0)
    outer = _box(shape)

    ramp = np.linspace(0.0, 1.0, shape[0])[:, None, None] * np.ones(shape)
    lat = graded_lattice(outer, tpms, ramp, vf_min=0.3, vf_max=0.6)

    # Smooth solid-occupancy band across the SDF zero level set: 1 inside, 0 in
    # void (and in the zero pad), 0.5 exactly on the surface.
    width = 1.0
    density = np.clip(0.5 - lat.values / (2.0 * width), 0.0, 1.0)
    verts, faces = _iso_surface(density, 0.5, grid.spacing)
    assert len(faces) > 0
    assert _mesh_is_closed(faces), "graded lattice surface must be watertight"
