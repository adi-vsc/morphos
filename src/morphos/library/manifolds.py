"""Manifold templates: branching flow distributors as signed distance Fields.

Sign convention (matches the geometry kernel): negative inside the open flow
passage, positive in solid wall, near zero on the passage wall. These act as
geometric priors for Darcy/Stokes flow topology runs that need a connected
inlet-to-outlets passage.

Both manifolds are built as the union (``min`` of SDFs) of cylindrical pipe
segments, then locally blended with a thin gyroid TPMS shell from
:mod:`morphos.geometry.tpms` at each branch junction: a TPMS sheet has zero
mean curvature and blends two intersecting pipes into one smooth, connected,
self-supporting surface instead of a sharp (and printability-poor) corner.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from morphos.field import Field
from morphos.geometry.tpms import gyroid_sdf


def _coordinate_grids(field: Field):
    axes = [np.arange(n) * s for n, s in zip(field.shape, field.spacing)]
    return np.meshgrid(*axes, indexing="ij")


def _cylinder_sdf(grids, p0, p1, radius: float) -> np.ndarray:
    """SDF of a finite-length cylindrical pipe segment from p0 to p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("cylinder endpoints must be distinct")
    axis_hat = axis / length
    rel = np.stack([g - p0[i] for i, g in enumerate(grids)], axis=-1)
    t = rel @ axis_hat
    t_clamped = np.clip(t, 0.0, length)
    closest = t_clamped[..., None] * axis_hat
    perp = rel - closest
    radial = np.linalg.norm(perp, axis=-1)
    axial_excess = np.maximum(t - length, -t)  # >0 past either end
    # union of the infinite-cylinder radial distance with the end-cap planes
    return np.maximum(radial - radius, axial_excess)


def _junction_blend(sdf: np.ndarray, grids, junction, blend_radius: float, period: float) -> np.ndarray:
    """Smooth a branch junction with a thin gyroid shell, local to the joint.

    The blend is applied only within ``blend_radius`` of the junction point so
    the rest of the manifold stays a clean pipe; near the joint the sharp pipe
    union is replaced by the gentler gyroid TPMS surface, removing the abrupt
    corner where two cylinders meet.
    """
    dist_to_junction = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, junction)))
    weight = np.clip(1.0 - dist_to_junction / blend_radius, 0.0, 1.0)
    if weight.max() <= 0.0:
        return sdf
    gy = (
        np.sin(2 * np.pi * grids[0] / period) * np.cos(2 * np.pi * grids[1] / period)
        + np.sin(2 * np.pi * grids[1] / period) * np.cos(2 * np.pi * grids[2] / period)
        + np.sin(2 * np.pi * grids[2] / period) * np.cos(2 * np.pi * grids[0] / period)
    ) / (2 * np.pi / period * 3)
    return sdf * (1.0 - weight) + gy * weight


def y_manifold(params: Dict[str, float], grid: Field) -> Field:
    """A single inlet splitting into two outlets with a smooth Y junction.

    The inlet runs along the -x to center, then two outlet branches fan out
    symmetrically toward +x. Params: ``inlet_radius``, ``outlet_radius``
    (defaults to ``inlet_radius`` if absent), ``branch_angle_deg`` (default
    30).
    """
    if grid.ndim != 3:
        raise ValueError("y_manifold requires a 3D grid")
    inlet_r = float(params.get("inlet_radius", 3.0))
    outlet_r = float(params.get("outlet_radius", inlet_r))
    angle = np.deg2rad(float(params.get("branch_angle_deg", 30.0)))

    grids = _coordinate_grids(grid)
    size = grid.physical_size
    cx, cy, cz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    junction = (cx, cy, cz)

    inlet_start = (0.0, cy, cz)
    sdf = _cylinder_sdf(grids, inlet_start, junction, inlet_r)

    branch_len = min(cx, cy) if cx > 0 and cy > 0 else max(size) / 2.0
    for sign in (+1.0, -1.0):
        end = (
            cx + branch_len * np.cos(angle),
            cy + sign * branch_len * np.sin(angle),
            cz,
        )
        branch = _cylinder_sdf(grids, junction, end, outlet_r)
        sdf = np.minimum(sdf, branch)

    blend_radius = max(inlet_r, outlet_r) * 2.5
    period = max(inlet_r, outlet_r) * 2.0
    sdf = _junction_blend(sdf, grids, junction, blend_radius, period)
    return grid.like(sdf)


def tree_manifold(params: Dict[str, float], grid: Field) -> Field:
    """A binary tree manifold: one inlet recursively branching to 2**depth outlets.

    Each generation halves the branch radius and length, fanning outward;
    junctions are TPMS-blended exactly as in :func:`y_manifold`, so every
    branch joins its parent through a connected, smoothly curved surface
    (verified by :class:`morphos.manufacturing.constraints.Connectivity`).
    Params: ``depth`` (int, number of branching generations), ``inlet_radius``,
    ``branch_angle_deg`` (default 35), ``radius_taper`` (default 0.75 per
    generation).
    """
    if grid.ndim != 3:
        raise ValueError("tree_manifold requires a 3D grid")
    depth = int(params.get("depth", 2))
    if depth < 1:
        raise ValueError("depth must be >= 1")
    inlet_r = float(params.get("inlet_radius", 3.0))
    angle = np.deg2rad(float(params.get("branch_angle_deg", 35.0)))
    taper = float(params.get("radius_taper", 0.75))

    grids = _coordinate_grids(grid)
    size = grid.physical_size
    root = (0.0, size[1] / 2.0, size[2] / 2.0)
    center = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)

    sdf = np.full(grid.shape, np.inf)
    junctions = []

    def _branch(start, direction, radius, generation):
        nonlocal sdf
        seg_len = (size[0] / 2.0) / (generation + 1) + size[0] / (2 * (depth + 1))
        end = (
            start[0] + seg_len * direction[0],
            start[1] + seg_len * direction[1],
            start[2] + seg_len * direction[2],
        )
        sdf = np.minimum(sdf, _cylinder_sdf(grids, start, end, radius))
        junctions.append((end, radius))
        if generation >= depth:
            return
        for sign in (+1.0, -1.0):
            dx, dy = direction[0], direction[1]
            new_dir = (
                dx * np.cos(angle) - sign * dy * np.sin(angle),
                dx * np.sin(angle) + sign * dy * np.cos(angle),
                direction[2],
            )
            norm = np.linalg.norm(new_dir)
            new_dir = tuple(c / norm for c in new_dir)
            _branch(end, new_dir, radius * taper, generation + 1)

    direction0 = (
        (center[0] - root[0]) / max(np.linalg.norm(np.array(center) - np.array(root)), 1e-9),
        0.0,
        0.0,
    )
    _branch(root, direction0, inlet_r, 0)

    period = inlet_r * 2.0
    for junction, radius in junctions:
        sdf = _junction_blend(sdf, grids, junction, radius * 2.5, period)

    return grid.like(sdf)
