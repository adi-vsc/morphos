"""3D heat-exchanger / lattice geometry generators (occupancy Fields).

These build *manufacturable solid occupancy* (1.0 solid, 0.0 void) on a 3D grid,
ready for STL extraction via :func:`morphos.manufacturing.export.export_bundle`.
They are the implicit-geometry (SDF) leg of the engine: geometry defined by a
closed-form level set and produced by construction, so the result is crisp and
printable rather than a thresholded gray density field.

The headline generator, :func:`sheet_gyroid_exchanger`, builds a counter-flow
gyroid core: the thickened gyroid surface is the solid wall, and the two sides
of that surface are two interpenetrating fluid labyrinths that never connect --
the defining, leak-tight property of a counter-flow plate exchanger. The
separation is verifiable (:func:`gyroid_exchanger_metrics`), not assumed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from morphos.field import Field
from morphos.geometry.tpms import _gyroid_level_set, _wave_number


def _grids(grid: Field):
    axes = [np.arange(n) * s for n, s in zip(grid.shape, grid.spacing)]
    return np.meshgrid(*axes, indexing="ij")


def _envelope_mask(grid: Field, params: Dict[str, float]) -> np.ndarray:
    """Solid envelope the lattice is confined to: a cylinder (default) or box."""
    X, Y, Z = _grids(grid)
    N = grid.shape[0]
    shape = str(params.get("envelope", "cylinder"))
    if shape == "box":
        margin = float(params.get("margin_voxels", 1.0))
        m = np.ones(grid.shape, bool)
        for C in (X, Y, Z):
            m &= (C >= margin) & (C <= C.max() - margin)
        return m
    cx = (grid.shape[0] - 1) * grid.spacing[0] / 2.0
    cy = (grid.shape[1] - 1) * grid.spacing[1] / 2.0
    R = float(params.get("radius_voxels", 0.46 * N))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return r <= R


def gyroid_flow_domains(
    params: Dict[str, float], grid: Field
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean ``(wall, hot, cold)`` masks for a sheet-gyroid exchanger."""
    X, Y, Z = _grids(grid)
    period = float(params.get("period_voxels", 33.0))
    wall_t = float(params.get("wall_thickness_voxels", 1.4))
    k = _wave_number(period)
    f = _gyroid_level_set(X, Y, Z, k)
    grad_scale = k * 3.0
    sheet = np.abs(f) / grad_scale - wall_t / 2.0   # <= 0 is solid wall
    env = _envelope_mask(grid, params)
    wall = (sheet <= 0.0) & env
    fluid = (sheet > 0.0) & env
    hot = fluid & (f > 0.0)
    cold = fluid & (f < 0.0)
    return wall, hot, cold


def sheet_gyroid_exchanger(params: Dict[str, float], grid: Field) -> Field:
    """Counter-flow gyroid heat-exchanger core: occupancy of the solid wall.

    Params (voxel units): ``period_voxels`` (gyroid cell size), ``wall_thickness_voxels``,
    ``envelope`` (``"cylinder"`` | ``"box"``), ``radius_voxels``. The returned Field
    is occupancy in [0, 1] (the wall, lightly smoothed for clean marching cubes).
    """
    wall, _hot, _cold = gyroid_flow_domains(params, grid)
    from scipy import ndimage
    occ = np.clip(ndimage.gaussian_filter(wall.astype(float), 0.8), 0.0, 1.0)
    return grid.like(occ)


def gyroid_exchanger_metrics(params: Dict[str, float], grid: Field) -> Dict[str, float]:
    """Quantitative quality of a sheet-gyroid exchanger, computed from the masks.

    Returns wall/hot/cold volume fractions, the fraction of each fluid that lies
    in its single largest connected component (1.0 == one continuous network),
    and ``leak_paths`` (number of connected void components that touch *both*
    fluids; 0 means hot and cold are provably sealed from each other).
    """
    from scipy import ndimage

    wall, hot, cold = gyroid_flow_domains(params, grid)
    env = wall | hot | cold
    tot = float(env.sum())

    def largest_fraction(mask):
        if not mask.any():
            return 0.0
        lbl, n = ndimage.label(mask)
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        return float(sizes.max() / mask.sum())

    fluid = hot | cold
    lbl, ncomp = ndimage.label(fluid)
    leak = 0
    for c in range(1, ncomp + 1):
        comp = lbl == c
        if (hot & comp).any() and (cold & comp).any():
            leak += 1

    return {
        "wall_fraction": float(wall.sum() / tot),
        "hot_fraction": float(hot.sum() / tot),
        "cold_fraction": float(cold.sum() / tot),
        "hot_single_network_fraction": largest_fraction(hot),
        "cold_single_network_fraction": largest_fraction(cold),
        "leak_paths": float(leak),
    }


def gyroid_lattice_block(params: Dict[str, float], grid: Field) -> Field:
    """A solid envelope filled with a gyroid *network* lattice at a target solid
    volume fraction -- lightweight infill for structural / lightweighting parts.

    Params: ``period_voxels``, ``volume_fraction`` (target solid fraction inside
    the envelope), ``envelope`` (``"box"`` default here | ``"cylinder"``).
    """
    X, Y, Z = _grids(grid)
    period = float(params.get("period_voxels", 22.0))
    vf = float(params.get("volume_fraction", 0.3))
    vf = min(max(vf, 0.02), 0.98)
    k = _wave_number(period)
    f = _gyroid_level_set(X, Y, Z, k)
    env = _envelope_mask(grid, {"envelope": params.get("envelope", "box"), **params})
    thr = np.sort(f[env].ravel())[int(vf * (env.sum() - 1))] if env.any() else 0.0
    solid = (f <= thr) & env
    from scipy import ndimage
    occ = np.clip(ndimage.gaussian_filter(solid.astype(float), 0.7), 0.0, 1.0)
    return grid.like(occ)


def pin_fin_heat_sink(params: Dict[str, float], grid: Field) -> Field:
    """A 3D pin-fin heat sink: a base plate (in z) with a periodic array of
    square pins standing up along z. Occupancy in [0, 1].

    Params: ``base_voxels`` (plate thickness), ``pitch_voxels`` (pin spacing),
    ``pin_voxels`` (pin side), ``fin_height_voxels``.
    """
    nz, ny, nx = grid.shape
    base = int(params.get("base_voxels", max(2, nz // 8)))
    pitch = int(params.get("pitch_voxels", 6))
    pin = int(params.get("pin_voxels", 3))
    height = int(params.get("fin_height_voxels", nz - base))
    pitch = max(pitch, 1)
    occ = np.zeros(grid.shape)
    occ[:base, :, :] = 1.0  # base plate at z = 0..base
    top = min(base + height, nz)
    for iy in range(0, ny, pitch):
        for ix in range(0, nx, pitch):
            occ[base:top, iy:min(iy + pin, ny), ix:min(ix + pin, nx)] = 1.0
    return grid.like(occ)
