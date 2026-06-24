"""Structural stiffening templates: ribs and lattices as density Fields.

Density convention (matches the elasticity/thermo-elastic oracles): ``1.0`` is
solid structural material, ``0.0`` is void. These act as geometric priors for
stiffness-driven topology runs, where ribs add bending/torsional stiffness for
a small material fraction.

All builders work on a 2D grid Field ``(ny, nx)``, matching the existing
channel templates in :mod:`morphos.library.channels`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from morphos.field import Field


def cross_rib(params: Dict[str, float], grid: Field) -> Field:
    """An X-shaped diagonal cross brace spanning the domain corners.

    Two diagonal ribs of ``thickness_voxels`` width crossing through the
    domain center, a minimal-material pattern that resists bending in both
    diagonal directions. Params: ``thickness_voxels``.
    """
    ny, nx = grid.values.shape
    thickness = float(params.get("thickness_voxels", 2.0))
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    # normalise so both diagonals are measured in voxel units regardless of
    # the domain's aspect ratio
    scale = ny / nx if nx else 1.0
    diag1 = np.abs(y - scale * x) / np.sqrt(1.0 + scale ** 2)
    diag2 = np.abs(y - (ny - 1) + scale * x) / np.sqrt(1.0 + scale ** 2)
    out = ((diag1 <= thickness / 2.0) | (diag2 <= thickness / 2.0)).astype(float)
    return grid.like(out)


def i_beam_rib(params: Dict[str, float], grid: Field) -> Field:
    """An I-beam cross-section rib: two flanges (top/bottom) joined by a web.

    Params: ``flange_voxels`` (flange thickness), ``web_voxels`` (web
    thickness). The I-section places material where bending stress is
    highest (flanges, far from the neutral axis) and a thin web resisting
    shear, the classic minimal-material bending-stiff cross-section.
    """
    ny, nx = grid.values.shape
    flange = int(params.get("flange_voxels", 2))
    web = int(params.get("web_voxels", 2))
    out = np.zeros((ny, nx))
    flange = min(flange, ny)
    out[0:flange, :] = 1.0
    out[ny - flange:ny, :] = 1.0
    c = nx // 2
    lo = max(0, c - web // 2)
    hi = min(nx, lo + web)
    out[:, lo:hi] = 1.0
    return grid.like(out)


def honeycomb_rib(params: Dict[str, float], grid: Field) -> Field:
    """A hexagonal honeycomb wall pattern: thin walls, high stiffness-to-weight.

    Params: ``cell_size_voxels`` (hexagon pitch), ``wall_voxels`` (wall
    thickness). Built from the standard hexagonal lattice distance function:
    the minimum distance to the nearest of three rotated periodic line
    families gives the cell wall network directly, no explicit polygon mesh
    needed.
    """
    ny, nx = grid.values.shape
    cell = float(params.get("cell_size_voxels", 6.0))
    wall = float(params.get("wall_voxels", 1.0))
    if cell <= 0.0:
        raise ValueError("cell_size_voxels must be positive")
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)

    # Three line families at 0, 60, 120 degrees, each periodic with period
    # `cell`; distance-to-nearest-line-of-the-family via a triangle wave.
    def _line_family(coord, period):
        t = np.mod(coord, period)
        return np.minimum(t, period - t)

    period = cell
    fam0 = _line_family(y, period)
    fam60 = _line_family(0.5 * y + (np.sqrt(3.0) / 2.0) * x, period)
    fam120 = _line_family(0.5 * y - (np.sqrt(3.0) / 2.0) * x, period)
    dist = np.minimum(np.minimum(fam0, fam60), fam120)
    out = (dist <= wall / 2.0).astype(float)
    return grid.like(out)
