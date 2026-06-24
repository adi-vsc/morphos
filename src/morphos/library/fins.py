"""Heat-exchanger fin templates: parameterized fin arrays as density Fields.

Density convention (matches the flow/heat oracles): ``1.0`` is solid fin
material, ``0.0`` is void/fluid. These act as geometric priors for conjugate
heat transfer topology runs, where fins increase finned surface area to boost
convective heat transfer.

All builders work on a 2D grid Field ``(ny, nx)``, matching the existing
channel templates in :mod:`morphos.library.channels`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from morphos.field import Field


def pin_fin_array(params: Dict[str, float], grid: Field) -> Field:
    """A row of rectangular pin fins protruding from the base (y=0) wall.

    Fins are placed periodically along x with the given ``pitch_voxels``,
    each ``thickness_voxels`` wide and ``height_voxels`` tall, solid (1.0)
    elsewhere void (0.0). Params: ``pitch_voxels``, ``height_voxels``,
    ``thickness_voxels``.
    """
    ny, nx = grid.values.shape
    pitch = int(params.get("pitch_voxels", 4))
    height = int(params.get("height_voxels", max(1, ny // 2)))
    thickness = int(params.get("thickness_voxels", 1))
    if pitch < 1:
        raise ValueError("pitch_voxels must be >= 1")
    out = np.zeros((ny, nx))
    height = min(height, ny)
    for x0 in range(0, nx, pitch):
        x1 = min(x0 + thickness, nx)
        out[0:height, x0:x1] = 1.0
    return grid.like(out)


def plate_fin_array(params: Dict[str, float], grid: Field) -> Field:
    """Full-height plate fins spanning the entire y extent, periodic along x.

    Params: ``pitch_voxels``, ``thickness_voxels``.
    """
    ny, nx = grid.values.shape
    pitch = int(params.get("pitch_voxels", 4))
    thickness = int(params.get("thickness_voxels", 1))
    if pitch < 1:
        raise ValueError("pitch_voxels must be >= 1")
    out = np.zeros((ny, nx))
    for x0 in range(0, nx, pitch):
        x1 = min(x0 + thickness, nx)
        out[:, x0:x1] = 1.0
    return grid.like(out)


def corrugated_fin(params: Dict[str, float], grid: Field) -> Field:
    """A sinusoidal corrugated fin sheet, periodic along x with given
    ``pitch_voxels`` and lateral ``amplitude_voxels``, ``thickness_voxels`` thick.

    Corrugated fins maximise finned surface area per unit footprint by folding
    a thin sheet into a wave, used in compact heat exchangers.
    """
    ny, nx = grid.values.shape
    pitch = int(params.get("pitch_voxels", 6))
    amplitude = float(params.get("amplitude_voxels", 3.0))
    thickness = float(params.get("thickness_voxels", 1.0))
    if pitch < 1:
        raise ValueError("pitch_voxels must be >= 1")
    x = np.arange(nx)
    y = np.arange(ny)[:, None]
    center_x = nx / 2.0 + amplitude * np.sin(2.0 * np.pi * y / pitch)
    out = (np.abs(x[None, :] - center_x) <= thickness / 2.0).astype(float)
    return grid.like(out)
