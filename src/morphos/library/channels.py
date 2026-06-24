"""Channel templates: parameterized flow-channel geometries as density Fields.

Density convention: ``1.0`` is open fluid / channel, ``0.0`` is solid wall. These
serve as geometric priors / initialisations for Stokes or Darcy topology runs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from morphos.field import Field


def straight_channel(params: Dict[str, float], grid: Field) -> Field:
    """A single straight rectangular channel (open fluid) of a given width,
    centred in the domain, solid elsewhere.

    Params: ``width_voxels`` (int), ``orientation`` ('x' flow along the last
    axis, 'y' flow along the first axis; default 'x').
    """
    ny, nx = grid.values.shape
    width = int(params.get("width_voxels", max(1, min(ny, nx) // 2)))
    orientation = params.get("orientation", "x")
    out = np.zeros((ny, nx))
    if orientation == "x":
        c = ny // 2
        lo = max(0, c - width // 2)
        hi = min(ny, lo + width)
        out[lo:hi, :] = 1.0
    elif orientation == "y":
        c = nx // 2
        lo = max(0, c - width // 2)
        hi = min(nx, lo + width)
        out[:, lo:hi] = 1.0
    else:
        raise ValueError("orientation must be 'x' or 'y'")
    return grid.like(out)


def serpentine_channel(params: Dict[str, float], grid: Field) -> Field:
    """A serpentine (boustrophedon) channel: horizontal open passes stacked in
    ``n_passes`` rows, joined at alternating ends, maximising path length in a
    fixed footprint. Params: ``n_passes`` (int), ``width_voxels`` (int)."""
    ny, nx = grid.values.shape
    n_passes = int(params.get("n_passes", 3))
    width = int(params.get("width_voxels", 1))
    out = np.zeros((ny, nx))
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")
    rows = np.linspace(0, ny - width, n_passes).round().astype(int)
    for p, r in enumerate(rows):
        out[r : r + width, :] = 1.0  # horizontal pass
        # vertical connector to the next pass, alternating ends
        if p < len(rows) - 1:
            r2 = rows[p + 1]
            col = (nx - width) if (p % 2 == 0) else 0
            top, bot = sorted((r, r2))
            out[top : bot + width, col : col + width] = 1.0
    return grid.like(out)


def register_channels(library) -> None:
    """Register the channel builders into a :class:`ComponentLibrary`."""
    library.register(
        "straight_channel", straight_channel,
        expected_performance={"pressure_drop_Pa": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["StokesFlowOracle", "DarcyFlowOracle"],
    )
    library.register(
        "serpentine_channel", serpentine_channel,
        expected_performance={"pressure_drop_Pa": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["StokesFlowOracle", "DarcyFlowOracle"],
    )
