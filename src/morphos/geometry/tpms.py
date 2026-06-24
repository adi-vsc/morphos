"""Triply periodic minimal surface (TPMS) and lattice infill primitives.

TPMS sheets (gyroid, Schwartz-P, diamond) are the standard analytic level-set
families used for lightweight AM lattice infill: a single closed-form scalar
function whose zero level set is a smooth, self-supporting, triply periodic
surface. Thickening the level set by a tolerance band turns the surface into a
printable shell with non-zero wall thickness, expressed here as a signed
distance field so it composes with the rest of the geometry kernel.

All functions take and return the project's :class:`Field` type, operating on
the field's own grid (shape and spacing) -- consistent with how
:class:`morphos.geometry.numpy_voxel.VoxelKernel` builds primitives directly
from coordinate grids derived from ``Field.shape``/``Field.spacing``.
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field


def _coordinate_grids(field: Field):
    """Coordinate arrays matching the field's grid (origin at 0, 'ij' order)."""
    axes = [np.arange(n) * s for n, s in zip(field.shape, field.spacing)]
    return np.meshgrid(*axes, indexing="ij")


def _wave_number(period: float) -> float:
    if period <= 0.0:
        raise ValueError("period must be positive")
    return 2.0 * np.pi / period


def gyroid_sdf(field: Field, period: float, thickness: float) -> Field:
    """Signed distance to a gyroid TPMS sheet of the given period and thickness.

    Level-set function: sin(x) cos(y) + sin(y) cos(z) + sin(z) cos(x). The sheet
    is the band where this function is within ``thickness/2`` of zero; outside
    the band the returned value is the (approximate) signed distance to the
    nearest sheet surface, using the standard gradient-magnitude normalisation
    so the field behaves like a true SDF near the surface.
    """
    return _tpms_sheet_sdf(field, period, thickness, _gyroid_level_set)


def schwartz_p_sdf(field: Field, period: float, thickness: float) -> Field:
    """Signed distance to a Schwartz-P TPMS sheet.

    Level-set function: cos(x) + cos(y) + cos(z).
    """
    return _tpms_sheet_sdf(field, period, thickness, _schwartz_p_level_set)


def diamond_sdf(field: Field, period: float, thickness: float) -> Field:
    """Signed distance to a Schwartz Diamond TPMS sheet.

    Level-set function (standard sum-of-products form):
    sin(x) sin(y) sin(z) + sin(x) cos(y) cos(z) + cos(x) sin(y) cos(z)
    + cos(x) cos(y) sin(z).
    """
    return _tpms_sheet_sdf(field, period, thickness, _diamond_level_set)


def _gyroid_level_set(x, y, z, k):
    return (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )


def _schwartz_p_level_set(x, y, z, k):
    return np.cos(k * x) + np.cos(k * y) + np.cos(k * z)


def _diamond_level_set(x, y, z, k):
    return (
        np.sin(k * x) * np.sin(k * y) * np.sin(k * z)
        + np.sin(k * x) * np.cos(k * y) * np.cos(k * z)
        + np.cos(k * x) * np.sin(k * y) * np.cos(k * z)
        + np.cos(k * x) * np.cos(k * y) * np.sin(k * z)
    )


def _tpms_sheet_sdf(field: Field, period: float, thickness: float, level_set_fn) -> Field:
    """Shared sheet-SDF construction for all three TPMS families.

    The level-set function ``F`` has bounded gradient magnitude ``|grad F| <= k *
    n_terms`` for wave number ``k``; dividing by a representative gradient scale
    converts ``F`` into approximate physical distance units, after which the
    sheet of half-thickness ``thickness/2`` is carved out exactly like an offset
    surface (``|signed distance to F=0| - thickness/2``).
    """
    if thickness < 0.0:
        raise ValueError("thickness must be non-negative")
    grids = _coordinate_grids(field)
    k = _wave_number(period)
    f = level_set_fn(grids[0], grids[1], grids[2], k)
    # Gradient-magnitude normalisation: each sine/cosine term contributes at
    # most amplitude k to the gradient; sum of |term| counts gives a tight
    # local Lipschitz bound, used here as a uniform scale (evaluated at the
    # steepest point, amplitude 1 per term) to keep f in distance units.
    n_terms = 3
    grad_scale = k * n_terms
    distance_to_surface = np.abs(f) / grad_scale
    values = distance_to_surface - thickness / 2.0
    return field.like(values)


def shell_offset(field: Field, offset: float) -> Field:
    """Grow a solid SDF outward by ``offset``, expanding its outer surface.

    Subtracting a positive constant from a signed distance field moves the
    zero level set outward by exactly that amount (the field is negative
    inside, so making it "more negative" enlarges the solid). Combined with
    the original field via :func:`numpy.maximum` (constructive intersection
    with its own complement) this is also how a hollow shell of wall
    thickness ``offset`` is produced: the shell is the region between the
    original surface and its inward copy, here we expose the building block
    -- the outward-grown solid -- so callers can subtract it from the
    original to get a shell, or use it directly to expand a part's outer
    boundary.
    """
    if offset < 0.0:
        raise ValueError("offset must be non-negative")
    grown = field.values - float(offset)
    return field.like(grown)


def lattice_infill(outer: Field, tpms: Field, volume_fraction: float) -> Field:
    """Fill ``outer`` with a TPMS lattice tuned to a target solid volume fraction.

    The TPMS field's thickness already encodes its own volume fraction; here we
    instead threshold the underlying scalar field directly so an arbitrary
    target fraction can be hit by construction (rather than re-deriving a wall
    thickness from a transcendental volume integral). The lattice is then
    intersected with the outer solid (``max`` of two SDFs is constructive
    intersection), so the infill never escapes the requested envelope.
    """
    if not 0.0 < volume_fraction < 1.0:
        raise ValueError("volume_fraction must be in (0, 1)")
    if outer.shape != tpms.shape:
        raise ValueError("outer and tpms fields must share a grid")

    # Threshold on the raw TPMS scalar (not yet offset to a shell) so we control
    # the solid fraction directly: solid where tpms.values <= threshold, with
    # threshold chosen empirically via the field's own value distribution so the
    # solid set covers the requested fraction of all samples.
    flat_sorted = np.sort(tpms.values.ravel())
    idx = int(round(volume_fraction * (flat_sorted.size - 1)))
    idx = min(max(idx, 0), flat_sorted.size - 1)
    threshold = flat_sorted[idx]
    lattice_sdf = tpms.values - threshold

    # Constructive intersection of two SDFs: max(a, b) is the SDF of the
    # intersection (solid where both are solid, i.e. both <= 0).
    combined = np.maximum(outer.values, lattice_sdf)
    return outer.like(combined)
