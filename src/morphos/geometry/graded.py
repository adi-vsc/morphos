"""Graded, physics-coupled TPMS lattices.

``morphos.geometry.tpms.lattice_infill`` thresholds the TPMS scalar with a single
global constant, giving a lattice of uniform volume fraction. That is the wrong
material distribution for a load-bearing part: stiffness (and heat-transfer area,
and flow path) should be spent where the physics demands it and saved where it
does not. Biology already does this. Trabecular bone grades its local density to
the stress field; leaf venation grades channel density to transport demand. This
module transfers that idea: it grades the TPMS threshold *per voxel* by a scalar
demand field, so the local solid volume fraction tracks demand.

The demand field is deliberately the same currency the optimization leg already
produces. For a SIMP compliance problem the physics oracle's sensitivity
magnitude ``|d(fom)/drho|`` is, term for term, the element strain-energy density
(``p * rho**(p-1) * u_e^T k0 u_e``). So :func:`physics_demand_field` reads the
demand straight off a :class:`~morphos.physics.oracle.PhysicsResult`, coupling the
geometry-generation leg to the physics leg with no new physics code: solve once
on a solid probe, grade the lattice to the resulting strain-energy field.

Solid convention (shared with the rest of the geometry kernel): a voxel is solid
where the returned signed-distance value is <= 0.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from morphos.field import Field


def demand_to_volume_fraction(
    demand: np.ndarray,
    vf_min: float,
    vf_max: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """Map a scalar demand field to a target local volume-fraction field.

    The demand is normalized by *rank* (its empirical CDF position in the demand
    distribution), raised to ``gamma`` (a contrast control: ``gamma > 1``
    concentrates material more aggressively in the highest-demand regions), then
    affinely mapped into ``[vf_min, vf_max]``. The map is monotone non-decreasing,
    so more demand never yields less material.

    Rank (not min-max) normalization is deliberate: physics demand fields such as
    strain-energy density are heavy-tailed, with a sharp peak at a load or source
    point. Min-max normalization lets that single peak set the scale and collapses
    the entire rest of the domain to ``vf_min``. Rank normalization spends the full
    ``[vf_min, vf_max]`` range across the demand *ordering*, so material follows the
    stress gradient everywhere, the way trabecular bone grades across the whole
    bone rather than only at the single most-loaded point. A constant demand field
    (all ranks tied) maps to the midpoint of the range.

    Parameters
    ----------
    demand:
        Per-voxel demand, any non-negative scalar field (e.g. strain-energy
        density, heat-flux magnitude, velocity magnitude). Only its relative
        distribution matters; absolute scale is normalized away.
    vf_min, vf_max:
        The local volume fraction assigned to the lowest- and highest-demand
        voxels respectively. Require ``0 < vf_min <= vf_max < 1``.
    gamma:
        Contrast exponent applied to the normalized demand. Default 1 (linear).
    """
    if not (0.0 < vf_min <= vf_max < 1.0):
        raise ValueError("require 0 < vf_min <= vf_max < 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")

    demand = np.asarray(demand, dtype=float)
    n = demand.size
    if n <= 1:
        normed = np.zeros_like(demand)
    else:
        # Average-rank CDF position in [0, 1]; ties (e.g. a constant field)
        # average to 0.5, the midpoint of the range.
        ranks = rankdata(demand, method="average").reshape(demand.shape)
        normed = (ranks - 1.0) / (n - 1.0)
    normed = np.clip(normed, 0.0, 1.0) ** float(gamma)
    return vf_min + (vf_max - vf_min) * normed


def graded_threshold(tpms_values: np.ndarray, vf_field: np.ndarray) -> np.ndarray:
    """Per-voxel TPMS threshold that realizes a target local volume fraction.

    For a uniform threshold ``t`` the solid set is ``{tpms <= t}``, whose global
    fraction is the empirical quantile level of ``t`` in the TPMS value
    distribution. Inverting that: to make a voxel sit at local fraction
    ``vf``, its threshold is the ``vf``-quantile of the global TPMS values. The
    TPMS scalar is spatially periodic, so over any region a slowly varying
    ``vf_field`` is statistically independent of the local TPMS phase and the
    realized local fraction tracks ``vf`` closely.
    """
    if tpms_values.shape != vf_field.shape:
        raise ValueError("tpms and vf fields must share a grid")
    flat = tpms_values.ravel()
    thresholds = np.quantile(flat, np.clip(vf_field.ravel(), 0.0, 1.0))
    return thresholds.reshape(vf_field.shape)


def graded_lattice(
    outer: Field,
    tpms: Field,
    demand: np.ndarray,
    vf_min: float,
    vf_max: float,
    gamma: float = 1.0,
) -> Field:
    """Fill ``outer`` with a TPMS lattice graded by a per-voxel demand field.

    The local solid volume fraction follows ``demand`` through
    :func:`demand_to_volume_fraction`, realized by a spatially varying threshold
    on the TPMS scalar (:func:`graded_threshold`). The graded lattice is then
    intersected with the ``outer`` envelope (constructive ``max`` of two SDFs),
    so material never escapes the requested envelope.

    Parameters
    ----------
    outer:
        Envelope SDF (solid where <= 0). The lattice is clipped to it.
    tpms:
        A TPMS scalar field on the same grid (e.g. ``gyroid_sdf(..., thickness=0)``).
    demand:
        Per-voxel demand field on the same grid.
    vf_min, vf_max, gamma:
        Forwarded to :func:`demand_to_volume_fraction`.
    """
    demand = np.asarray(demand, dtype=float)
    if not (outer.shape == tpms.shape == demand.shape):
        raise ValueError("outer, tpms, and demand must share a grid")

    vf_field = demand_to_volume_fraction(demand, vf_min, vf_max, gamma)
    threshold = graded_threshold(tpms.values, vf_field)
    lattice_sdf = tpms.values - threshold
    combined = np.maximum(outer.values, lattice_sdf)
    return outer.like(combined)


def physics_demand_field(gradient: np.ndarray) -> np.ndarray:
    """Demand field read off a physics-oracle sensitivity.

    For a SIMP compliance objective the sensitivity magnitude is the element
    strain-energy density, so its absolute value is exactly the "stress demand"
    a bone-like grading should follow. Passing an oracle's ``result.gradient``
    here couples the geometry leg to the physics leg.
    """
    if gradient is None:
        raise ValueError(
            "physics_demand_field needs an analytic sensitivity, but the "
            "oracle returned gradient=None"
        )
    return np.abs(np.asarray(gradient, dtype=float))
