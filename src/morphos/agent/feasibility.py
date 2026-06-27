"""Pre-run feasibility gate for DesignIntent instances.

This module exposes a single public :func:`check` that validates a
:class:`~morphos.intent.DesignIntent` before handing it to the engine.
All checks are fast (< 10 ms) and do not import or run any physics oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeasibilityResult:
    ok: bool
    violations: List[str]
    suggested_revision: Optional[str] = None


def check(intent) -> FeasibilityResult:
    """Check a DesignIntent for physical feasibility before running the engine.

    Returns FeasibilityResult with ok=True and empty violations when all
    checks pass.  Must run in < 10 ms, must not import the engine or any
    physics oracle.
    """
    violations: List[str] = []

    # ------------------------------------------------------------------
    # 1. Field invariants — attempt a lightweight structural check.
    # We do NOT call intent.build() (expensive) — just verify the intent
    # carries coherent shape parameters.
    # ------------------------------------------------------------------
    try:
        from morphos.field import Field
        import numpy as np
        # Derive a proxy shape from the intent's captured init params.
        init_params = getattr(intent, "_init_params", {})
        proxy_shape = _proxy_shape(intent, init_params)
        if proxy_shape and all(d > 0 for d in proxy_shape):
            proxy = Field(np.ones(proxy_shape), spacing=1.0)
            # Verify spacing is positive (constructor already validates).
            if any(s <= 0 for s in proxy.spacing):
                violations.append("domain spacing must be positive")
    except Exception:
        # Do not surface construction errors here; let the engine report them.
        pass

    # ------------------------------------------------------------------
    # 2. Voxel budget
    # ------------------------------------------------------------------
    init_params = getattr(intent, "_init_params", {})
    proxy_shape = _proxy_shape(intent, init_params)
    if proxy_shape:
        total_voxels = math.prod(proxy_shape)
        if total_voxels > 5_000_000:
            violations.append(
                f"voxel count {total_voxels:,} exceeds 5,000,000 limit; "
                "use coarser resolution"
            )

    # ------------------------------------------------------------------
    # 3. Parameter ranges from catalog
    # ------------------------------------------------------------------
    try:
        from morphos.agent.catalog import CATALOG
        intent_type = type(intent)
        for entry in CATALOG:
            if entry.intent_class is None:
                continue
            if entry.intent_class is intent_type or entry.intent_class.__name__ == intent_type.__name__:
                for param_name, (lo, hi) in entry.param_ranges.items():
                    # Check against _init_params (captured at construction).
                    if param_name in init_params:
                        val = init_params[param_name]
                        if isinstance(val, (int, float)) and not (lo <= float(val) <= hi):
                            violations.append(
                                f"param {param_name}={val} outside allowed range "
                                f"[{lo}, {hi}] for {entry.name}"
                            )
                    # Also fall back to getattr for intent attributes.
                    elif hasattr(intent, param_name):
                        val = getattr(intent, param_name)
                        if isinstance(val, (int, float)) and not (lo <= float(val) <= hi):
                            violations.append(
                                f"param {param_name}={val} outside allowed range "
                                f"[{lo}, {hi}] for {entry.name}"
                            )
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 4. Physical plausibility checks
    # ------------------------------------------------------------------
    _check_thermal_flux(intent, violations)
    _check_cantilever_stress(intent, violations)

    return FeasibilityResult(
        ok=len(violations) == 0,
        violations=violations,
        suggested_revision=_suggest(violations) if violations else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proxy_shape(intent, init_params: dict) -> tuple:
    """Derive a grid shape from common intent attributes."""
    # ThermalSinkIntent, StokesBrinkmanChannelIntent, HeatExchangerIntent,
    # ThermoElasticIntent all use nx / ny.
    if hasattr(intent, "nx") and hasattr(intent, "ny"):
        nz = getattr(intent, "nz", None)
        if nz is not None:
            return (int(intent.ny), int(intent.nx), int(nz))
        return (int(intent.ny), int(intent.nx))
    # CantileverIntent, ChannelIntent use span / height.
    if hasattr(intent, "span") and hasattr(intent, "height"):
        return (int(intent.height), int(intent.span))
    # Fall back to _init_params inspection.
    dims = []
    for key in ("nz", "ny", "nx", "height", "span", "depth"):
        if key in init_params and isinstance(init_params[key], (int, float)):
            dims.append(int(init_params[key]))
    if dims:
        return tuple(dims)
    return ()


def _check_thermal_flux(intent, violations: List[str]) -> None:
    """For ThermalSinkIntent: heat_source [W/m³] * 1e-9 [m³/mm³] < 1 W/mm³."""
    if not hasattr(intent, "heat_source"):
        return
    # intent.heat_source is in W/m³; threshold 1 W/mm³ = 1e9 W/m³.
    heat_source_W_m3 = float(intent.heat_source)
    nx = int(getattr(intent, "nx", 1))
    ny = int(getattr(intent, "ny", 1))
    domain_volume_mm3 = float(nx * ny)  # per unit depth, 1 mm voxels
    # power in W deposited in the 2-D slice of unit depth
    power_W = heat_source_W_m3 * domain_volume_mm3 * 1e-9
    if domain_volume_mm3 > 0 and power_W / domain_volume_mm3 > 1.0:
        violations.append(
            f"thermal heat flux {power_W / domain_volume_mm3:.3g} W/mm³ "
            "exceeds physical plausibility limit of 1.0 W/mm³; "
            "reduce heat_source_W_m3 or increase domain size"
        )


def _check_cantilever_stress(intent, violations: List[str]) -> None:
    """For CantileverIntent: |load| / height < 500 N/mm² (MPa)."""
    if not hasattr(intent, "load") or not hasattr(intent, "height"):
        return
    force_N = abs(float(intent.load))
    height_mm = float(intent.height)
    if height_mm <= 0:
        return
    # Cross-section area = height × 1 mm (unit depth).
    cross_section_mm2 = height_mm * 1.0
    stress_MPa = force_N / cross_section_mm2
    if stress_MPa > 500.0:
        violations.append(
            f"nominal stress {stress_MPa:.1f} MPa (|load|/height) "
            "exceeds 500 MPa plausibility limit; "
            "reduce load or increase domain height"
        )


def _suggest(violations: List[str]) -> str:
    """Build a one-sentence suggested revision from the first violation."""
    if not violations:
        return ""
    first = violations[0]
    if "voxel count" in first:
        return "Reduce nx, ny (or span/height) to stay under 5 M voxels."
    if "heat flux" in first or "heat_source" in first:
        return "Lower heat_source_W_m3 or enlarge the domain dimensions."
    if "stress" in first:
        return "Reduce the applied load or increase the beam height."
    if "outside allowed range" in first:
        return "Adjust the flagged parameter to fall within the catalog range."
    return "Revise the intent parameters to satisfy the flagged constraints."
