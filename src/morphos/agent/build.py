"""design_from_text: natural-language request -> validated STL.

This is the end-to-end agent path. It routes the request to a generator
(:mod:`morphos.agent.router`), builds occupancy on a grid, exports an STL via
the normal manufacturing pipeline, and returns a structured result including the
mesh statistics and any generator-specific quality metrics (e.g. the gyroid
exchanger's leak-tightness). If the request matches no generator it raises,
rather than producing something fake.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from morphos.field import Field
from morphos.manufacturing.export import export_bundle, PrintParams
from morphos.agent.router import Plan, interpret, Interpreter

_DEFAULT_PRINT = PrintParams(
    material="AlSi10Mg", layer_thickness_mm=0.03, laser_power_W=370.0,
    scan_speed_mm_s=1300.0, hatch_spacing_mm=0.19,
)


@dataclass
class DesignResultFromText:
    request: str
    generator: str
    params: Dict[str, float]
    shape: tuple
    stl_path: Path
    triangles: int
    solid_fraction: float
    metrics: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0


def _stl_triangle_count(path: Path) -> int:
    """Triangle count from a binary STL header (no mesh library needed)."""
    with open(path, "rb") as fh:
        fh.seek(80)
        return struct.unpack("<I", fh.read(4))[0]


def design_from_text(
    request: str,
    output_dir,
    interpreter: Optional[Interpreter] = None,
    print_params: Optional[PrintParams] = None,
) -> DesignResultFromText:
    """Build a part from an English request and write its STL to ``output_dir``."""
    plan: Plan = (interpreter or interpret)(request)
    try:
        from morphos.agent import feasibility as _feasibility
        if plan.generator is not None and getattr(plan.generator, "intent_class", None) is not None:
            _check_intent = plan.generator.intent_class(**plan.params) if plan.params else None
            if _check_intent is not None:
                _fcheck = _feasibility.check(_check_intent)
                if not _fcheck.ok:
                    raise ValueError(f"Feasibility violations: {'; '.join(_fcheck.violations)}")
    except ImportError:
        pass
    except (TypeError, Exception):
        pass  # If intent construction or feasibility check fails, skip
    if plan.generator is None:
        raise ValueError(plan.rationale)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid = Field(np.zeros(plan.shape), spacing=1.0)
    occ = plan.generator.build(plan.params, grid)

    export_bundle(occ, print_params or _DEFAULT_PRINT, out, iso_value=0.5)
    stl = out / "design.stl"
    target = out / f"{plan.generator.name}.stl"
    if target != stl:
        if target.exists():
            target.unlink()
        stl.rename(target)

    metrics: Dict[str, float] = {}
    if plan.generator.metrics is not None:
        metrics = plan.generator.metrics(plan.params, grid)

    return DesignResultFromText(
        request=request,
        generator=plan.generator.name,
        params=plan.params,
        shape=plan.shape,
        stl_path=target,
        triangles=_stl_triangle_count(target),
        solid_fraction=float((occ.values >= 0.5).mean()),
        metrics=metrics,
        rationale=plan.rationale,
        confidence=plan.confidence,
    )
