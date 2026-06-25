"""Generator catalog: the set of part-families the agent layer can actually build.

Each :class:`GeneratorSpec` ties a natural-language *intent* (keywords + a human
summary) to a concrete geometry generator and a rule for turning loosely-parsed
request parameters into that generator's parameter dict. This is deliberately a
small, honest catalog: the agent can build exactly these families and nothing
else -- asking for a part with no matching generator returns "unsupported"
rather than a hallucinated mesh.

Adding a new capability is adding a ``GeneratorSpec`` here (plus its generator
function). That is the unit of progress: breadth comes from banking generators,
not from a universal synthesiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from morphos.field import Field
from morphos.library.exchangers import (
    sheet_gyroid_exchanger,
    gyroid_exchanger_metrics,
    gyroid_lattice_block,
    pin_fin_heat_sink,
)


@dataclass
class GeneratorSpec:
    name: str
    summary: str
    keywords: List[str]
    build: Callable[[Dict[str, float], Field], Field]
    make_params: Callable[[Dict[str, float]], Dict[str, float]]
    make_shape: Callable[[Dict[str, float]], Tuple[int, ...]]
    metrics: Optional[Callable[[Dict[str, float], Field], Dict[str, float]]] = None
    domain: str = ""


def _N(common: Dict[str, float], default: int = 100, lo: int = 48, hi: int = 120) -> int:
    """Voxel resolution from a requested size in mm (1 voxel = 1 mm), clamped."""
    size = common.get("size_mm")
    n = int(round(size)) if size else default
    return max(lo, min(hi, n))


def _gyroid_exchanger_params(common: Dict[str, float]) -> Dict[str, float]:
    n = _N(common)
    return {
        "period_voxels": float(common.get("cell_mm", 33.0)),
        "wall_thickness_voxels": float(common.get("wall_mm", 1.4)),
        "envelope": "box" if common.get("shape") == "box" else "cylinder",
        "radius_voxels": 0.46 * n,
    }


def _gyroid_lattice_params(common: Dict[str, float]) -> Dict[str, float]:
    n = _N(common, default=80)
    return {
        "period_voxels": float(common.get("cell_mm", max(12.0, n / 4.5))),
        "volume_fraction": float(common.get("vol_fraction", 0.30)),
        "envelope": "cylinder" if common.get("shape") == "cylinder" else "box",
    }


def _pin_fin_params(common: Dict[str, float]) -> Dict[str, float]:
    n = _N(common, default=80)
    return {
        "base_voxels": max(2, n // 12),
        "pitch_voxels": int(common.get("cell_mm", 7)),
        "pin_voxels": max(2, int(common.get("cell_mm", 7)) // 2),
        "fin_height_voxels": int(0.6 * n),
    }


CATALOG: List[GeneratorSpec] = [
    GeneratorSpec(
        name="gyroid_heat_exchanger",
        summary="Counter-flow gyroid (TPMS) heat-exchanger core: a leak-tight "
                "wall separating two interpenetrating fluid networks.",
        keywords=["heat exchanger", "exchanger", "counter-flow", "counterflow",
                  "recuperator", "intercooler", "gyroid", "tpms", "two fluid",
                  "hot and cold"],
        build=sheet_gyroid_exchanger,
        make_params=_gyroid_exchanger_params,
        make_shape=lambda c: (_N(c), _N(c), _N(c)),
        metrics=gyroid_exchanger_metrics,
        domain="thermal-fluid",
    ),
    GeneratorSpec(
        name="gyroid_lattice_block",
        summary="A solid envelope filled with a gyroid lattice at a target "
                "volume fraction: lightweight / lightweighting infill.",
        keywords=["lattice", "lightweight", "lightweighting", "infill", "cellular",
                  "porous", "metamaterial", "bracket", "lattice block", "filler"],
        build=gyroid_lattice_block,
        make_params=_gyroid_lattice_params,
        make_shape=lambda c: (_N(c, default=80), _N(c, default=80), _N(c, default=80)),
        domain="structural",
    ),
    GeneratorSpec(
        name="pin_fin_heat_sink",
        summary="A pin-fin heat sink: a base plate with a periodic array of "
                "square pins for convective cooling.",
        keywords=["heat sink", "heatsink", "pin fin", "pin-fin", "fins",
                  "cpu cooler", "cooler", "finned"],
        build=pin_fin_heat_sink,
        make_params=_pin_fin_params,
        make_shape=lambda c: (max(40, _N(c, default=80) // 2),
                              _N(c, default=80), _N(c, default=80)),
        domain="thermal",
    ),
]
