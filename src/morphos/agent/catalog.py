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
from typing import Callable, Dict, List, Optional, Tuple, Type

from morphos.field import Field
from morphos.library.exchangers import (
    sheet_gyroid_exchanger,
    gyroid_exchanger_metrics,
    gyroid_lattice_block,
    pin_fin_heat_sink,
)
from morphos.library.channels import straight_channel, serpentine_channel
from morphos.library.fins import pin_fin_array, plate_fin_array, corrugated_fin
from morphos.library.manifolds import y_manifold, tree_manifold
from morphos.library.structural import cross_rib, i_beam_rib, honeycomb_rib
from morphos.intent import (
    ChannelIntent,
    ThermalSinkIntent,
    CantileverIntent,
    HeatExchangerIntent,
    StokesBrinkmanChannelIntent,
    ThermoElasticIntent,
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
    intent_class: Optional[type] = None
    physics_tags: List[str] = field(default_factory=list)
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    description: str = ""


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
        intent_class=HeatExchangerIntent,
        physics_tags=["thermal", "flow"],
        description="Counter-flow TPMS heat exchanger for two-fluid heat transfer.",
        param_ranges={"period_voxels": (10.0, 50.0), "wall_thickness_voxels": (0.5, 5.0)},
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
        intent_class=CantileverIntent,
        physics_tags=["structural"],
        description="Lightweight gyroid lattice infill for structural brackets.",
        param_ranges={"period_voxels": (8.0, 40.0), "volume_fraction": (0.1, 0.7)},
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
        intent_class=ThermalSinkIntent,
        physics_tags=["thermal"],
        description="Pin-fin heat sink array for convective cooling of electronic components.",
        param_ranges={"base_voxels": (2.0, 20.0), "pitch_voxels": (4.0, 20.0),
                      "pin_voxels": (2.0, 10.0), "fin_height_voxels": (10.0, 80.0)},
    ),
    # --- Channel family ---
    GeneratorSpec(
        name="straight_channel",
        summary="A single straight rectangular open-fluid channel centred in the "
                "domain, solid wall elsewhere; flow runs along the chosen axis.",
        keywords=["straight channel", "channel", "duct", "pipe", "flow channel",
                  "rectangular channel", "flow duct"],
        build=straight_channel,
        make_params=lambda c: {
            "width_voxels": int(max(4, _N(c, default=60) // 4)),
            "orientation": "x",
        },
        make_shape=lambda c: (_N(c, default=60), _N(c, default=60)),
        domain="flow",
        intent_class=ChannelIntent,
        physics_tags=["flow", "thermal"],
        description="Single straight rectangular channel for internal flow optimisation.",
        param_ranges={"span": (10.0, 500.0), "height": (10.0, 500.0),
                      "volume_fraction": (0.1, 1.0)},
    ),
    GeneratorSpec(
        name="serpentine_channel",
        summary="A serpentine (boustrophedon) channel: horizontal passes stacked "
                "in rows and joined at alternating ends, maximising path length.",
        keywords=["serpentine", "serpentine channel", "boustrophedon", "winding channel",
                  "meandering channel", "zigzag channel", "multi-pass channel"],
        build=serpentine_channel,
        make_params=lambda c: {
            "n_passes": int(c.get("n_passes", 3)),
            "width_voxels": int(max(2, _N(c, default=64) // 10)),
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64)),
        domain="flow",
        intent_class=ChannelIntent,
        physics_tags=["flow", "thermal"],
        description="Serpentine multi-pass channel for extended residence time or heat transfer.",
        param_ranges={"span": (10.0, 500.0), "height": (10.0, 500.0),
                      "volume_fraction": (0.1, 1.0)},
    ),
    # --- Fin family ---
    GeneratorSpec(
        name="pin_fin_array",
        summary="A periodic row of rectangular pin fins protruding from the base "
                "wall, parameterised by pitch, height, and thickness.",
        keywords=["pin fin array", "pin array", "pin fins", "pin-fin array",
                  "staggered pins", "fin array"],
        build=pin_fin_array,
        make_params=lambda c: {
            "pitch_voxels": int(c.get("cell_mm", 8)),
            "height_voxels": int(0.5 * _N(c, default=64)),
            "thickness_voxels": int(max(1, int(c.get("cell_mm", 8)) // 4)),
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64)),
        domain="thermal",
        intent_class=ThermalSinkIntent,
        physics_tags=["thermal", "flow"],
        description="Periodic pin-fin array for topology-optimised convective cooling.",
        param_ranges={"nx": (10.0, 500.0), "ny": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
    GeneratorSpec(
        name="plate_fin_array",
        summary="Full-height plate fins spanning the entire domain height, "
                "periodic along x; classic compact heat exchanger geometry.",
        keywords=["plate fin", "plate fins", "plate-fin", "plate fin array",
                  "compact heat exchanger", "parallel fins", "flat fins"],
        build=plate_fin_array,
        make_params=lambda c: {
            "pitch_voxels": int(c.get("cell_mm", 8)),
            "thickness_voxels": 2,
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64)),
        domain="thermal",
        intent_class=ThermalSinkIntent,
        physics_tags=["thermal"],
        description="Full-height plate-fin array for compact heat exchanger initialisation.",
        param_ranges={"nx": (10.0, 500.0), "ny": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
    GeneratorSpec(
        name="corrugated_fin",
        summary="A sinusoidal corrugated fin sheet, periodic along x with given "
                "pitch and lateral amplitude; maximises finned surface per footprint.",
        keywords=["corrugated fin", "corrugated", "wavy fin", "sinusoidal fin",
                  "corrugated heat exchanger", "louvered fin", "offset-strip fin"],
        build=corrugated_fin,
        make_params=lambda c: {
            "pitch_voxels": int(c.get("cell_mm", 10)),
            "amplitude_voxels": float(c.get("wall_mm", 4.0)),
            "thickness_voxels": float(c.get("wall_mm", 2.0)),
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64)),
        domain="thermal",
        intent_class=ThermalSinkIntent,
        physics_tags=["thermal"],
        description="Sinusoidal corrugated-fin sheet for compact heat exchanger cores.",
        param_ranges={"nx": (10.0, 500.0), "ny": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
    # --- Manifold family ---
    GeneratorSpec(
        name="y_manifold",
        summary="A single inlet splitting into two outlets through a smooth Y "
                "junction blended with a local TPMS surface.",
        keywords=["y manifold", "y-manifold", "bifurcation", "branch manifold",
                  "flow splitter", "two-outlet manifold", "y junction"],
        build=y_manifold,
        make_params=lambda c: {
            "inlet_radius": float(c.get("wall_mm", 5.0)),
            "outlet_radius": float(c.get("wall_mm", 5.0)),
            "branch_angle_deg": 30.0,
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64), _N(c, default=64)),
        domain="flow",
        intent_class=StokesBrinkmanChannelIntent,
        physics_tags=["flow"],
        description="Y-shaped bifurcation manifold with smooth TPMS-blended junction.",
        param_ranges={"nx": (16.0, 200.0), "ny": (16.0, 200.0)},
    ),
    GeneratorSpec(
        name="tree_manifold",
        summary="A binary tree manifold: one inlet recursively branching to "
                "2**depth outlets, with TPMS-blended junctions at each level.",
        keywords=["tree manifold", "tree-manifold", "fractal manifold",
                  "bifurcating manifold", "multi-outlet manifold",
                  "binary tree manifold", "hierarchical manifold"],
        build=tree_manifold,
        make_params=lambda c: {
            "depth": int(c.get("n_passes", 2)),
            "inlet_radius": float(c.get("wall_mm", 4.0)),
            "branch_angle_deg": 35.0,
            "radius_taper": 0.75,
        },
        make_shape=lambda c: (_N(c, default=64), _N(c, default=64), _N(c, default=64)),
        domain="flow",
        intent_class=StokesBrinkmanChannelIntent,
        physics_tags=["flow"],
        description="Recursive binary-tree manifold with bio-inspired Murray's-law branching.",
        param_ranges={"nx": (16.0, 200.0), "ny": (16.0, 200.0)},
    ),
    # --- Structural rib family ---
    GeneratorSpec(
        name="cross_rib",
        summary="An X-shaped diagonal cross brace spanning the domain corners: "
                "two diagonal ribs crossing through the center.",
        keywords=["cross rib", "cross brace", "x brace", "diagonal rib",
                  "cross-rib", "diagonal brace", "x-brace", "stiffener"],
        build=cross_rib,
        make_params=lambda c: {
            "thickness_voxels": float(c.get("wall_mm", 3.0)),
        },
        make_shape=lambda c: (_N(c, default=80), _N(c, default=80)),
        domain="structural",
        intent_class=CantileverIntent,
        physics_tags=["structural"],
        description="X-shaped diagonal cross brace for minimal-material bending resistance.",
        param_ranges={"span": (10.0, 500.0), "height": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
    GeneratorSpec(
        name="i_beam_rib",
        summary="An I-beam cross-section rib: two flanges joined by a central web, "
                "the classic minimal-material bending-stiff cross-section.",
        keywords=["i beam", "i-beam", "i beam rib", "i section", "flange",
                  "web flange", "bending rib", "structural rib", "beam"],
        build=i_beam_rib,
        make_params=lambda c: {
            "flange_voxels": int(c.get("wall_mm", 4)),
            "web_voxels": int(c.get("wall_mm", 3)),
        },
        make_shape=lambda c: (_N(c, default=80), _N(c, default=80)),
        domain="structural",
        intent_class=CantileverIntent,
        physics_tags=["structural"],
        description="I-beam cross-section rib placing material at peak bending stress locations.",
        param_ranges={"span": (10.0, 500.0), "height": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
    GeneratorSpec(
        name="honeycomb_rib",
        summary="A hexagonal honeycomb wall pattern: thin walls in a periodic hex "
                "lattice, combining high stiffness-to-weight with in-plane isotropy.",
        keywords=["honeycomb", "honeycomb rib", "hex lattice", "hexagonal lattice",
                  "hexagonal infill", "honeycomb infill", "honeycomb panel", "hex rib"],
        build=honeycomb_rib,
        make_params=lambda c: {
            "cell_size_voxels": float(c.get("cell_mm", 8.0)),
            "wall_voxels": float(c.get("wall_mm", 1.5)),
        },
        make_shape=lambda c: (_N(c, default=80), _N(c, default=80)),
        domain="structural",
        intent_class=CantileverIntent,
        physics_tags=["structural"],
        description="Hexagonal honeycomb rib pattern for isotropic in-plane stiffness.",
        param_ranges={"span": (10.0, 500.0), "height": (10.0, 500.0),
                      "volume_fraction": (0.05, 0.9)},
    ),
]
