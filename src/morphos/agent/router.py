"""Route a natural-language request to a generator + parameters.

The default :class:`KeywordInterpreter` is deterministic: it scores each
catalog generator by keyword overlap with the request and parses common
parameters (size, wall thickness, volume fraction, envelope shape) with regular
expressions. It is intentionally simple and auditable -- no model, no network.

The :class:`Interpreter` protocol is the seam where a stronger backend (an LLM
that emits the same :class:`Plan`) drops in without changing anything downstream.
The honesty rule lives here: if nothing matches, the plan's ``generator`` is
``None`` and the build refuses rather than inventing geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from morphos.agent.catalog import CATALOG, GeneratorSpec


@dataclass
class Plan:
    request: str
    generator: Optional[GeneratorSpec]
    params: Dict[str, float] = field(default_factory=dict)
    shape: tuple = ()
    confidence: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    rationale: str = ""


def extract_common(text: str) -> Dict[str, float]:
    """Pull loosely-specified parameters out of free text (best effort)."""
    t = text.lower()
    common: Dict[str, float] = {}

    m = re.search(r"wall\s*(?:thickness|of)?\s*([0-9]*\.?[0-9]+)\s*mm", t)
    if m:
        common["wall_mm"] = float(m.group(1))

    m = re.search(r"(?:cell|unit cell|pore|pitch)\s*(?:size|of)?\s*([0-9]*\.?[0-9]+)\s*mm", t)
    if m:
        common["cell_mm"] = float(m.group(1))

    # size: prefer an explicit "NN mm cube/cylinder/box", else first NN mm.
    m = re.search(r"([0-9]+)\s*mm\s*(cube|cylinder|box|core|block)", t)
    if m:
        common["size_mm"] = float(m.group(1))
    else:
        sizes = [float(x) for x in re.findall(r"([0-9]+)\s*mm", t)]
        # drop ones we already consumed as wall/cell
        used = {common.get("wall_mm"), common.get("cell_mm")}
        sizes = [s for s in sizes if s not in used]
        if sizes:
            common["size_mm"] = max(sizes)

    m = re.search(r"([0-9]*\.?[0-9]+)\s*%", t)
    if m:
        common["vol_fraction"] = float(m.group(1)) / 100.0
    else:
        m = re.search(r"(?:volume fraction|density|solid fraction)\s*(?:of)?\s*([0-9]*\.?[0-9]+)", t)
        if m:
            v = float(m.group(1))
            common["vol_fraction"] = v if v <= 1.0 else v / 100.0

    if "cylinder" in t or "cylindrical" in t or "round" in t:
        common["shape"] = "cylinder"
    elif "box" in t or "cube" in t or "rectangular" in t or "block" in t:
        common["shape"] = "box"

    return common


def _score(spec: GeneratorSpec, text: str) -> List[str]:
    t = text.lower()
    return [kw for kw in spec.keywords if kw in t]


class Interpreter(Protocol):
    def __call__(self, request: str) -> Plan: ...


class KeywordInterpreter:
    """Deterministic keyword+regex router (the default Interpreter)."""

    def __init__(self, catalog: Optional[List[GeneratorSpec]] = None) -> None:
        self.catalog = catalog if catalog is not None else CATALOG

    def __call__(self, request: str) -> Plan:
        common = extract_common(request)
        scored = [(spec, _score(spec, request)) for spec in self.catalog]
        scored = [(s, hits) for s, hits in scored if hits]
        if not scored:
            avail = ", ".join(s.name for s in self.catalog)
            return Plan(
                request=request, generator=None,
                rationale=f"No generator matched this request. Available: {avail}.",
            )
        scored.sort(key=lambda sh: len(sh[1]), reverse=True)
        spec, hits = scored[0]
        params = spec.make_params(common)
        shape = spec.make_shape(common)
        runner_up = scored[1][1] if len(scored) > 1 else []
        conf = len(hits) / (len(hits) + len(runner_up) + 1e-9)
        return Plan(
            request=request, generator=spec, params=params, shape=shape,
            confidence=float(conf), matched_keywords=hits,
            rationale=(f"Matched '{spec.name}' on {hits}; "
                       f"parsed {common or 'no explicit parameters (using defaults)'}."),
        )


def interpret(request: str) -> Plan:
    """Default entry point: route a request with the deterministic interpreter."""
    return KeywordInterpreter()(request)
