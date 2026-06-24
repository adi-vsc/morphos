"""Component library: a registry of parameterized, validated sub-components.

Every design need not start from a blank domain. A :class:`ComponentSpec` records
a named, parameterized sub-component, the engineering performance envelope it is
expected to hit (for validation gating), and the manufacturability constraints it
should be checked against. A :class:`ComponentLibrary` is a registry of builder
functions that instantiate these components as density Fields on a given grid --
geometric priors that seed or template a topology run.

Density convention (matches the flow oracles): ``1.0`` is open fluid / channel,
``0.0`` is solid wall.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Callable, Dict, List, Optional, Tuple

from morphos.field import Field


@dataclass
class ComponentSpec:
    """A parameterized engineering sub-component and its validation envelope."""

    name: str
    params: Dict[str, float]
    expected_performance: Dict[str, Tuple[float, float]] = _dc_field(default_factory=dict)
    applicable_constraints: List[str] = _dc_field(default_factory=list)
    oracle_types: List[str] = _dc_field(default_factory=list)


# A builder maps (params, grid Field) -> a density Field on that grid.
Builder = Callable[[Dict[str, float], Field], Field]


class ComponentLibrary:
    """Registry of parameterized sub-component builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, Builder] = {}
        self._specs: Dict[str, ComponentSpec] = {}

    def register(
        self,
        name: str,
        builder: Builder,
        expected_performance: Optional[Dict[str, Tuple[float, float]]] = None,
        applicable_constraints: Optional[List[str]] = None,
        oracle_types: Optional[List[str]] = None,
    ) -> None:
        if name in self._builders:
            raise ValueError(f"component {name!r} is already registered")
        self._builders[name] = builder
        self._specs[name] = ComponentSpec(
            name=name,
            params={},
            expected_performance=dict(expected_performance or {}),
            applicable_constraints=list(applicable_constraints or []),
            oracle_types=list(oracle_types or []),
        )

    def names(self) -> List[str]:
        return sorted(self._builders)

    def spec(self, name: str) -> ComponentSpec:
        if name not in self._specs:
            raise KeyError(f"unknown component {name!r}")
        return self._specs[name]

    def build(self, name: str, params: Dict[str, float], grid: Field) -> Field:
        if name not in self._builders:
            raise KeyError(f"unknown component {name!r}")
        return self._builders[name](params, grid)

    def list_compatible(self, oracle_type: str) -> List[str]:
        """Component names validated for (or unrestricted to) an oracle type."""
        return sorted(
            n for n, s in self._specs.items()
            if not s.oracle_types or oracle_type in s.oracle_types
        )
