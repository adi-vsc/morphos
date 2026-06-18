"""DesignIntent: build a DesignSpec from engineering-unit parameters.

Today, using an oracle means hand-assembling boundary conditions, loads, and
an objective from scratch (see the ``cantilever_bcs`` / ``channel_bcs`` test
helpers this module promotes to production code). A DesignIntent is a narrow,
honest builder for one well-defined load case per oracle: it takes the
quantities an engineer actually has (span, height, load, volume budget) and
returns a ready-to-run DesignSpec. It deliberately does *not* claim to cover
arbitrary problems (BCs, multi-load cases, custom geometry); each concrete
intent documents the exact load case it builds. Broader auto-configuration
from domain-level specs (e.g. propellant type and thrust) needs a component
library of validated templates first -- see docs/strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from morphos.field import Field
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.darcy import DarcyFlowOracle
from morphos.physics.elasticity import ElasticityOracle
from morphos.spec import DesignSpec


def _check_volume_fraction(volume_fraction: float) -> float:
    vf = float(volume_fraction)
    if not (0.0 < vf <= 1.0):
        raise ValueError("volume_fraction must be in (0, 1]")
    return vf


class DesignIntent(ABC):
    """Base for a builder that turns engineering quantities into a DesignSpec."""

    @abstractmethod
    def build(self) -> DesignSpec:
        raise NotImplementedError


class CantileverIntent(DesignIntent):
    """A cantilever beam under a tip point load: fixed left edge, downward
    point load at the bottom-right corner. SIMP compliance minimization."""

    def __init__(
        self,
        span: int,
        height: int,
        load: float,
        volume_fraction: float,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        step_size: float = 2e-3,
        max_iter: int = 200,
        bounds=(1e-3, 1.0),
    ) -> None:
        self.span = int(span)
        self.height = int(height)
        self.load = float(load)
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.bounds = bounds

    def build(self) -> DesignSpec:
        shape = (self.height, self.span)
        nny, nnx = self.height + 1, self.span + 1
        fixed = [(0, j, axis) for j in range(nny) for axis in ("x", "y")]
        loads = {(nnx - 1, nny - 1, "y"): self.load}
        oracle = ElasticityOracle(
            shape=shape,
            fixed_dofs=fixed,
            loads=loads,
            young_modulus=self.young_modulus,
            poisson_ratio=self.poisson_ratio,
        )
        initial = Field(np.full(shape, self.volume_fraction), spacing=1.0)
        return DesignSpec(
            initial=initial,
            oracle=oracle,
            objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid-limit")),
            optimizer=TopologyOptimizer(
                step_size=self.step_size, max_iter=self.max_iter, tol=-1.0, bounds=self.bounds
            ),
            name="cantilever",
        )


class ChannelIntent(DesignIntent):
    """A flow channel under one inlet point source and a zero-pressure outlet
    edge: left edge held at zero pressure, source injected at the
    bottom-right corner. SIMP dissipation minimization (Darcy/potential
    flow -- see morphos.physics.darcy for the scope of this surrogate)."""

    def __init__(
        self,
        span: int,
        height: int,
        source: float,
        volume_fraction: float,
        permeability0: float = 1.0,
        step_size: float = 2e-3,
        max_iter: int = 200,
        bounds=(1e-3, 1.0),
    ) -> None:
        self.span = int(span)
        self.height = int(height)
        self.source = float(source)
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.permeability0 = float(permeability0)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.bounds = bounds

    def build(self) -> DesignSpec:
        shape = (self.height, self.span)
        nny, nnx = self.height + 1, self.span + 1
        fixed = [(0, j) for j in range(nny)]
        sources = {(nnx - 1, nny - 1): self.source}
        oracle = DarcyFlowOracle(
            shape=shape,
            fixed_nodes=fixed,
            sources=sources,
            permeability0=self.permeability0,
        )
        initial = Field(np.full(shape, self.volume_fraction), spacing=1.0)
        return DesignSpec(
            initial=initial,
            oracle=oracle,
            objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="open-channel-limit")),
            optimizer=TopologyOptimizer(
                step_size=self.step_size, max_iter=self.max_iter, tol=-1.0, bounds=self.bounds
            ),
            name="channel",
        )
