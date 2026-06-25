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

import functools
import inspect
import json
from abc import ABC, abstractmethod

import numpy as np

from morphos.field import Field
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.darcy import DarcyFlowOracle
from morphos.physics.elasticity import ElasticityOracle
from morphos.spec import CoupledSpec, DesignSpec

_EDGE_NODES = {
    # (edge -> predicate building flat node indices on an (nny, nnx) node grid)
    "left": lambda nny, nnx: [j * nnx for j in range(nny)],
    "right": lambda nny, nnx: [j * nnx + (nnx - 1) for j in range(nny)],
    "bottom": lambda nny, nnx: list(range(nnx)),
    "top": lambda nny, nnx: [(nny - 1) * nnx + i for i in range(nnx)],
}


def _edge_node_indices(edge: str, nny: int, nnx: int):
    if edge not in _EDGE_NODES:
        raise ValueError(f"edge must be one of {tuple(_EDGE_NODES)}, got {edge!r}")
    return _EDGE_NODES[edge](nny, nnx)


def _check_volume_fraction(volume_fraction: float) -> float:
    vf = float(volume_fraction)
    if not (0.0 < vf <= 1.0):
        raise ValueError("volume_fraction must be in (0, 1]")
    return vf


def _intent_registry() -> dict:
    """Every concrete :class:`DesignIntent` subclass keyed by class name."""
    registry: dict = {}

    def collect(cls):
        for sub in cls.__subclasses__():
            registry[sub.__name__] = sub
            collect(sub)

    collect(DesignIntent)
    return registry


def _jsonable(value):
    """Coerce a captured constructor argument into a JSON-safe value (tuples and
    numpy scalars/arrays become lists/Python scalars)."""
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class DesignIntent(ABC):
    """Base for a builder that turns engineering quantities into a DesignSpec.

    Every concrete intent's constructor arguments are captured at construction
    time (see :meth:`__init_subclass__`), which makes an intent the one design
    object in the engine that round-trips losslessly through JSON: a built
    DesignSpec cannot, because its oracle consumes the boundary conditions into
    solver-internal state at construction. Serialise at the intent layer.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        @functools.wraps(original_init)
        def _capturing_init(self, *args, **kw):
            bound = inspect.signature(original_init).bind(self, *args, **kw)
            bound.apply_defaults()
            self._init_params = {
                name: value
                for name, value in bound.arguments.items()
                if name != "self"
            }
            original_init(self, *args, **kw)

        cls.__init__ = _capturing_init

    @abstractmethod
    def build(self) -> DesignSpec:
        raise NotImplementedError

    def to_dict(self) -> dict:
        """The intent as a JSON-safe spec dict (Shape A): a ``version``, the
        intent class name, and its exact constructor parameters."""
        params = {k: _jsonable(v) for k, v in getattr(self, "_init_params", {}).items()}
        return {"version": "1.0", "intent": type(self).__name__, "params": params}

    @classmethod
    def from_dict(cls, data: dict) -> "DesignIntent":
        """Reconstruct an intent from a Shape A spec dict, resolving the intent
        class by name from the registry of all concrete subclasses."""
        name = data.get("intent")
        if name is None:
            raise ValueError(
                "spec dict has no intent: expected an 'intent' key naming a "
                f"DesignIntent subclass, found keys {sorted(data)}"
            )
        registry = _intent_registry()
        if name not in registry:
            raise ValueError(
                f"unknown intent {name!r}: expected one of {sorted(registry)}"
            )
        return registry[name](**data.get("params", {}))

    def to_json(self) -> str:
        """Serialise this intent to a JSON string (Shape A)."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "DesignIntent":
        """Reconstruct an intent from a JSON string (Shape A)."""
        return cls.from_dict(json.loads(s))


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


def _parabolic_inflow(u_max: float, axis: int):
    """Parabolic velocity profile peaking at ``u_max``, flowing along ``axis``
    (0 = +x, 1 = +y), zero across the span. Suitable as a Stokes inlet."""
    def profile(x):
        span = x[1 - axis]
        lo, hi = span.min(), span.max()
        length = max(hi - lo, 1e-30)
        s = (span - lo) / length
        mag = u_max * 4.0 * s * (1.0 - s)
        comps = [np.zeros_like(mag), np.zeros_like(mag)]
        comps[axis] = mag
        return np.stack(comps)
    return profile


class ThermalSinkIntent(DesignIntent):
    """Distribute conductive material in a domain with a uniform volumetric heat
    source and one fixed-temperature sink edge, to minimise mean temperature.

    Built on :class:`ConjugateHeatOracle` at zero velocity, so the conductivity
    field ``k(rho) = k_fluid + rho^p (k_solid - k_fluid)`` is the design variable
    (pure SIMP conduction). Objective: maximise ``-mean_temperature``.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        heat_source_W_m3: float,
        k_solid: float,
        sink_edge: str = "left",
        volume_fraction: float = 0.5,
        k_fluid: float = 1e-3,
        p_simp: float = 3.0,
        step_size: float = 0.2,
        max_iter: int = 60,
        bounds=(1e-3, 1.0),
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.heat_source = float(heat_source_W_m3)
        self.k_solid = float(k_solid)
        self.k_fluid = float(k_fluid)
        self.sink_edge = sink_edge
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.p_simp = float(p_simp)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.bounds = bounds

    def build(self) -> DesignSpec:
        from morphos.physics.conjugate_heat import ConjugateHeatOracle

        shape = (self.ny, self.nx)
        nny, nnx = self.ny + 1, self.nx + 1
        sink = _edge_node_indices(self.sink_edge, nny, nnx)
        oracle = ConjugateHeatOracle(
            shape=shape,
            velocity=np.zeros((nny, nnx, 2)),
            source=np.full(shape, self.heat_source),
            fixed_nodes=sink,
            fixed_values=[0.0] * len(sink),
            k_solid=self.k_solid,
            k_fluid=self.k_fluid,
            p_simp=self.p_simp,
        )
        initial = Field(np.full(shape, self.volume_fraction), spacing=1.0)
        return DesignSpec(
            initial=initial,
            oracle=oracle,
            objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="isothermal-limit")),
            optimizer=TopologyOptimizer(
                step_size=self.step_size, max_iter=self.max_iter, tol=-1.0, bounds=self.bounds
            ),
            name="thermal-sink",
        )


class StokesBrinkmanChannelIntent(DesignIntent):
    """Find the internal channel topology that minimises viscous dissipation for
    a prescribed parabolic inflow, built on :class:`StokesFlowOracle`."""

    def __init__(
        self,
        nx: int,
        ny: int,
        mu: float = 1.0,
        u_max: float = 1.0,
        inlet_edge: str = "left",
        volume_fraction: float = 1.0,
        step_size: float = 1.0,
        max_iter: int = 8,
        bounds=(0.0, 1.0),
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.mu = float(mu)
        self.u_max = float(u_max)
        self.inlet_edge = inlet_edge
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.bounds = bounds

    def build(self) -> DesignSpec:
        from morphos.physics.stokes import StokesFlowOracle

        shape = (self.ny, self.nx)
        # Flow axis is +x for a left/right inlet, +y for a top/bottom inlet.
        axis = 0 if self.inlet_edge in ("left", "right") else 1
        walls = ("top", "bottom") if axis == 0 else ("left", "right")
        oracle = StokesFlowOracle(
            shape=shape,
            inlet=(self.inlet_edge, _parabolic_inflow(self.u_max, axis)),
            noslip_edges=walls,
            viscosity=self.mu,
        )
        initial = Field(np.full(shape, self.volume_fraction), spacing=1.0)
        return DesignSpec(
            initial=initial,
            oracle=oracle,
            objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="zero-dissipation-limit")),
            optimizer=TopologyOptimizer(
                step_size=self.step_size, max_iter=self.max_iter, tol=-1.0, bounds=self.bounds
            ),
            name="stokes-channel",
        )


class ThermoElasticIntent(DesignIntent):
    """A structure under a mechanical load and a steady thermal gradient,
    minimising the coupled thermo-elastic objective. Built on
    :class:`ThermoElasticOracle` (left edge clamped and held at the sink
    temperature; a point load and a corner heat source)."""

    def __init__(
        self,
        nx: int,
        ny: int,
        load: float = -1.0,
        alpha_cte: float = 1.0,
        volume_fraction: float = 0.5,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        step_size: float = 2e-3,
        max_iter: int = 60,
        bounds=(1e-3, 1.0),
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.load = float(load)
        self.alpha_cte = float(alpha_cte)
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.bounds = bounds

    def build(self) -> DesignSpec:
        from morphos.physics.thermoelastic import ThermoElasticOracle

        shape = (self.ny, self.nx)
        nny, nnx = self.ny + 1, self.nx + 1
        fixed_dofs = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
        loads = {(nnx - 1, nny // 2, "y"): self.load}
        fixed_temps = [(0, j) for j in range(nny)]
        heat_sources = {(nnx - 1, nny - 1): 1.0}
        oracle = ThermoElasticOracle(
            shape=shape,
            fixed_dofs=fixed_dofs,
            loads=loads,
            fixed_temps=fixed_temps,
            heat_sources=heat_sources,
            thermal_expansion=self.alpha_cte,
            young_modulus=self.young_modulus,
            poisson_ratio=self.poisson_ratio,
        )
        initial = Field(np.full(shape, self.volume_fraction), spacing=1.0)
        return DesignSpec(
            initial=initial,
            oracle=oracle,
            objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid-isothermal-limit")),
            optimizer=TopologyOptimizer(
                step_size=self.step_size, max_iter=self.max_iter, tol=-1.0, bounds=self.bounds
            ),
            name="thermoelastic",
        )


class HeatExchangerIntent(DesignIntent):
    """Two-stage staggered intent: a Stokes channel optimisation followed by a
    conjugate-heat optimisation that consumes the solved velocity field. Returns
    a :class:`CoupledSpec` (see CoupledEngine)."""

    def __init__(
        self,
        nx: int,
        ny: int,
        mu: float = 1.0,
        u_max: float = 1.0,
        heat_source_W_m3: float = 1.0,
        k_solid: float = 1.0,
        k_fluid: float = 1e-2,
        rho_cp: float = 1.0,
        volume_fraction: float = 1.0,
        n_outer: int = 1,
        step_size: float = 1e-3,
        max_iter: int = 5,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.mu = float(mu)
        self.u_max = float(u_max)
        self.heat_source = float(heat_source_W_m3)
        self.k_solid = float(k_solid)
        self.k_fluid = float(k_fluid)
        self.rho_cp = float(rho_cp)
        self.volume_fraction = _check_volume_fraction(volume_fraction)
        self.n_outer = int(n_outer)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)

    def build(self) -> CoupledSpec:
        from morphos.physics.conjugate_heat import ConjugateHeatOracle
        from morphos.physics.stokes import StokesFlowOracle

        shape = (self.ny, self.nx)
        nny, nnx = self.ny + 1, self.nx + 1
        inlet = _edge_node_indices("left", nny, nnx)

        stokes = StokesFlowOracle(
            shape=shape,
            inlet=("left", _parabolic_inflow(self.u_max, axis=0)),
            noslip_edges=("top", "bottom"),
            viscosity=self.mu,
        )
        cht = ConjugateHeatOracle(
            shape=shape,
            velocity=np.zeros((nny, nnx, 2)),  # filled by passthrough from stage 0
            source=np.full(shape, self.heat_source),
            fixed_nodes=inlet,
            fixed_values=[0.0] * len(inlet),
            k_solid=self.k_solid,
            k_fluid=self.k_fluid,
            rho_cp=self.rho_cp,
        )
        opt = TopologyOptimizer(
            step_size=self.step_size, max_iter=self.max_iter, bounds=(0.0, 1.0)
        )
        return CoupledSpec(
            stages=[
                (stokes, MaximizeValue(), None),
                (cht, MaximizeValue(), None),
            ],
            optimizer=opt,
            initial_field=Field(np.full(shape, self.volume_fraction), spacing=1.0),
            passthrough={0: ["velocity"]},
            n_outer=self.n_outer,
            name="heat-exchanger",
        )
