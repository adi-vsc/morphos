"""DesignSpec and DesignResult: the inputs and outputs of the engine.

A DesignSpec wires together the swappable pieces (geometry, physics, objective,
optimizer, manufacturability) for one design run. A DesignResult carries the
optimized geometry and, crucially, the margin to the physical limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from morphos.field import Field
from morphos.objective.objective import Objective, PhysicalBound
from morphos.optimize.optimizer import Optimizer
from morphos.physics.oracle import PhysicsOracle


@dataclass
class DesignSpec:
    """Topology design: the Field itself is the design variable."""

    initial: Field
    oracle: PhysicsOracle
    objective: Objective
    optimizer: Optimizer
    constraint: object = None
    name: str = ""


@dataclass
class ParametricSpec:
    """Parametric design: a small parameter vector built into a Field by a kernel."""

    initial_params: np.ndarray
    build: Callable[[np.ndarray], Field]
    oracle: PhysicsOracle
    objective: Objective
    optimizer: object
    constraint: object = None
    name: str = ""


@dataclass
class CoupledSpec:
    """Specification for a sequentially (staggered) coupled multi-oracle run.

    Each stage is an ``(oracle, objective, constraint)`` tuple optimised in order
    on a single shared design ``Field``. ``passthrough`` maps a stage index to the
    list of ``PhysicsResult.aux`` attributes that stage forwards into the *next*
    stage's oracle (set as attributes on it), e.g. ``{0: ["velocity"]}`` passes the
    velocity field solved in stage 0 into stage 1's oracle. The whole sequence is
    repeated ``n_outer`` times (the outer staggered loop). This is the
    operator-split approach; monolithic (simultaneous) coupling is deferred.

    A single ``optimizer`` is shared across stages (every stage optimises the same
    density Field, so one topology optimiser suffices); per-stage objectives and
    constraints differ.

    ``coupling_mode`` selects the outer coupling strategy: ``"staggered"``
    (default, preserves existing behaviour) runs the stage loop described
    above; ``"monolithic"`` instead solves every stage's oracle as a single
    one-shot block system per outer iteration (see
    :class:`morphos.physics.coupled.MonolithicCoupledOracle`), appropriate
    when a stage's oracle is itself a monolithic coupled oracle.
    """

    stages: List[Tuple[PhysicsOracle, Objective, object]]
    optimizer: Optimizer
    initial_field: Field
    passthrough: Dict[int, List[str]] = _dc_field(default_factory=dict)
    n_outer: int = 1
    coupling_mode: Literal["staggered", "monolithic"] = "staggered"
    name: str = ""


@dataclass
class DesignResult:
    field: Field
    figure_of_merit: float
    bound: Optional[PhysicalBound] = None
    margin: Optional[float] = None
    attained_fraction: Optional[float] = None
    history: List[float] = _dc_field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    used_finite_differences: bool = False
    manufacturability: dict = _dc_field(default_factory=dict)
    # Geometry-export state, populated by morphos.manufacturing.export.export_bundle
    # (or by Engine.run(..., export=True)). Latent until the design is materialised.
    mesh_path: Optional[Path] = None
    vdb_path: Optional[Path] = None
    manufacturing_bundle: Optional[Any] = None

    @property
    def is_exported(self) -> bool:
        return self.mesh_path is not None
