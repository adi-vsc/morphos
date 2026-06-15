"""DesignSpec and DesignResult: the inputs and outputs of the engine.

A DesignSpec wires together the swappable pieces (geometry, physics, objective,
optimizer, manufacturability) for one design run. A DesignResult carries the
optimized geometry and, crucially, the margin to the physical limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import List, Optional

from morphos.field import Field
from morphos.objective.objective import Objective, PhysicalBound
from morphos.optimize.optimizer import Optimizer
from morphos.physics.oracle import PhysicsOracle


@dataclass
class DesignSpec:
    initial: Field
    oracle: PhysicsOracle
    objective: Objective
    optimizer: Optimizer
    constraint: object = None
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
