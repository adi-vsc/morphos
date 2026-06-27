"""DesignSpec and DesignResult: the inputs and outputs of the engine.

A DesignSpec wires together the swappable pieces (geometry, physics, objective,
optimizer, manufacturability) for one design run. A DesignResult carries the
optimized geometry and the gap to a user-supplied reference bound.
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


def _optimizer_params(optimizer) -> dict:
    """JSON-safe constructor parameters of an optimizer, recovered by matching
    its ``__init__`` signature against its stored attributes (the topology
    optimizer stores every parameter under its own name). Paths and tuples are
    coerced to strings/lists; a parameter with no matching attribute is skipped.
    """
    import inspect

    params: dict = {}
    try:
        sig = inspect.signature(type(optimizer).__init__)
    except (TypeError, ValueError):
        return params
    for name in sig.parameters:
        if name == "self" or not hasattr(optimizer, name):
            continue
        value = getattr(optimizer, name)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        if isinstance(value, (str, int, float, bool, list, type(None))):
            params[name] = value
    return params


@dataclass
class DesignSpec:
    """Topology design: the Field itself is the design variable."""

    initial: Field
    oracle: PhysicsOracle
    objective: Objective
    optimizer: Optimizer
    constraint: object = None
    name: str = ""

    def to_dict(self) -> dict:
        """A JSON-safe structural description of this spec (Shape B).

        Captures the parts that survive serialisation: the name, the initial
        density field, the optimizer's constructor parameters, and the class
        names of the oracle, objective and constraint. It deliberately does NOT
        attempt to serialise the oracle's boundary conditions: an oracle
        consumes its loads/supports into solver-internal state at construction
        and cannot reproduce them, so a raw-spec dict is structural metadata,
        not a lossless round trip. To persist a fully reconstructable design,
        serialise the originating :class:`~morphos.intent.DesignIntent`
        (Shape A) instead.
        """
        return {
            "name": self.name,
            "initial": {
                "values": np.asarray(self.initial.values).tolist(),
                "spacing": float(np.asarray(self.initial.spacing).reshape(-1)[0]),
            },
            "optimizer": {
                "class": type(self.optimizer).__name__,
                "params": _optimizer_params(self.optimizer),
            },
            "oracle": {"class": type(self.oracle).__name__},
            "objective": {"class": type(self.objective).__name__},
            "constraint": (
                None if self.constraint is None else {"class": type(self.constraint).__name__}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DesignSpec":
        """Reconstruct a DesignSpec from a Shape B dict.

        Because an oracle's boundary conditions are not serialisable (see
        :meth:`to_dict`), this rebuilds only specs whose dict carries enough to
        reconstruct the oracle; for the structural metadata emitted by
        :meth:`to_dict` it raises a clear error directing the caller to the
        intent (Shape A) form, which is the lossless path.
        """
        raise ValueError(
            "cannot reconstruct a DesignSpec from a raw-spec dict: an oracle's "
            "boundary conditions (loads, supports, sources) are consumed into "
            "solver-internal state at construction and are not recoverable from "
            "the structural metadata in 'spec'. Expected an intent spec (shape "
            "A: an 'intent' key naming a DesignIntent), found a raw 'spec' dict. "
            "Serialise the originating DesignIntent instead."
        )


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
    outer_tol: float = 1e-6


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
