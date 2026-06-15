"""Optimizer: the inverse-design loop.

An optimizer searches a design (a Field) that maximizes an objective evaluated
through a physics oracle, optionally subject to a manufacturability constraint.
It uses the objective gradient (the adjoint) when present and falls back to
finite differences otherwise, recording which it used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _dc_field
from typing import List, Optional

import numpy as np

from morphos.field import Field


@dataclass
class OptimizeResult:
    field: Field
    fom: float
    history: List[float] = _dc_field(default_factory=list)
    iterations: int = 0
    used_finite_differences: bool = False
    converged: bool = False
    params: Optional[np.ndarray] = None


class Optimizer(ABC):
    @abstractmethod
    def run(self, initial: Field, oracle, objective, constraint=None) -> OptimizeResult:
        raise NotImplementedError
