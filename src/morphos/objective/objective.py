"""Objective: turns a physics result into a figure of merit, and knows the limit.

The objective maps a :class:`~morphos.physics.oracle.PhysicsResult` to a scalar
figure of merit that the optimizer maximizes, carrying the gradient through by
the chain rule. It can also expose a :class:`PhysicalBound`, the best value
physics allows, so a result can report how close to the limit it reached. That
margin to the limit is a first-class output of the engine, not an afterthought.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from morphos.physics.oracle import PhysicsResult


@dataclass
class ObjectiveValue:
    """A figure of merit and its gradient with respect to field values."""

    fom: float
    gradient: Optional[np.ndarray] = None


@dataclass
class PhysicalBound:
    """The best figure of merit physics allows for a problem.

    ``value`` is the attainable ceiling of the (maximized) figure of merit.
    """

    value: float
    name: str

    def margin(self, fom: float) -> float:
        """Gap from a figure of merit up to the ceiling (positive is below)."""
        return self.value - fom

    def attained_fraction(self, fom: float) -> Optional[float]:
        """Fraction of the ceiling attained, or None when the ceiling is zero."""
        if abs(self.value) < 1e-300:
            return None
        return fom / self.value


class Objective(ABC):
    """Abstract base for a figure of merit over a physics result."""

    @abstractmethod
    def evaluate(self, result: PhysicsResult) -> ObjectiveValue:
        raise NotImplementedError

    def bound(self) -> Optional[PhysicalBound]:
        """The physical limit for this objective, or None if unknown."""
        return None


class MaximizeValue(Objective):
    """Use the raw physics value as the figure of merit, optionally scaled."""

    def __init__(self, scale: float = 1.0, bound: Optional[PhysicalBound] = None) -> None:
        self.scale = float(scale)
        self._bound = bound

    def evaluate(self, result: PhysicsResult) -> ObjectiveValue:
        grad = None if result.gradient is None else self.scale * result.gradient
        return ObjectiveValue(fom=self.scale * result.value, gradient=grad)

    def bound(self) -> Optional[PhysicalBound]:
        return self._bound
