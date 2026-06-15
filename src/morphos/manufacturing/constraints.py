"""Manufacturability constraints and projections.

A constraint can project a design Field onto the manufacturable set (for example
removing features smaller than a printer can resolve), provide the vector
Jacobian product of that projection so the optimizer can chain gradients through
it, and report manufacturability diagnostics for the final result.
"""

from __future__ import annotations

from abc import ABC

import numpy as np
from scipy import ndimage

from morphos.field import Field


class ManufacturabilityConstraint(ABC):
    """Base class. Defaults make a constraint a no-op identity projection."""

    def project(self, field: Field) -> Field:
        return field

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        """Vector Jacobian product of :meth:`project` at ``field``.

        For an identity projection this is the gradient unchanged.
        """
        return grad

    def report(self, field: Field) -> dict:
        return {}


class MinFeatureSize(ManufacturabilityConstraint):
    """A linear density filter that removes features below a radius.

    Implemented as a normalized box filter with zero padding, which is a
    self-adjoint linear operator, so its vector Jacobian product is the same
    filter applied to the incoming gradient.
    """

    def __init__(self, radius: int) -> None:
        self.radius = int(radius)
        self.size = 2 * self.radius + 1

    def _filter(self, values: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(
            values, size=self.size, mode="constant", cval=0.0
        )

    def project(self, field: Field) -> Field:
        return field.like(self._filter(field.values))

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        return self._filter(grad)


class Connectivity(ManufacturabilityConstraint):
    """Reports whether the solid region is a single connected component.

    This is a diagnostic constraint: its projection and vjp are identity. The
    solid region is taken as values at or above ``threshold``.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def report(self, field: Field) -> dict:
        mask = field.values >= self.threshold
        _, num = ndimage.label(mask)
        return {"num_components": int(num), "connected": bool(num == 1)}
