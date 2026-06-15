"""PhysicsOracle: solves a physical problem on a geometry Field.

An oracle maps a Field to a :class:`PhysicsResult`: the raw solved quantity plus,
where available, the gradient of that quantity with respect to every field value
(the adjoint). The optimizer uses the gradient when present and falls back to
finite differences otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _dc_field
from typing import Optional

import numpy as np

from morphos.field import Field


@dataclass
class PhysicsResult:
    """The outcome of one physics solve.

    Attributes
    ----------
    value:
        The raw solved scalar quantity (the objective will turn this into a
        figure of merit).
    gradient:
        ``d value / d field_values`` with the same shape as the field, or
        ``None`` if the backend does not supply gradients.
    aux:
        Optional backend specific extras (fields, residuals, diagnostics).
    """

    value: float
    gradient: Optional[np.ndarray] = None
    aux: dict = _dc_field(default_factory=dict)


class PhysicsOracle(ABC):
    """Abstract base for a physics solver that scores a geometry Field."""

    #: whether :meth:`solve` returns a gradient in its result
    provides_gradient: bool = False

    @abstractmethod
    def solve(self, field: Field) -> PhysicsResult:
        raise NotImplementedError
