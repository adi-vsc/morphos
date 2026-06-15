"""Analytic physics oracle: a closed-form stand-in for a real solver.

This oracle exists to bring the whole engine pipeline up green without any heavy
dependency, and to verify gradient flow exactly. It is not physics; the real
electromagnetic backend lives in :mod:`morphos.physics.ceviche_em`.

The response is a negative weighted sum of squared error against a target field:

    value = - sum( weight * (field - target) ** 2 )

so the value is maximized (and equals zero) when the field equals the target.
The gradient is exact, which makes this the reference problem for testing the
optimizer and the end to end engine.
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult


class AnalyticOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(self, target: np.ndarray, weight: float = 1.0) -> None:
        self.target = np.asarray(target, dtype=float)
        self.weight = float(weight)

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.target.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match target "
                f"shape {self.target.shape}"
            )
        diff = field.values - self.target
        value = -float(np.sum(self.weight * diff ** 2))
        gradient = -2.0 * self.weight * diff
        return PhysicsResult(value=value, gradient=gradient)
