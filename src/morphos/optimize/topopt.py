"""Density-based topology optimization by gradient ascent.

The design variable is the field itself. Each iteration projects the design
through the manufacturability constraint (if any), solves the physics, scores the
objective, and steps the field along the gradient. When the objective carries a
gradient it is chained back through the constraint projection by a vector
Jacobian product; otherwise the gradient is estimated by finite differences.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from morphos.field import Field
from morphos.objective.objective import ObjectiveValue
from morphos.optimize.optimizer import Optimizer, OptimizeResult


def _evaluate(design: Field, oracle, objective) -> ObjectiveValue:
    """Score a design, transparently supporting either a single ``Objective``
    (evaluated on ``oracle.solve``) or a ``MultiObjective`` (which runs its own
    oracles and returns the scalarised value/gradient)."""
    if hasattr(objective, "evaluate_design"):  # MultiObjective duck type
        ov = objective.evaluate_design(design)
        return ObjectiveValue(fom=ov.scalar_fom, gradient=ov.scalar_grad)
    return objective.evaluate(oracle.solve(design))


class TopologyOptimizer(Optimizer):
    def __init__(
        self,
        step_size: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        bounds: Optional[Tuple[float, float]] = None,
        fd_eps: float = 1e-6,
    ) -> None:
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bounds = bounds
        self.fd_eps = float(fd_eps)

    @staticmethod
    def _project(field: Field, constraint) -> Field:
        return field if constraint is None else constraint.project(field)

    def _fom(self, x: Field, oracle, objective, constraint) -> float:
        design = self._project(x, constraint)
        return _evaluate(design, oracle, objective).fom

    def _fd_gradient(self, x: Field, oracle, objective, constraint) -> np.ndarray:
        base = self._fom(x, oracle, objective, constraint)
        grad = np.zeros_like(x.values)
        it = np.nditer(x.values, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            xp = x.copy()
            xp.values[idx] += self.fd_eps
            grad[idx] = (self._fom(xp, oracle, objective, constraint) - base) / self.fd_eps
            it.iternext()
        return grad

    def run(self, initial: Field, oracle, objective, constraint=None) -> OptimizeResult:
        x = initial.copy()
        history = []
        used_fd = False
        converged = False
        prev_fom = None
        iterations = 0
        best_fom = -np.inf
        best_field = None

        for i in range(self.max_iter):
            iterations = i + 1
            design = self._project(x, constraint)
            ov = _evaluate(design, oracle, objective)
            history.append(ov.fom)

            # Track the best design seen. Fixed-step ascent can overshoot on a
            # non-convex problem, so the last design is not always the best.
            if ov.fom > best_fom:
                best_fom = ov.fom
                best_field = design.copy()

            if prev_fom is not None and abs(ov.fom - prev_fom) < self.tol:
                converged = True
                break
            prev_fom = ov.fom

            if ov.gradient is not None:
                g = ov.gradient
                if constraint is not None:
                    g = constraint.vjp(x, g)
            else:
                used_fd = True
                g = self._fd_gradient(x, oracle, objective, constraint)

            x.values = x.values + self.step_size * g
            if self.bounds is not None:
                np.clip(x.values, self.bounds[0], self.bounds[1], out=x.values)

        final_design = self._project(x, constraint)
        final_fom = _evaluate(final_design, oracle, objective).fom
        if final_fom > best_fom:
            best_fom = final_fom
            best_field = final_design.copy()

        return OptimizeResult(
            field=best_field,
            fom=best_fom,
            history=history,
            iterations=iterations,
            used_finite_differences=used_fd,
            converged=converged,
        )
