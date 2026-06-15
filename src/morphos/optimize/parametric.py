"""Parametric design: optimize a small parameter vector through a geometry kernel.

Where topology optimization treats every voxel as a design variable, parametric
design optimizes a handful of meaningful parameters (a radius, a wall thickness,
a set of control points) that a geometry kernel turns into a Field. This is the
mode that fits hand-meaningful device geometry.

Geometry kernels are generally not differentiable, and a single parameter
accumulates sensitivity over the whole grid, so the gradient scale depends on
resolution. A fixed-step ascent is therefore the wrong tool. Because the
parameters are few, a robust bounded quasi-Newton optimizer (L-BFGS-B with a
numerical gradient) is both appropriate and scale tolerant.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from morphos.field import Field
from morphos.optimize.optimizer import OptimizeResult


class ParametricOptimizer:
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-9,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bounds = bounds

    def _forward(self, params, build, constraint) -> Field:
        field = build(np.asarray(params, dtype=float))
        if constraint is not None:
            field = constraint.project(field)
        return field

    def _fom(self, params, build, oracle, objective, constraint) -> float:
        field = self._forward(params, build, constraint)
        return objective.evaluate(oracle.solve(field)).fom

    def run(
        self,
        initial_params,
        build: Callable[[np.ndarray], Field],
        oracle,
        objective,
        constraint=None,
    ) -> OptimizeResult:
        x0 = np.array(initial_params, dtype=float)
        history = []

        def negative_fom(p):
            f = self._fom(p, build, oracle, objective, constraint)
            history.append(f)
            return -f

        bounds = None
        if self.bounds is not None:
            bounds = [(self.bounds[0], self.bounds[1])] * x0.size

        result = minimize(
            negative_fom,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        x = np.asarray(result.x, dtype=float)
        final_field = self._forward(x, build, constraint)
        final_fom = objective.evaluate(oracle.solve(final_field)).fom
        return OptimizeResult(
            field=final_field,
            fom=final_fom,
            history=history,
            iterations=int(result.nit),
            used_finite_differences=True,
            converged=bool(result.success),
            params=x,
        )
