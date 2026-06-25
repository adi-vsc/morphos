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

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from morphos.field import Field
from morphos.optimize.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from morphos.optimize.optimizer import OptimizeResult


class ParametricOptimizer:
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-9,
        bounds: Optional[Tuple[float, float]] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 10,
        resume_from: Optional[Path] = None,
    ) -> None:
        """L-BFGS-B parameter-vector optimizer.

        ``checkpoint_dir``/``checkpoint_every``/``resume_from`` mirror
        :class:`~morphos.optimize.topopt.TopologyOptimizer`'s checkpointing
        contract for API consistency, but L-BFGS-B has no SIMP penalty/beta
        continuation schedule, so the checkpoint's ``p``/``beta`` slots are
        unused placeholders (``0.0``) here; what is actually warm-started is
        the parameter vector itself (stored in the checkpoint's field slot,
        as a 1-D Field) and the FOM history accumulated so far.
        """
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bounds = bounds
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = int(checkpoint_every)
        self.resume_from = Path(resume_from) if resume_from is not None else None

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
        on_iteration=None,
    ) -> OptimizeResult:
        # ``on_iteration`` is accepted for a uniform Engine.run() optimizer
        # contract. The L-BFGS-B path is driven by scipy.optimize.minimize and
        # does not expose a per-evaluation hook with the (fom, delta, p, beta)
        # signature the topology optimizer reports, so the callback is left
        # unwired here rather than fed misleading values.
        if self.resume_from is not None:
            # Warm start: resume the parameter vector (stored as a 1-D Field
            # in the checkpoint's field slot) and history; L-BFGS-B itself
            # carries no other persistent state between independent calls to
            # minimize, so restoring x0 and the FOM history fully resumes
            # the search from where it left off.
            param_field, _, _, _, history = load_checkpoint(self.resume_from)
            x0 = np.asarray(param_field.values, dtype=float)
        else:
            x0 = np.array(initial_params, dtype=float)
            history = []

        iteration = [len(history)]

        def negative_fom(p):
            f = self._fom(p, build, oracle, objective, constraint)
            history.append(f)
            return -f

        def _checkpoint_callback(xk):
            iteration[0] += 1
            if (
                self.checkpoint_dir is not None
                and iteration[0] % self.checkpoint_every == 0
            ):
                save_checkpoint(
                    checkpoint_path(self.checkpoint_dir),
                    Field(np.asarray(xk, dtype=float), spacing=1.0),
                    iteration[0], p=0.0, beta=0.0, history=history,
                )

        bounds = None
        if self.bounds is not None:
            bounds = [(self.bounds[0], self.bounds[1])] * x0.size

        callback = _checkpoint_callback if self.checkpoint_dir is not None else None
        result = minimize(
            negative_fom,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
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
