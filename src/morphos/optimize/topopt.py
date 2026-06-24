"""Density-based topology optimization by gradient ascent.

The design variable is the field itself. Each iteration filters the raw design
(a cone/linear density filter, optional), applies a smooth Heaviside
projection (optional), projects the result through the manufacturability
constraint (if any), solves the physics, scores the objective, and steps the
field along the gradient. When the objective carries a gradient it is chained
back through the projection, the filter and the constraint (in that reverse
order) by a vector Jacobian product; otherwise the gradient is estimated by
finite differences.

SIMP p-continuation (ramping the penalization exponent from a near-convex
``p_start`` up to the target ``p_end`` over the early iterations) helps the
ascent escape the poor local optima that a fixed, sharply-penalized p falls
into from a naive uniform start. The exponent itself is consumed inside the
oracle (every SIMP oracle in this engine -- ``ElasticityOracle``,
``DarcyOracle``, ``ThermoelasticOracle`` -- already scales material response
by ``rho**self.penalty`` and differentiates analytically with respect to that
same ``self.penalty``), so the least invasive correct wiring is for the
optimizer to set ``oracle.penalty`` (or, for a ``MultiObjective``, every
oracle in its ``oracles`` list that exposes a ``penalty`` attribute) to the
current ramped value before each solve. Oracles without a ``penalty``
attribute (for example ``AnalyticOracle``) are left untouched, so continuation
is a transparent no-op for problems that have no SIMP exponent at all.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

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


def _set_penalty(oracle, objective, p: float) -> None:
    """Set the current SIMP exponent on every oracle that exposes ``penalty``.

    Supports a single oracle directly, or a ``MultiObjective`` that holds a
    list of oracles in ``objective.oracles``. Oracles without a ``penalty``
    attribute are left alone, making continuation a no-op for non-SIMP
    problems (for example the analytic reference oracle).
    """
    targets = getattr(objective, "oracles", None) or [oracle]
    for o in targets:
        if hasattr(o, "penalty"):
            o.penalty = float(p)


class TopologyOptimizer(Optimizer):
    def __init__(
        self,
        step_size: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        bounds: Optional[Tuple[float, float]] = None,
        fd_eps: float = 1e-6,
        p_start: float = 1.0,
        p_end: float = 1.0,
        p_ramp_fraction: float = 0.5,
        filter_radius: float = 0,
        beta_start: float = 1.0,
        beta_end: float = 1.0,
        patience: int = 1,
    ) -> None:
        """Gradient-ascent SIMP topology optimizer.

        Parameters
        ----------
        step_size:
            Fixed gradient-ascent step.
        max_iter:
            Maximum number of iterations.
        tol:
            FOM-change tolerance for the early-stop test (see ``patience``).
        bounds:
            Optional ``(lo, hi)`` clip applied to the raw design after each
            step.
        fd_eps:
            Step for the finite-difference gradient fallback.
        p_start, p_end, p_ramp_fraction:
            SIMP penalization continuation: the exponent ramps linearly from
            ``p_start`` to ``p_end`` over the first ``p_ramp_fraction`` of
            ``max_iter`` iterations, then holds at ``p_end``. Default
            ``p_start == p_end == 1.0`` makes continuation a no-op, matching
            the pre-continuation behavior (callers that want SIMP sharpening
            must opt in by passing ``p_end`` and the oracle's own default
            ``penalty``, or rely on the oracle's own fixed ``penalty`` and
            leave ``p_start == p_end`` unset, which never touches it).
        filter_radius:
            Radius (in voxels) of a linear cone/box density filter applied to
            the raw design before projection and before the oracle solve.
            ``0`` (the default) disables filtering entirely, preserving the
            pre-filter behavior exactly.
        beta_start, beta_end:
            Smooth Heaviside projection sharpness, applied after filtering;
            anneals linearly from ``beta_start`` to ``beta_end`` over the same
            schedule as the p-continuation ramp. Default
            ``beta_start == beta_end == 1.0`` keeps the projection close to
            identity-like (a mild smooth step) and, combined with
            ``filter_radius = 0``, leaves existing callers' behavior
            unchanged in shape; callers that want true 0/1 projection should
            pass ``beta_end`` larger (for example 16).
        patience:
            Number of CONSECUTIVE iterations the FOM change must stay below
            ``tol`` before the run is declared converged and stopped early.
            The default of 1 reproduces the original single-iteration
            convergence test.
        """
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bounds = bounds
        self.fd_eps = float(fd_eps)
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.p_ramp_fraction = float(p_ramp_fraction)
        self.filter_radius = float(filter_radius)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.patience = int(patience)

    # --- SIMP p / beta continuation schedule -----------------------------

    def _ramp_fraction(self, iteration: int) -> float:
        """Fraction (in [0, 1]) of the way through the p/beta ramp at
        ``iteration`` (0-based), holding at 1.0 once the ramp window ends."""
        ramp_iters = max(1, int(self.p_ramp_fraction * self.max_iter))
        return min(1.0, iteration / ramp_iters)

    def _current_p(self, iteration: int) -> float:
        t = self._ramp_fraction(iteration)
        return self.p_start + t * (self.p_end - self.p_start)

    def _current_beta(self, iteration: int) -> float:
        t = self._ramp_fraction(iteration)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    # --- Density filter (linear cone/box filter, self-adjoint) -----------

    def _filter_size(self) -> int:
        radius = int(round(self.filter_radius))
        return 2 * radius + 1

    def _apply_filter(self, values: np.ndarray) -> np.ndarray:
        """Cone/box density filter (uniform_filter, zero padded) on the raw
        design, the standard SIMP regularizer that removes isolated
        single-voxel spikes and checkerboarding. ``filter_radius <= 0``
        disables filtering and returns ``values`` unchanged."""
        if self.filter_radius <= 0:
            return values
        return ndimage.uniform_filter(
            values, size=self._filter_size(), mode="constant", cval=0.0
        )

    def _filter_vjp(self, grad: np.ndarray) -> np.ndarray:
        """Vector-Jacobian product of :meth:`_apply_filter`.

        The box filter is a linear, symmetric-kernel, zero-padded operator,
        so it is self-adjoint: its own transpose is the same filter applied
        to the incoming gradient (identical to the reasoning used by
        ``MinFeatureSize`` in ``morphos.manufacturing.constraints``).
        """
        if self.filter_radius <= 0:
            return grad
        return ndimage.uniform_filter(
            grad, size=self._filter_size(), mode="constant", cval=0.0
        )

    # --- Smooth Heaviside projection ---------------------------------------

    @staticmethod
    def _heaviside_project(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
        """Smooth Heaviside (tanh) projection sharpening a filtered density
        field toward 0/1, the standard companion to a density filter in SIMP
        continuation (Wang, Lazarov, Sigmund 2011 projection scheme)."""
        num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return num / den

    @staticmethod
    def _heaviside_vjp(
        rho: np.ndarray, beta: float, grad: np.ndarray, eta: float = 0.5
    ) -> np.ndarray:
        """Vector-Jacobian product of :meth:`_heaviside_project`.

        ``d(rho_proj)/d(rho) = beta * (1 - tanh(beta*(rho-eta))**2) / den``,
        an elementwise (diagonal Jacobian) scale, so the VJP is just that
        derivative multiplied into the incoming gradient.
        """
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        d = beta * (1.0 - np.tanh(beta * (rho - eta)) ** 2) / den
        return grad * d

    @staticmethod
    def _project(field: Field, constraint) -> Field:
        return field if constraint is None else constraint.project(field)

    def _design_chain(self, x: Field, beta: float, constraint):
        """Run the raw design ``x`` through filter -> Heaviside -> constraint,
        returning the final projected ``Field`` plus the intermediates needed
        to chain a gradient back through every stage in :meth:`_chain_vjp`.
        """
        filtered = self._apply_filter(x.values)
        projected_values = self._heaviside_project(filtered, beta)
        projected = x.like(projected_values)
        design = self._project(projected, constraint)
        return design, filtered

    def _chain_vjp(
        self, x: Field, filtered: np.ndarray, beta: float, grad: np.ndarray, constraint
    ) -> np.ndarray:
        """Chain a gradient (in design-Field space) back through the
        constraint projection, then the Heaviside projection, then the
        density filter, to gradient-in-raw-x space, in that reverse order."""
        g = grad
        if constraint is not None:
            g = constraint.vjp(x, g)
        g = self._heaviside_vjp(filtered, beta, g)
        g = self._filter_vjp(g)
        return g

    def _fom(self, x: Field, oracle, objective, constraint, beta: float = 1.0) -> float:
        design, _ = self._design_chain(x, beta, constraint)
        return _evaluate(design, oracle, objective).fom

    def _fd_gradient(
        self, x: Field, oracle, objective, constraint, beta: float = 1.0
    ) -> np.ndarray:
        base = self._fom(x, oracle, objective, constraint, beta)
        grad = np.zeros_like(x.values)
        it = np.nditer(x.values, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            xp = x.copy()
            xp.values[idx] += self.fd_eps
            grad[idx] = (self._fom(xp, oracle, objective, constraint, beta) - base) / self.fd_eps
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
        stall_count = 0

        for i in range(self.max_iter):
            iterations = i + 1
            p = self._current_p(i)
            beta = self._current_beta(i)
            _set_penalty(oracle, objective, p)

            design, filtered = self._design_chain(x, beta, constraint)
            ov = _evaluate(design, oracle, objective)
            history.append(ov.fom)

            # Track the best design seen. Fixed-step ascent can overshoot on a
            # non-convex problem, so the last design is not always the best.
            if ov.fom > best_fom:
                best_fom = ov.fom
                best_field = design.copy()

            if prev_fom is not None and abs(ov.fom - prev_fom) < self.tol:
                stall_count += 1
                if stall_count >= self.patience:
                    converged = True
                    break
            else:
                stall_count = 0
            prev_fom = ov.fom

            if ov.gradient is not None:
                g = ov.gradient
                g = self._chain_vjp(x, filtered, beta, g, constraint)
            else:
                used_fd = True
                g = self._fd_gradient(x, oracle, objective, constraint, beta)

            x.values = x.values + self.step_size * g
            if self.bounds is not None:
                np.clip(x.values, self.bounds[0], self.bounds[1], out=x.values)

        final_p = self._current_p(max(0, iterations - 1))
        final_beta = self._current_beta(max(0, iterations - 1))
        _set_penalty(oracle, objective, final_p)
        final_design, _ = self._design_chain(x, final_beta, constraint)
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
