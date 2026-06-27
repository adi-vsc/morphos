"""Method of Moving Asymptotes (Svanberg 1987) topology optimizer.

The design variable is the raw element density field x in [xmin, xmax]. Each
iteration filters x (box density filter, optional), applies a smooth
Heaviside projection (optional), projects through the manufacturability
constraint (if any), solves the physics, scores the objective, and solves a
separable convex MMA subproblem (built from moving asymptotes L, U around the
current point) to produce the next x. The single constraint handled natively
is the mean-density volume fraction; its subproblem multiplier is found by
bisection, exactly as in OCOptimizer.

morphos maximizes; MMA (as classically posed) minimizes. Internally this
optimizer minimizes f0(x) = -fom(x), so df0/dx = -gradient. The volume
constraint is g1(x) = mean(x) - volume_fraction <= 0, with df1/dx_j = 1/N.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from morphos.field import Field
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import (
    _evaluate,
    _set_penalty,
    TopologyOptimizer,
)

_ASYMPTOTE_SHRINK = 0.7
_ASYMPTOTE_GROW = 1.2
_MOVE_MIN = 0.01
_MOVE_MAX = 10.0
_ALBE_FRAC = 0.1
_RAW_EPS = 1e-4


class MMAOptimizer(Optimizer):
    def __init__(
        self,
        volume_fraction: float,
        max_iter: int = 200,
        tol: float = 1e-4,
        filter_radius: float = 0,
        beta_start: float = 1.0,
        beta_end: float = 8.0,
        p_start: float = 1.0,
        p_end: float = 3.0,
        p_ramp_fraction: float = 0.4,
        xmin: float = 1e-3,
        xmax: float = 1.0,
    ) -> None:
        self.volume_fraction = float(volume_fraction)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.filter_radius = float(filter_radius)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.p_ramp_fraction = float(p_ramp_fraction)
        self.xmin = float(xmin)
        self.xmax = float(xmax)

        # Reuse the filter / projection / chain-rule machinery from
        # TopologyOptimizer instead of duplicating it.
        self._shared = TopologyOptimizer(
            filter_radius=filter_radius,
            beta_start=beta_start,
            beta_end=beta_end,
            p_start=p_start,
            p_end=p_end,
            p_ramp_fraction=p_ramp_fraction,
            max_iter=max_iter,
        )

        self.xold1: Optional[np.ndarray] = None
        self.xold2: Optional[np.ndarray] = None
        self.low: Optional[np.ndarray] = None
        self.upp: Optional[np.ndarray] = None

    # --- delegate filter/projection/chain-rule to TopologyOptimizer ---------

    def _design_chain(self, x: Field, beta: float, constraint):
        return self._shared._design_chain(x, beta, constraint)

    def _chain_vjp(self, x: Field, filtered, beta: float, grad, constraint):
        return self._shared._chain_vjp(x, filtered, beta, grad, constraint)

    def _current_p(self, iteration: int) -> float:
        return self._shared._current_p(iteration)

    def _current_beta(self, iteration: int) -> float:
        return self._shared._current_beta(iteration)

    # --- asymptote update ----------------------------------------------------

    def _update_asymptotes(self, x: np.ndarray, iteration: int) -> None:
        span = self.xmax - self.xmin
        if iteration < 2 or self.xold1 is None or self.xold2 is None:
            self.low = x - 0.5 * span
            self.upp = x + 0.5 * span
            return

        sign = (x - self.xold1) * (self.xold1 - self.xold2)
        gamma = np.where(sign > 0, _ASYMPTOTE_GROW, np.where(sign < 0, _ASYMPTOTE_SHRINK, 1.0))

        low = x - gamma * (self.xold1 - self.low)
        upp = x + gamma * (self.upp - self.xold1)

        low = np.clip(low, x - _MOVE_MAX * span, x - _MOVE_MIN * span)
        upp = np.clip(upp, x + _MOVE_MIN * span, x + _MOVE_MAX * span)

        self.low, self.upp = low, upp

    # --- separable convex subproblem -----------------------------------------

    def _subproblem_coeffs(self, x: np.ndarray, df: np.ndarray):
        low, upp = self.low, self.upp
        um_x = upp - x
        xm_l = x - low
        p = np.clip(df, 0.0, None) * um_x ** 2 + _RAW_EPS * um_x ** 2
        q = np.clip(-df, 0.0, None) * xm_l ** 2 + _RAW_EPS * xm_l ** 2
        return p, q

    def _x_star(self, lam: float, p0, q0, p1, q1, alpha, clip_hi) -> np.ndarray:
        # Stationary point of p/(U-x) + q/(x-L): sqrt(p)*(x-L) = sqrt(q)*(U-x),
        # i.e. x* = (sqrt(q)*U + sqrt(p)*L) / (sqrt(p) + sqrt(q)). A large p
        # (steep penalty against approaching U) pulls x* toward L, and vice
        # versa, so the U/L pairing is with sqrt(q)/sqrt(p) respectively.
        P = p0 + lam * p1
        Q = q0 + lam * q1
        sqrtP = np.sqrt(P)
        sqrtQ = np.sqrt(Q)
        x = (sqrtQ * self.upp + sqrtP * self.low) / (sqrtP + sqrtQ)
        return np.clip(x, alpha, clip_hi)

    def _bisect_lambda(self, p0, q0, p1, q1, alpha, clip_hi, df1_flat, rhs) -> np.ndarray:
        # df1 >= 0 everywhere (chain VJP through non-negative Heaviside+filter),
        # so dot(df1, x*(lambda)) is monotone decreasing in lambda.
        # Bisect to enforce the linearized volume constraint: dot(df1, x*) = rhs.
        lam_lo, lam_hi = 0.0, 1e9
        for _ in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            x_mid = self._x_star(lam_mid, p0, q0, p1, q1, alpha, clip_hi)
            if np.dot(df1_flat, x_mid.ravel()) > rhs:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        return self._x_star(0.5 * (lam_lo + lam_hi), p0, q0, p1, q1, alpha, clip_hi)

    def _mma_step(
        self,
        x: Field,
        df0: np.ndarray,
        filtered: np.ndarray,
        hv_beta: float,
        design: Field,
        constraint,
    ) -> np.ndarray:
        x_vals = x.values
        alpha = np.maximum(self.xmin, self.low + _ALBE_FRAC * (x_vals - self.low))
        clip_hi = np.minimum(self.xmax, self.upp - _ALBE_FRAC * (self.upp - x_vals))

        N = x_vals.size
        # Volume constraint gradient in raw-x space via chain VJP through
        # Heaviside (and filter if active). This correctly handles the
        # Heaviside volume shift when beta >> 1, so the MMA subproblem
        # approximates mean(projected(x)) = V* rather than mean(raw_x) = V*.
        ones_field = np.ones_like(x_vals) / N
        df1 = self._chain_vjp(x, filtered, hv_beta, ones_field, constraint)

        p0, q0 = self._subproblem_coeffs(x_vals, df0)
        p1, q1 = self._subproblem_coeffs(x_vals, df1)

        # Linearized constraint RHS: df1^T x*(λ) = df1^T x_k - g1(x_k)
        # where g1(x_k) = mean(design) - V* (current projected volume deficit).
        rhs = np.dot(df1.ravel(), x_vals.ravel()) - (np.mean(design.values) - self.volume_fraction)
        return self._bisect_lambda(p0, q0, p1, q1, alpha, clip_hi, df1.ravel(), rhs)

    # --- public API ------------------------------------------------------------

    def run(
        self,
        initial: Field,
        oracle,
        objective,
        constraint=None,
        on_iteration: Optional[Callable] = None,
    ) -> OptimizeResult:
        x: Field = initial.copy()
        history: List[float] = []
        best_fom: float = -np.inf
        best_field: Optional[Field] = None
        converged: bool = False
        prev_fom: Optional[float] = None

        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None

        iterations = 0
        for i in range(self.max_iter):
            iterations = i + 1
            p = self._current_p(i)
            beta = self._current_beta(i)
            _set_penalty(oracle, objective, p)

            design, filtered = self._design_chain(x, beta, constraint)
            ov = _evaluate(design, oracle, objective)
            if not np.isfinite(ov.fom):
                raise RuntimeError(
                    f"non-finite figure of merit ({ov.fom}) at iteration "
                    f"{iterations}; the physics solve diverged"
                )
            history.append(ov.fom)

            if ov.fom > best_fom:
                best_fom = ov.fom
                best_field = design.copy()

            delta = abs(ov.fom - prev_fom) if prev_fom is not None else 0.0
            if on_iteration is not None:
                on_iteration(iterations, ov.fom, delta, p, beta)

            if prev_fom is not None and delta < self.tol:
                converged = True
                prev_fom = ov.fom
                break
            prev_fom = ov.fom

            if ov.gradient is None:
                raise ValueError(
                    "MMAOptimizer requires an analytic gradient but the "
                    "objective returned gradient=None."
                )
            g = self._chain_vjp(x, filtered, beta, ov.gradient, constraint)
            if not np.all(np.isfinite(g)):
                raise RuntimeError(
                    f"non-finite gradient at iteration {iterations}; the "
                    f"physics solve diverged"
                )
            df0 = -g

            self._update_asymptotes(x.values, i)
            x_new = self._mma_step(x, df0, filtered, beta, design, constraint)

            self.xold2 = self.xold1
            self.xold1 = x.values.copy()
            x = x.like(np.clip(x_new, self.xmin, self.xmax))

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
            used_finite_differences=False,
            converged=converged,
        )
