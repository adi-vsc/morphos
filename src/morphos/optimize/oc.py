"""Optimality-Criteria (OC) topology optimizer.

The design variable is the element density field rho in [rho_min, 1].

Each iteration:
  1. Set SIMP penalty (p-continuation schedule, same as TopologyOptimizer).
  2. Filter rho with a box density filter (optional; filter_radius=0 disables it).
  3. Evaluate fom and gradient g = d(fom)/d(rho) through the physics oracle.
  4. Chain the gradient back through the self-adjoint filter VJP.
  5. Apply the OC multiplicative update with bisection to enforce the target
     volume fraction V* exactly.
  6. Track and return the BEST design seen (highest fom), not the last.

The OC update at Lagrange multiplier lambda is:

    B_e  = clip( N * g_e / lambda, 0, inf )   # guard against negative sensitivity
    lo_e = max(rho_min, rho_e - move)
    hi_e = min(1,       rho_e + move)
    rho_new_e = clip( rho_e * B_e**eta, lo_e, hi_e )

where N = number of elements, eta = damping exponent (default 0.5), and
move is the per-iteration move limit (default 0.2). Lambda is found by
bisection over [1e-9, 1e9] (~60 iterations) so that mean(rho_new) == V*.

This is the Bendsoe-Sigmund 88-line canonical method. It eliminates the
need for a hand-tuned step size and drives densities toward 0/1 (crisp),
in contrast to the fixed-step gradient ascent in TopologyOptimizer.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy import ndimage

from morphos.field import Field
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import _evaluate, _set_penalty


class OCOptimizer(Optimizer):
    """Optimality-Criteria topology optimizer with bisection volume control.

    Parameters
    ----------
    volume_fraction:
        Target volume fraction V* in (0, 1].
    max_iter:
        Maximum number of iterations.
    move:
        Per-iteration move limit on element density changes.
    eta:
        Damping exponent for the OC update (B_e**eta). Canonical value: 0.5.
    rho_min:
        Minimum density floor to keep the stiffness matrix nonsingular.
    filter_radius:
        Radius (voxels for "box", physical length for "pde") of the density
        filter applied before evaluation. 0 (default) disables filtering.
    filter_type:
        "box" (default) uses the directionally-biased uniform_filter
        cone/box filter, preserving existing behavior exactly. "pde" uses
        the isotropic Helmholtz PDE density filter (Lazarov & Sigmund 2016,
        see morphos.optimize.pde_filter.PDEFilter) instead.
    filter_h:
        Physical grid spacing used to assemble the PDE filter's Laplacian
        when filter_type == "pde". Ignored for "box".
    p_start, p_end, p_ramp_fraction:
        SIMP p-continuation schedule. Identical semantics to TopologyOptimizer.
    tol:
        FOM-change tolerance for early convergence.
    """

    def __init__(
        self,
        volume_fraction: float,
        max_iter: int = 50,
        move: float = 0.2,
        eta: float = 0.5,
        rho_min: float = 1e-3,
        filter_radius: float = 0.0,
        filter_type: str = "box",
        filter_h: float = 1.0,
        p_start: float = 1.0,
        p_end: float = 3.0,
        p_ramp_fraction: float = 0.5,
        tol: float = 1e-4,
    ) -> None:
        self.volume_fraction = float(volume_fraction)
        self.max_iter = int(max_iter)
        self.move = float(move)
        self.eta = float(eta)
        self.rho_min = float(rho_min)
        self.filter_radius = float(filter_radius)
        self.filter_type = str(filter_type)
        self.filter_h = float(filter_h)
        self._pde_filter = None
        self._pde_filter_shape = None
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.p_ramp_fraction = float(p_ramp_fraction)
        self.tol = float(tol)

    # --- SIMP p continuation schedule (mirrors TopologyOptimizer) -----------

    def _current_p(self, iteration: int) -> float:
        ramp_iters = max(1, int(self.p_ramp_fraction * self.max_iter))
        t = min(1.0, iteration / ramp_iters)
        return self.p_start + t * (self.p_end - self.p_start)

    # --- Self-adjoint box density filter ------------------------------------

    def _filter_size(self) -> int:
        return 2 * int(round(self.filter_radius)) + 1

    def _get_pde_filter(self, shape):
        if self._pde_filter is None or self._pde_filter_shape != shape:
            from morphos.optimize.pde_filter import PDEFilter

            self._pde_filter = PDEFilter(shape, self.filter_h, self.filter_radius)
            self._pde_filter_shape = shape
        return self._pde_filter

    def _apply_filter(self, values: np.ndarray) -> np.ndarray:
        if self.filter_radius <= 0:
            return values
        if self.filter_type == "pde":
            return self._get_pde_filter(values.shape).apply(values)
        return ndimage.uniform_filter(
            values, size=self._filter_size(), mode="constant", cval=0.0
        )

    def _filter_vjp(self, grad: np.ndarray) -> np.ndarray:
        """VJP of the density filter. Both the box filter and the PDE filter
        are self-adjoint, so VJP = the same filtering operation."""
        if self.filter_radius <= 0:
            return grad
        if self.filter_type == "pde":
            return self._get_pde_filter(grad.shape).vjp(grad)
        return ndimage.uniform_filter(
            grad, size=self._filter_size(), mode="constant", cval=0.0
        )

    # --- OC multiplicative update -------------------------------------------

    def _oc_update(self, rho: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
        """OC update for a given Lagrange multiplier lambda.

        dV/drho_e = 1/N for the uniform mean-density volume measure, so
        B_e = g_e / (lambda * 1/N) = N * g_e / lambda.
        """
        N = rho.size
        B = np.clip(N * g / lam, 0.0, None)  # guard negative sensitivity
        rho_new = rho * (B ** self.eta)
        lo = np.maximum(self.rho_min, rho - self.move)
        hi = np.minimum(1.0, rho + self.move)
        rho_new = np.clip(rho_new, lo, hi)
        return np.clip(rho_new, self.rho_min, 1.0)  # safety clip

    def _bisect_lambda(self, rho: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Bisect lambda in [1e-9, 1e9] so that mean(rho_new) == volume_fraction.

        mean(rho_new(lambda)) is monotone decreasing in lambda, guaranteeing a
        unique solution when V* is inside the achievable range.
        """
        lam_lo, lam_hi = 1e-9, 1e9
        target = self.volume_fraction
        for _ in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if np.mean(self._oc_update(rho, g, lam_mid)) > target:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        return self._oc_update(rho, g, 0.5 * (lam_lo + lam_hi))

    # --- Public API ---------------------------------------------------------

    def run(
        self,
        initial: Field,
        oracle,
        objective,
        constraint=None,
        on_iteration: Optional[Callable] = None,
    ) -> OptimizeResult:
        """Run the OC optimizer and return the best design seen.

        Parameters
        ----------
        initial:
            Starting density field. Typically uniform at volume_fraction.
        oracle:
            Physics oracle. Must supply an analytic gradient; otherwise
            ValueError is raised immediately on the first evaluation.
        objective:
            Objective to maximize. Must produce a gradient.
        constraint:
            Accepted for API compatibility but ignored: volume is enforced
            internally by the OC bisection step.
        on_iteration:
            Optional callback called as on_iteration(iter, fom, delta, p)
            at the end of each iteration.

        Returns
        -------
        OptimizeResult with .field = best design seen, .fom = best fom,
        .history = per-iteration fom list, .iterations, .converged.

        Raises
        ------
        ValueError
            If the objective returns gradient=None (OC requires analytic
            sensitivity; it cannot fall back to finite differences).
        """
        rho: Field = initial.copy()
        history: List[float] = []
        best_fom: float = -np.inf
        best_field: Optional[Field] = None
        converged: bool = False
        prev_fom: Optional[float] = None

        for i in range(self.max_iter):
            p = self._current_p(i)
            _set_penalty(oracle, objective, p)

            filtered_vals = self._apply_filter(rho.values)
            design = rho.like(filtered_vals)

            ov = _evaluate(design, oracle, objective)

            if ov.gradient is None:
                raise ValueError(
                    "OCOptimizer requires an analytic gradient but the objective "
                    "returned gradient=None. Use TopologyOptimizer for problems "
                    "where no analytic gradient is available."
                )

            g = self._filter_vjp(ov.gradient)
            history.append(ov.fom)

            delta = abs(ov.fom - prev_fom) if prev_fom is not None else 0.0
            if on_iteration is not None:
                on_iteration(i + 1, ov.fom, delta, p)

            if ov.fom > best_fom:
                best_fom = ov.fom
                best_field = design.copy()

            if prev_fom is not None and delta < self.tol:
                converged = True
                break

            prev_fom = ov.fom
            rho = rho.like(self._bisect_lambda(rho.values, g))

        return OptimizeResult(
            field=best_field,
            fom=best_fom,
            history=history,
            iterations=len(history),
            used_finite_differences=False,
            converged=converged,
        )
