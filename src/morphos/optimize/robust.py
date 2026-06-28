"""Robust topology optimization via the delta-p (three-field) method.

LPBF (laser powder bed fusion) printing has a tolerance of roughly +/-delta in
feature size: a design that looks optimal at its nominal density may perform
much worse once its boundary is actually printed a bit smaller (erosion) or a
bit larger (dilation) than intended. The delta-p / three-field method (Wang,
Lazarov & Sigmund 2011; Lazarov, Wang & Sigmund 2016) makes the optimizer
account for this directly: at every iteration the same raw design is run
through three Heaviside projections at three different thresholds,

    rho_nom = H(rho; beta, eta=0.5)            nominal boundary
    rho_ero = H(rho; beta, eta=0.5 + delta)     eroded (solid shrinks)
    rho_dil = H(rho; beta, eta=0.5 - delta)     dilated (solid grows)

the oracle is solved on all three, and the optimizer ascends the *worst case*
of the three figures of merit, taking the gradient only from whichever field
is currently worst. This directly optimizes a min-max (robust) objective
instead of the nominal-only objective, trading a bit of nominal performance
for a design that degrades gracefully under manufacturing tolerance.

The volume fraction constraint is enforced on the nominal field (or the mean
of the three fields, configurable via ``volume_on``) using the same OC
bisection update as :class:`morphos.optimize.oc.OCOptimizer`.

This optimizer is implemented as a standalone OC-style loop (rather than by
wrapping another optimizer's internal iteration) because intercepting a base
optimizer's own oracle calls to substitute the three-field evaluation would
require fragile assumptions about each optimizer's private control flow
(callback signatures differ between OCOptimizer, MMAOptimizer and
TopologyOptimizer). Running its own loop keeps the three-field evaluation,
the gradient chain-rule through the worst-case projection, and the volume
bisection all in one place, mirroring OCOptimizer's structure exactly.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy import ndimage

from morphos.field import Field
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import _evaluate, _set_penalty


# --- Smooth Heaviside projection (Wang, Lazarov, Sigmund 2011) -------------


def _heaviside(rho: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Smooth Heaviside (tanh) projection of ``rho`` at threshold ``eta``.

    ``eta`` above 0.5 erodes the solid (shrinks it); ``eta`` below 0.5
    dilates it (grows it). ``eta == 0.5`` is the nominal projection.
    """
    num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return np.clip(num / den, 0.0, 1.0)


def _heaviside_deriv(rho: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Elementwise derivative d(H(rho; beta, eta))/d(rho)."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (rho - eta)) ** 2) / den


class RobustOptimizer(Optimizer):
    """Robust topology optimizer via the three-field (eroded/nominal/dilated)
    worst-case formulation.

    Runs an OC-style bisection loop internally (same multiplicative update
    and lambda-bisection as :class:`morphos.optimize.oc.OCOptimizer`), but at
    every iteration evaluates the oracle on three Heaviside projections of
    the same raw design -- nominal, eroded, and dilated -- and optimizes the
    worst (minimum) of the three figures of merit. The gradient used for the
    update is the gradient of whichever field is currently worst, chained
    back through that field's own Heaviside projection.

    Parameters
    ----------
    volume_fraction:
        Target volume fraction V* in (0, 1].
    max_iter:
        Maximum number of iterations.
    move:
        Per-iteration move limit on element density changes (OC update).
    eta:
        Damping exponent for the OC update (``B_e**eta``). Canonical 0.5.
    rho_min:
        Minimum density floor to keep the stiffness matrix nonsingular.
    delta:
        Threshold perturbation for the eroded/dilated fields: the eroded
        field projects at ``eta=0.5+delta``, the dilated field at
        ``eta=0.5-delta``. Typical LPBF tolerance: 0.05-0.15.
    volume_on:
        Which field the volume constraint is enforced against: ``"nominal"``
        (default) targets ``mean(rho_nom) == volume_fraction``; ``"mean"``
        targets the mean of the three projected fields' means.
    beta_start, beta_end:
        Heaviside sharpness continuation, ramping linearly over
        ``p_ramp_fraction`` of ``max_iter`` like the SIMP exponent.
    p_start, p_end, p_ramp_fraction:
        SIMP penalization continuation, identical semantics to OCOptimizer.
    tol:
        FOM-change tolerance for early convergence.
    filter_radius:
        Radius (voxels) of the box density filter applied to the raw design
        before projection. ``0`` (default) disables filtering.
    """

    def __init__(
        self,
        volume_fraction: float,
        max_iter: int = 100,
        move: float = 0.2,
        eta: float = 0.5,
        rho_min: float = 1e-3,
        delta: float = 0.1,
        volume_on: str = "nominal",
        beta_start: float = 1.0,
        beta_end: float = 8.0,
        p_start: float = 1.0,
        p_end: float = 3.0,
        p_ramp_fraction: float = 0.5,
        tol: float = 1e-4,
        filter_radius: float = 0.0,
    ) -> None:
        self.volume_fraction = float(volume_fraction)
        self.max_iter = int(max_iter)
        self.move = float(move)
        self.eta = float(eta)
        self.rho_min = float(rho_min)
        self.delta = float(delta)
        if volume_on not in ("nominal", "mean"):
            raise ValueError('volume_on must be "nominal" or "mean"')
        self.volume_on = str(volume_on)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.p_ramp_fraction = float(p_ramp_fraction)
        self.tol = float(tol)
        self.filter_radius = float(filter_radius)
        self.eta_nom = 0.5

    # --- SIMP p / Heaviside beta continuation schedule ----------------------

    def _ramp_fraction(self, iteration: int) -> float:
        ramp_iters = max(1, int(self.p_ramp_fraction * self.max_iter))
        return min(1.0, iteration / ramp_iters)

    def _current_p(self, iteration: int) -> float:
        t = self._ramp_fraction(iteration)
        return self.p_start + t * (self.p_end - self.p_start)

    def _current_beta(self, iteration: int) -> float:
        t = self._ramp_fraction(iteration)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    # --- Self-adjoint box density filter ------------------------------------

    def _filter_size(self) -> int:
        return 2 * int(round(self.filter_radius)) + 1

    def _apply_filter(self, values: np.ndarray) -> np.ndarray:
        if self.filter_radius <= 0:
            return values
        return ndimage.uniform_filter(
            values, size=self._filter_size(), mode="constant", cval=0.0
        )

    def _filter_vjp(self, grad: np.ndarray) -> np.ndarray:
        if self.filter_radius <= 0:
            return grad
        return ndimage.uniform_filter(
            grad, size=self._filter_size(), mode="constant", cval=0.0
        )

    # --- OC multiplicative update (same as OCOptimizer) ---------------------

    def _oc_update(self, rho: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
        N = rho.size
        B = np.clip(N * g / lam, 0.0, None)
        rho_new = rho * (B ** self.eta)
        lo = np.maximum(self.rho_min, rho - self.move)
        hi = np.minimum(1.0, rho + self.move)
        rho_new = np.clip(rho_new, lo, hi)
        return np.clip(rho_new, self.rho_min, 1.0)

    def _volume_measure(self, rho: np.ndarray, beta: float) -> Callable[[np.ndarray], float]:
        """Return a function mapping an OC-updated raw rho to the volume
        measure that the bisection should drive to ``volume_fraction``.

        The measure projects the *filtered* raw density, matching the
        forward pass (which evaluates the oracle on
        ``H(filter(rho), beta, eta)``). Omitting the filter here -- when
        ``filter_radius > 0`` -- makes the bisection enforce the volume of
        an unfiltered field while the oracle sees a filtered one, so the
        design systematically undershoots V* (the zero-padded box filter
        shrinks boundary density)."""
        if self.volume_on == "nominal":
            return lambda rho_new: float(
                np.mean(_heaviside(self._apply_filter(rho_new), beta, self.eta_nom))
            )
        # "mean": average of the three projected fields' means.
        def _mean_measure(rho_new: np.ndarray) -> float:
            f = self._apply_filter(rho_new)
            m_nom = np.mean(_heaviside(f, beta, self.eta_nom))
            m_ero = np.mean(_heaviside(f, beta, self.eta_nom + self.delta))
            m_dil = np.mean(_heaviside(f, beta, self.eta_nom - self.delta))
            return float((m_nom + m_ero + m_dil) / 3.0)

        return _mean_measure

    def _bisect_lambda(self, rho: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
        """Bisect lambda in [1e-9, 1e9] so that the configured volume measure
        of the OC-updated raw rho equals ``volume_fraction``."""
        lam_lo, lam_hi = 1e-9, 1e9
        target = self.volume_fraction
        measure = self._volume_measure(rho, beta)
        for _ in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if measure(self._oc_update(rho, g, lam_mid)) > target:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        return self._oc_update(rho, g, 0.5 * (lam_lo + lam_hi))

    # --- Public API -----------------------------------------------------

    def run(
        self,
        initial: Field,
        oracle,
        objective,
        constraint=None,
        on_iteration: Optional[Callable] = None,
    ) -> OptimizeResult:
        """Run the robust optimizer and return the best nominal design seen.

        Parameters
        ----------
        initial:
            Starting raw density field. Typically uniform at volume_fraction.
        oracle:
            Physics oracle. Must supply an analytic gradient; otherwise
            ValueError is raised on the first evaluation.
        objective:
            Objective to maximize. Must produce a gradient.
        constraint:
            Accepted for API compatibility but ignored: volume is enforced
            internally by the OC bisection step, exactly like OCOptimizer.
        on_iteration:
            Optional callback called as
            ``on_iteration(iter, fom_robust, delta, p, beta)`` at the end of
            each iteration, where ``fom_robust = min(fom_nom, fom_ero,
            fom_dil)``.

        Returns
        -------
        OptimizeResult with ``.field`` = best NOMINAL design seen (the
        design as it would actually print at the nominal threshold),
        ``.fom`` = the corresponding best robust (worst-case) fom,
        ``.history`` = per-iteration robust fom list, ``.iterations``,
        ``.converged``.

        Raises
        ------
        ValueError
            If the worst-case result's gradient is None (robust optimization
            requires analytic sensitivity; it cannot fall back to finite
            differences).
        """
        rho: Field = initial.copy()
        history: List[float] = []
        best_fom: float = -np.inf
        best_field: Optional[Field] = None
        converged: bool = False
        prev_fom: Optional[float] = None

        etas = (self.eta_nom, self.eta_nom + self.delta, self.eta_nom - self.delta)

        for i in range(self.max_iter):
            p = self._current_p(i)
            beta = self._current_beta(i)
            _set_penalty(oracle, objective, p)

            filtered_vals = self._apply_filter(rho.values)

            rho_nom = _heaviside(filtered_vals, beta, etas[0])
            rho_ero = _heaviside(filtered_vals, beta, etas[1])
            rho_dil = _heaviside(filtered_vals, beta, etas[2])

            ov_nom = _evaluate(rho.like(rho_nom), oracle, objective)
            ov_ero = _evaluate(rho.like(rho_ero), oracle, objective)
            ov_dil = _evaluate(rho.like(rho_dil), oracle, objective)

            foms = [ov_nom.fom, ov_ero.fom, ov_dil.fom]
            worst_idx = int(np.argmin(foms))
            worst_ov = (ov_nom, ov_ero, ov_dil)[worst_idx]
            worst_eta = etas[worst_idx]
            fom_robust = foms[worst_idx]

            if worst_ov.gradient is None:
                raise ValueError(
                    "RobustOptimizer requires an analytic gradient but the "
                    "objective returned gradient=None on the worst-case "
                    "field. Robust (min-max) optimization cannot fall back "
                    "to finite differences."
                )

            # Chain the worst-case gradient back through that field's own
            # Heaviside projection, then through the density filter.
            g = worst_ov.gradient * _heaviside_deriv(filtered_vals, beta, worst_eta)
            g = self._filter_vjp(g)

            history.append(fom_robust)

            delta_fom = abs(fom_robust - prev_fom) if prev_fom is not None else 0.0
            if on_iteration is not None:
                on_iteration(i + 1, fom_robust, delta_fom, p, beta)

            if fom_robust > best_fom:
                best_fom = fom_robust
                best_field = rho.like(rho_nom).copy()

            if prev_fom is not None and delta_fom < self.tol:
                converged = True
                break

            prev_fom = fom_robust
            rho = rho.like(self._bisect_lambda(rho.values, g, beta))

        return OptimizeResult(
            field=best_field,
            fom=best_fom,
            history=history,
            iterations=len(history),
            used_finite_differences=False,
            converged=converged,
        )
