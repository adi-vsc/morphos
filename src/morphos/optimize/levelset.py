"""Level-set topology optimizer (Wang, Mei & Wang 2003 / Allaire 2002).

The design variable is a level-set function ``phi`` defined on the same
Cartesian grid as the density field:

    phi < 0  -> solid material
    phi > 0  -> void
    phi = 0  -> interface (material boundary)

For SIMP physics evaluation the level set is converted to a density field by
a smooth Heaviside of ``-phi``:

    rho = H(-phi; eta) = 1 / (1 + exp(2*eta*phi))

so ``rho`` is close to 1 deep inside the solid (``phi << 0``) and close to 0
deep inside the void (``phi >> 0``), with a smooth transition of width
``O(1/eta)`` straddling the zero level set.

Each iteration:
  1. rho = H(-phi; eta)  (smooth density from the level set)
  2. Set the SIMP penalty (p-continuation, same schedule as TopologyOptimizer)
  3. Evaluate FOM and g = d(FOM)/d(rho) through the physics oracle
  4. Chain g back through the Heaviside to get an interface velocity in
     phi-space, then extend that velocity into the whole domain by smoothing
     it with a uniform (box) filter -- the cheap, robust stand-in for a true
     PDE velocity extension.
  5. Advect phi by one step of the Hamilton-Jacobi equation
     ``dphi/dt + V*|grad phi| = 0`` (sign chosen so the design ascends FOM).
  6. Re-enforce the volume fraction by bisecting a uniform additive shift on
     phi so that ``mean(H(-(phi+c))) = V*``.
  7. Periodically reinitialize phi to (an approximation of) a signed distance
     function via two Euclidean distance transforms, which keeps
     ``|grad phi|`` close to 1 and the advection step numerically well posed.

The optimizer tracks and returns the best RHO design (not phi) seen over the
run, matching the convention used by ``OCOptimizer``/``MMAOptimizer``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy import ndimage

from morphos.field import Field
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import _evaluate, _set_penalty


class LevelSetOptimizer(Optimizer):
    """Hamilton-Jacobi level-set topology optimizer.

    Parameters
    ----------
    volume_fraction:
        Target volume fraction V* in (0, 1].
    max_iter:
        Maximum number of iterations.
    dt:
        Advection step size (a fraction of the grid cell size; the
        Hamilton-Jacobi update uses unit grid spacing internally, so this is
        effectively the CFL number).
    eta:
        Smoothing parameter of the Heaviside used to turn ``phi`` into a
        density: larger values give a sharper (more 0/1) density.
    reinit_every:
        Reinitialize ``phi`` to (an approximate) signed distance function
        every this many iterations.
    p_start, p_end, p_ramp_fraction:
        SIMP penalization continuation, identical semantics to
        ``TopologyOptimizer``/``OCOptimizer``.
    tol:
        FOM-change tolerance for early convergence.
    velocity_smooth_sigma:
        Sigma (in voxels) of the Gaussian-like smoothing used to extend the
        interface velocity into the full domain.
    """

    def __init__(
        self,
        volume_fraction: float,
        max_iter: int = 200,
        dt: float = 0.1,
        eta: float = 10.0,
        reinit_every: int = 5,
        p_start: float = 1.0,
        p_end: float = 3.0,
        p_ramp_fraction: float = 0.4,
        tol: float = 1e-4,
        velocity_smooth_sigma: float = 2.0,
    ) -> None:
        if not (0.0 < float(volume_fraction) <= 1.0):
            raise ValueError("volume_fraction must be in (0, 1]")
        self.volume_fraction = float(volume_fraction)
        self.max_iter = int(max_iter)
        self.dt = float(dt)
        self.eta = float(eta)
        self.reinit_every = int(reinit_every)
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.p_ramp_fraction = float(p_ramp_fraction)
        self.tol = float(tol)
        self.velocity_smooth_sigma = float(velocity_smooth_sigma)

    # --- SIMP p continuation schedule (mirrors TopologyOptimizer/OC) -------

    def _current_p(self, iteration: int) -> float:
        ramp_iters = max(1, int(self.p_ramp_fraction * self.max_iter))
        t = min(1.0, iteration / ramp_iters)
        return self.p_start + t * (self.p_end - self.p_start)

    # --- phi <-> rho conversion ----------------------------------------------

    def _density_from_phi(self, phi: np.ndarray) -> np.ndarray:
        """Smooth Heaviside of ``-phi``: rho ~= 1 where phi << 0 (solid)."""
        return 1.0 / (1.0 + np.exp(2.0 * self.eta * phi))

    def _density_vjp(self, phi: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        """Chain a gradient in rho-space back to a gradient in phi-space.

        ``dH/dphi = -2*eta*rho*(1-rho)``, an elementwise (diagonal Jacobian)
        scale, so the VJP is just that derivative multiplied elementwise into
        the incoming gradient.
        """
        rho = self._density_from_phi(phi)
        d = -2.0 * self.eta * rho * (1.0 - rho)
        return grad_rho * d

    # --- finite-difference gradient / advection helpers ---------------------

    @staticmethod
    def _grad_phi(phi: np.ndarray):
        """Central finite differences of ``phi`` along every axis, clipped
        to [-3, 3] for numerical stability of the advection step."""
        grads = np.gradient(phi)
        if phi.ndim == 1:
            grads = [grads]
        return [np.clip(g, -3.0, 3.0) for g in grads]

    def _advect(self, phi: np.ndarray, velocity_extended: np.ndarray, dt: float) -> np.ndarray:
        grads = self._grad_phi(phi)
        grad_mag = np.sqrt(np.sum(np.stack([g ** 2 for g in grads]), axis=0) + 1e-8)
        return phi - dt * velocity_extended * grad_mag

    def _extend_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Extend the (narrow-band) interface velocity into the whole domain
        by smoothing it with a uniform box filter, the cheap stand-in for a
        true PDE-based velocity extension off the zero level set.

        The smoothed velocity is then normalized by its maximum absolute
        value so the Hamilton-Jacobi advection step always satisfies a unit
        CFL number for the configured ``dt``, regardless of the raw
        sensitivity magnitude (which can vary by orders of magnitude across
        SIMP p-continuation and across problems). This mirrors the standard
        level-set practice (Allaire 2002, Osher & Fedkiw 2003) of normalizing
        the extension velocity before advecting.
        """
        size = max(1, int(round(self.velocity_smooth_sigma)))
        smoothed = ndimage.uniform_filter(velocity, size=2 * size + 1, mode="nearest")
        scale = np.max(np.abs(smoothed))
        if scale > 1e-300:
            smoothed = smoothed / scale
        return smoothed

    # --- reinitialization (approximate signed distance) ---------------------

    @staticmethod
    def _reinit(phi: np.ndarray) -> np.ndarray:
        """Reinitialize phi to an approximate signed distance function.

        Positive outside the solid (phi >= 0), negative inside (phi < 0),
        via the exterior-minus-interior Euclidean distance transform trick.

        When ``phi`` has no sign change at all (no solid/void interface
        anywhere in the domain -- e.g. immediately after initializing from a
        uniform density field), ``distance_transform_edt`` on an all-True or
        all-False mask measures distance to the (nonexistent) nearest
        opposite-sign pixel by treating the array border as background,
        which produces a huge, meaningless gradient instead of a signed
        distance. Guard against that degenerate case by leaving ``phi``
        unchanged: there is no interface to reinitialize around yet, so the
        identity is the correct (and only sane) "signed distance".
        """
        solid = phi < 0
        if not solid.any() or solid.all():
            return phi

        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(phi >= 0) - distance_transform_edt(phi < 0)

    # --- volume correction (bisection on an additive shift) ------------------

    def _volume_bisect(self, phi: np.ndarray, target_vf: float) -> np.ndarray:
        """Bisect a uniform additive shift ``c`` on phi so that
        ``mean(H(-(phi + c))) == target_vf``.

        ``H(-(phi+c))`` is monotone decreasing in ``c`` (a larger shift moves
        the whole level set toward void), guaranteeing a unique root.
        """
        c_lo, c_hi = -5.0, 5.0
        for _ in range(40):
            c_mid = 0.5 * (c_lo + c_hi)
            vf_mid = float(np.mean(self._density_from_phi(phi + c_mid)))
            if vf_mid > target_vf:
                c_lo = c_mid
            else:
                c_hi = c_mid
        c = 0.5 * (c_lo + c_hi)
        return phi + c

    # --- phi initialization from an initial density Field --------------------

    def _phi_from_initial(self, initial: Field) -> np.ndarray:
        """Convert an initial (density) field to a level-set function.

        ``phi = 0.5 - rho`` is negative where rho > 0.5 (solid) and positive
        where rho < 0.5 (void). Attempt to normalize the result to an
        (approximate) signed distance via :meth:`_reinit`; for a uniform
        initial density (the common case -- a uniform field at V*) there is
        no sign change anywhere yet, so :meth:`_reinit` is a no-op (see its
        docstring) and the level set develops its own interface organically
        from the first iteration's advection.
        """
        return self._reinit(0.5 - initial.values)

    # --- public API -----------------------------------------------------------

    def run(
        self,
        initial: Field,
        oracle,
        objective,
        constraint=None,
        on_iteration: Optional[Callable] = None,
    ) -> OptimizeResult:
        """Run the level-set optimizer and return the best RHO design seen.

        Parameters
        ----------
        initial:
            Starting density field (typically uniform at volume_fraction).
            Converted internally to a level-set function.
        oracle:
            Physics oracle. Must supply an analytic gradient
            (``d(FOM)/d(rho)``); ``constraint`` is accepted for API
            compatibility but ignored -- volume is enforced internally by
            bisection on the level-set shift.
        on_iteration:
            Optional callback invoked as ``on_iteration(iter, fom, delta, p)``
            at the end of each iteration.

        Returns
        -------
        OptimizeResult with ``.field`` = best RHO design seen (a Field on the
        same grid as ``initial``), ``.fom`` = best fom, ``.history``, etc.

        Raises
        ------
        ValueError
            If the objective returns ``gradient=None`` (the level-set update
            requires an analytic shape sensitivity).
        """
        phi = self._phi_from_initial(initial)
        history: List[float] = []
        best_fom: float = -np.inf
        best_field: Optional[Field] = None
        converged: bool = False
        prev_fom: Optional[float] = None

        for i in range(self.max_iter):
            p = self._current_p(i)
            _set_penalty(oracle, objective, p)

            rho = self._density_from_phi(phi)
            design = initial.like(rho)

            ov = _evaluate(design, oracle, objective)

            if ov.gradient is None:
                raise ValueError(
                    "LevelSetOptimizer requires an analytic gradient but the "
                    "objective returned gradient=None."
                )
            if not np.isfinite(ov.fom):
                raise RuntimeError(
                    f"non-finite figure of merit ({ov.fom}) at iteration "
                    f"{i + 1}; the physics solve diverged"
                )

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

            # Shape sensitivity in phi-space (ascent direction for FOM).
            grad_phi = self._density_vjp(phi, ov.gradient)
            velocity = self._extend_velocity(grad_phi)

            # Ascend FOM: phi_new = phi + dt * velocity * |grad phi|, i.e. the
            # Hamilton-Jacobi advection phi - dt*(-velocity)*|grad phi| with
            # V_n = -velocity.
            phi = self._advect(phi, -velocity, self.dt)
            phi = self._volume_bisect(phi, self.volume_fraction)

            if (i + 1) % self.reinit_every == 0:
                phi = self._reinit(phi)
                phi = self._volume_bisect(phi, self.volume_fraction)

        final_p = self._current_p(max(0, len(history) - 1))
        _set_penalty(oracle, objective, final_p)
        final_rho = self._density_from_phi(phi)
        final_design = initial.like(final_rho)
        final_ov = _evaluate(final_design, oracle, objective)
        if np.isfinite(final_ov.fom) and final_ov.fom > best_fom:
            best_fom = final_ov.fom
            best_field = final_design.copy()

        return OptimizeResult(
            field=best_field,
            fom=best_fom,
            history=history,
            iterations=len(history),
            used_finite_differences=False,
            converged=converged,
        )
