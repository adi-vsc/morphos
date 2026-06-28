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

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from morphos.field import Field
from morphos.objective.objective import ObjectiveValue
from morphos.optimize.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
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
        filter_type: str = "box",
        filter_h: float = 1.0,
        beta_start: float = 1.0,
        beta_end: float = 1.0,
        patience: int = 1,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 10,
        resume_from: Optional[Path] = None,
        line_search: bool = False,
        max_backtracks: int = 20,
        volume_fraction: Optional[float] = None,
        min_length_scale: float = 0.0,
        min_length_eta: float = 0.75,
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
            Radius (in voxels for ``"box"``, physical length for ``"pde"``)
            of the density filter applied to the raw design before
            projection and before the oracle solve. ``0`` (the default)
            disables filtering entirely, preserving the pre-filter behavior
            exactly.
        filter_type:
            ``"box"`` (default) uses the directionally-biased
            ``scipy.ndimage.uniform_filter`` cone/box filter, preserving
            existing behavior exactly. ``"pde"`` uses the isotropic Helmholtz
            PDE density filter (Lazarov & Sigmund 2016, see
            ``morphos.optimize.pde_filter.PDEFilter``) instead.
        filter_h:
            Physical grid spacing used to assemble the PDE filter's Laplacian
            when ``filter_type == "pde"``. Ignored for ``"box"``.
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
        checkpoint_dir:
            When set, write a compressed checkpoint (the current raw design
            field, iteration index, current ``p``, current ``beta``, and the
            FOM history so far) to ``checkpoint_dir/checkpoint.npz`` every
            ``checkpoint_every`` iterations. ``None`` (default) disables
            checkpointing entirely, leaving ``run`` behaviour unchanged.
        checkpoint_every:
            Iteration interval between checkpoint writes.
        resume_from:
            When set, load this checkpoint before the run starts and resume
            from its saved raw design field, iteration index, ``p`` and
            ``beta`` (continuing the SAME continuation schedule, keyed off
            the restored absolute iteration, rather than restarting the ramp
            from ``p_start``/``beta_start``) instead of starting from the
            ``initial`` field passed to :meth:`run`.
        line_search:
            When True, each ascent step is backtracked (the step is halved up
            to ``max_backtracks`` times) until the candidate design's figure of
            merit does not decrease, guaranteeing a monotone non-decreasing FOM
            history at the cost of extra solves per iteration. Default False
            preserves the original fixed-step behavior exactly. A non-finite
            (NaN / Inf) figure of merit or gradient is always rejected, by
            backtracking when ``line_search`` is on and by raising a clear
            ``RuntimeError`` when it is off, so a diverged solve can never be
            returned silently as a result.
        max_backtracks:
            Maximum step halvings per iteration when ``line_search`` is on.
        min_length_scale:
            Minimum feature size (in voxels) enforced via the Guest et al.
            (2004) double-filter scheme: a second filter-project pass is
            inserted after the first Heaviside projection, eroding away
            solid features thinner than this scale before the complementary
            (void) projection restores the rest. ``0`` (the default)
            disables the double filter entirely, preserving the original
            single filter -> Heaviside -> constraint chain exactly.
        min_length_eta:
            Threshold ``eta`` used for the first (erosion) projection in the
            double-filter scheme. The second (dilation) projection uses the
            complementary threshold ``1 - min_length_eta``. Default ``0.75``
            enforces a minimum SOLID length scale. Ignored when
            ``min_length_scale <= 0``.
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
        self.filter_type = str(filter_type)
        self.filter_h = float(filter_h)
        self._pde_filter = None
        self._pde_filter_shape = None
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.patience = int(patience)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = int(checkpoint_every)
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.line_search = bool(line_search)
        self.max_backtracks = int(max_backtracks)
        if volume_fraction is not None and not (0.0 < float(volume_fraction) <= 1.0):
            raise ValueError("volume_fraction must be in (0, 1]")
        self.volume_fraction = None if volume_fraction is None else float(volume_fraction)
        self.min_length_scale = float(min_length_scale)
        self.min_length_eta = float(min_length_eta)

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

    def _get_pde_filter(self, shape):
        if self._pde_filter is None or self._pde_filter_shape != shape:
            from morphos.optimize.pde_filter import PDEFilter

            self._pde_filter = PDEFilter(shape, self.filter_h, self.filter_radius)
            self._pde_filter_shape = shape
        return self._pde_filter

    def _apply_filter(self, values: np.ndarray) -> np.ndarray:
        """Density filter applied to the raw design, the standard SIMP
        regularizer that removes isolated single-voxel spikes and
        checkerboarding. ``filter_radius <= 0`` disables filtering and
        returns ``values`` unchanged. ``filter_type == "pde"`` uses the
        isotropic Helmholtz PDE filter instead of the directionally-biased
        box filter (the default ``"box"``)."""
        if self.filter_radius <= 0:
            return values
        if self.filter_type == "pde":
            return self._get_pde_filter(values.shape).apply(values)
        return ndimage.uniform_filter(
            values, size=self._filter_size(), mode="constant", cval=0.0
        )

    def _filter_vjp(self, grad: np.ndarray) -> np.ndarray:
        """Vector-Jacobian product of :meth:`_apply_filter`.

        Both the box filter (a linear, symmetric-kernel, zero-padded
        operator) and the PDE filter (a symmetric positive-definite operator
        solve) are self-adjoint, so in either case the VJP is the same
        filtering operation applied to the incoming gradient (identical to
        the reasoning used by ``MinFeatureSize`` in
        ``morphos.manufacturing.constraints``).
        """
        if self.filter_radius <= 0:
            return grad
        if self.filter_type == "pde":
            return self._get_pde_filter(grad.shape).vjp(grad)
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

        When ``min_length_scale > 0`` a second filter-project pass (Guest et
        al. 2004) is inserted after the first Heaviside projection (still at
        the default ``eta=0.5``): the second filter re-smooths the first
        projection's output, and a second projection at the complementary
        threshold ``1 - min_length_eta`` (default ``min_length_eta=0.75``,
        so ``eta=0.25``) erodes thin solid features that the first stage
        left behind. ``filtered`` (the FIRST filter's output, before any
        projection) is still what is returned/reused so that
        :meth:`_chain_vjp` can recompute the same intermediate values.
        """
        filtered = self._apply_filter(x.values)
        projected_values = self._heaviside_project(filtered, beta)

        if self.min_length_scale > 0:
            # Second filter with same radius
            filtered2 = self._apply_filter(projected_values)
            # Second projection: complementary threshold enforces void minimum length
            projected_values = self._heaviside_project(
                filtered2, beta, eta=1.0 - self.min_length_eta
            )

        projected = x.like(projected_values)
        design = self._project(projected, constraint)
        return design, filtered

    def _chain_vjp(
        self, x: Field, filtered: np.ndarray, beta: float, grad: np.ndarray, constraint
    ) -> np.ndarray:
        """Chain a gradient (in design-Field space) back through the
        constraint projection, then the Heaviside projection(s) and filter(s),
        to gradient-in-raw-x space, in that reverse order.

        When ``min_length_scale > 0`` this recomputes the same double-filter
        forward intermediates :meth:`_design_chain` produced (the first
        projection's output, the second filter's output) and backprops
        through the second projection (complementary eta), the second
        filter, then the first projection (at the default ``eta=0.5``)
        before falling through to the shared first-filter VJP.
        """
        g = grad
        if constraint is not None:
            g = constraint.vjp(x, g)

        if self.min_length_scale > 0:
            # Backprop through second projection (eta_void = 1 - min_length_eta)
            filtered2 = self._apply_filter(self._heaviside_project(filtered, beta))
            g = self._heaviside_vjp(filtered2, beta, g, eta=1.0 - self.min_length_eta)
            # Backprop through second filter
            g = self._filter_vjp(g)
            # Backprop through first projection
            g = self._heaviside_vjp(filtered, beta, g, eta=0.5)
        else:
            g = self._heaviside_vjp(filtered, beta, g)

        g = self._filter_vjp(g)
        return g

    def _fom(self, x: Field, oracle, objective, constraint, beta: float = 1.0) -> float:
        design, _ = self._design_chain(x, beta, constraint)
        return _evaluate(design, oracle, objective).fom

    def _fd_gradient(
        self, x: Field, oracle, objective, constraint, beta: float = 1.0
    ) -> np.ndarray:
        n = x.values.size
        if n > 1000:
            import warnings
            warnings.warn(
                f"Finite-difference gradient on {n} elements requires {n} forward "
                f"solves per iteration (O(N²) total). Implement an adjoint "
                f"(provides_gradient=True on the oracle) to scale beyond small grids.",
                RuntimeWarning,
                stacklevel=3,
            )
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

    def _clip_bounds(self, values: np.ndarray) -> np.ndarray:
        if self.bounds is not None:
            return np.clip(values, self.bounds[0], self.bounds[1])
        return values

    def _enforce_volume(self, x: Field) -> None:
        """Project x onto {rho : mean(rho) = volume_fraction, lo <= rho <= hi}.

        Bisects a uniform shift c so that mean(clip(x + c, lo, hi)) == V*.
        This is the projected-gradient step for the equality volume constraint,
        identical in structure to the OC bisection but additive rather than
        multiplicative, making it applicable to any gradient-ascent step.
        """
        if self.volume_fraction is None:
            return
        lo, hi = self.bounds if self.bounds is not None else (0.0, 1.0)
        target = self.volume_fraction
        vals = x.values
        c_lo, c_hi = float(lo - vals.max()), float(hi - vals.min())
        for _ in range(60):
            c = 0.5 * (c_lo + c_hi)
            if np.mean(np.clip(vals + c, lo, hi)) > target:
                c_hi = c
            else:
                c_lo = c
        x.values = np.clip(vals + 0.5 * (c_lo + c_hi), lo, hi)

    def _take_step(
        self, x: Field, g: np.ndarray, base_fom: float, oracle, objective, constraint, beta: float
    ) -> None:
        """Advance the raw design ``x`` along the gradient ``g`` in place.

        Without ``line_search`` this is the original fixed-step ascent. With it,
        the step is halved (up to ``max_backtracks`` times) until the candidate
        design's FOM is finite and does not fall below ``base_fom``; if no such
        step is found the design is left unchanged, so the run stalls (and the
        early-stop test fires) rather than stepping downhill into garbage.
        """
        if not self.line_search:
            x.values = self._clip_bounds(x.values + self.step_size * g)
            return

        step = self.step_size
        for _ in range(self.max_backtracks):
            candidate = self._clip_bounds(x.values + step * g)
            cand_fom = self._fom(x.like(candidate), oracle, objective, constraint, beta)
            if np.isfinite(cand_fom) and cand_fom >= base_fom - 1e-12:
                x.values = candidate
                return
            step *= 0.5
        # No improving step found: leave the design where it is.

    def run(
        self, initial: Field, oracle, objective, constraint=None, on_iteration=None
    ) -> OptimizeResult:
        start_iter = 0
        if self.resume_from is not None:
            # Warm start: resume the raw design and the absolute iteration
            # count from the checkpoint, so the p/beta continuation schedule
            # (a function of that absolute iteration) picks up exactly where
            # it left off instead of restarting at p_start/beta_start.
            x, start_iter, _, _, history = load_checkpoint(self.resume_from)
        else:
            x = initial.copy()
            history = []
        used_fd = False
        converged = False
        prev_fom = history[-1] if history else None
        iterations = start_iter
        best_fom = -np.inf
        best_field = None
        stall_count = 0

        for i in range(start_iter, self.max_iter):
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

            if on_iteration is not None:
                delta = abs(ov.fom - prev_fom) if prev_fom is not None else 0.0
                on_iteration(iterations, ov.fom, delta, p, beta)

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

            if not np.all(np.isfinite(g)):
                raise RuntimeError(
                    f"non-finite gradient at iteration {iterations}; the "
                    f"physics solve diverged"
                )

            self._take_step(x, g, ov.fom, oracle, objective, constraint, beta)
            self._enforce_volume(x)

            if (
                self.checkpoint_dir is not None
                and iterations % self.checkpoint_every == 0
            ):
                save_checkpoint(
                    checkpoint_path(self.checkpoint_dir), x, iterations, p, beta, history
                )

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
