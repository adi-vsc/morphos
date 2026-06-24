"""Manufacturability constraints and projections.

A constraint can project a design Field onto the manufacturable set (for example
removing features smaller than a printer can resolve), provide the vector
Jacobian product of that projection so the optimizer can chain gradients through
it, and report manufacturability diagnostics for the final result.
"""

from __future__ import annotations

from abc import ABC

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from morphos.field import Field


class ManufacturabilityConstraint(ABC):
    """Base class. Defaults make a constraint a no-op identity projection."""

    def project(self, field: Field) -> Field:
        return field

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        """Vector Jacobian product of :meth:`project` at ``field``.

        For an identity projection this is the gradient unchanged.
        """
        return grad

    def report(self, field: Field) -> dict:
        return {}


class MinFeatureSize(ManufacturabilityConstraint):
    """A linear density filter that removes features below a radius.

    Implemented as a normalized box filter with zero padding, which is a
    self-adjoint linear operator, so its vector Jacobian product is the same
    filter applied to the incoming gradient.
    """

    def __init__(self, radius: int) -> None:
        self.radius = int(radius)
        self.size = 2 * self.radius + 1

    def _filter(self, values: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(
            values, size=self.size, mode="constant", cval=0.0
        )

    def project(self, field: Field) -> Field:
        return field.like(self._filter(field.values))

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        return self._filter(grad)


class Overhang(ManufacturabilityConstraint):
    """Langelaar (2016/2017) self-supporting density filter for AM/SLM/DMLS.

    Models the physical fact that a powder-bed fusion process cannot print a
    voxel unless the layer immediately below it provides support within the
    printer's maximum self-supporting overhang angle: steeper local slopes
    need supports, shallower ones print clean. The filter walks the grid one
    build layer at a time along ``build_axis`` and clamps each layer's
    printed density to what the (already printed) layer below can hold up,
    so unsupported islands and overhangs are smoothly suppressed.

    Per layer ``i`` (layer 0 is the baseplate and is always fully supported):

        support[i, j]  = smooth_max_{k in footprint(j)} printed[i-1, k]
        printed[i, j]  = smooth_min(rho[i, j], support[i, j])

    ``footprint(j)`` is the set of columns within the overhang cone, i.e.
    offset ``|k - j| <= w`` with ``w = round(tan(angle_deg))`` cells (so the
    classic 45 degree limit gives a 3-cell-wide footprint of the cell
    directly below plus its two diagonal neighbours). The smooth max/min are
    a Kreisselmeier-Steinhauser (KS) aggregation sharpened by ``p``, which
    recovers the hard max/min as ``p -> infinity`` while staying
    differentiable for gradient-based topology optimization.

    Both 2D and 3D fields are supported. The build direction is one grid axis
    (``build_axis``); the remaining axes form the in-plane layer. In 2D a layer
    is a 1D row of columns and the support footprint is a strip of half-width
    ``w``; in 3D a layer is a 2D plane and the footprint is the
    ``(2w+1) x (2w+1)`` square below each voxel (the discretised build cone).
    The recursive smooth-max support / smooth-min printing construction and its
    exact reverse pass are otherwise identical across dimensions.
    """

    def __init__(
        self,
        angle_deg: float = 45.0,
        build_axis: int = 0,
        p: float = 30.0,
        eps: float = 1e-6,
    ) -> None:
        self.angle_deg = float(angle_deg)
        self.build_axis = int(build_axis)
        self.p = float(p)
        self.eps = float(eps)
        self.footprint = max(1, int(round(np.tan(np.radians(self.angle_deg)))))

    def _ks_max(self, x: np.ndarray, axis: int) -> tuple:
        """Smooth, softmax-weighted max of ``x`` along ``axis``.

        Returns ``(value, weights)`` where ``value = sum(x * softmax(p*x))``
        and ``weights = softmax(p * x)`` (summing to 1 along ``axis``), so
        the same weights serve directly as the backward-pass Jacobian of
        ``value`` with respect to ``x`` to leading order.

        A plain log-sum-exp (Kreisselmeier-Steinhauser) aggregator is exact
        for a single dominant entry but overshoots a tied/uniform vector by
        ``log(N)/p``; normalizing that log term by the count ``N`` fixes the
        uniform case but then undershoots a single spike among zeros. The
        softmax-weighted average used here is exact at both extremes (one
        dominant entry, or all entries equal) for any ``N``, which matters
        because both cases occur every layer: a flat, fully supported
        region (uniform) and an isolated supporting voxel (single spike).
        """
        p = self.p
        m = np.max(x, axis=axis, keepdims=True)
        shifted = p * (x - m)
        exp_shifted = np.exp(shifted)
        denom = np.sum(exp_shifted, axis=axis, keepdims=True)
        weights = exp_shifted / denom
        value = np.sum(x * weights, axis=axis, keepdims=True)
        return value, weights

    @staticmethod
    def _ks_max_jacobian(x: np.ndarray, value: np.ndarray, weights: np.ndarray, p: float) -> np.ndarray:
        """Exact d(value)/d(x_k) for ``value = sum(x * softmax(p*x))``.

        ``value`` is itself a function of every entry (through the softmax
        weights), so the Jacobian has a correction term beyond the weights
        themselves: ``J_k = w_k * (1 + p * (x_k - value))``. ``sum_k J_k``
        is 1 to leading order, which is what makes the aggregate a
        consistent (if approximate) max operator.
        """
        return weights * (1.0 + p * (x - value))

    def _layer_footprint_stack(self, prev_layer: np.ndarray) -> np.ndarray:
        """Stack shifted copies of ``prev_layer`` (1D, length n) into shape
        ``(2*w+1, n)``, zero padded at the boundaries (void outside the
        plate counts as no support, never as free support).
        """
        w = self.footprint
        n = prev_layer.shape[0]
        padded = np.pad(prev_layer, (w, w), mode="constant", constant_values=0.0)
        stack = np.stack(
            [padded[k : k + n] for k in range(2 * w + 1)], axis=0
        )
        return stack

    def _project_2d(self, rho: np.ndarray) -> np.ndarray:
        n_layers = rho.shape[0]
        printed = np.empty_like(rho)
        printed[0] = rho[0]
        for i in range(1, n_layers):
            stack = self._layer_footprint_stack(printed[i - 1])
            support, _ = self._ks_max(stack, axis=0)
            support = support[0]
            # smooth_min(a, b) = -smooth_max(-a, -b)
            pair = np.stack([-rho[i], -support], axis=0)
            neg_min, _ = self._ks_max(pair, axis=0)
            printed[i] = -neg_min[0]
        return printed

    def _vjp_2d(self, rho: np.ndarray, grad: np.ndarray) -> np.ndarray:
        n_layers = rho.shape[0]
        w = self.footprint
        n = rho.shape[1]
        p = self.p

        # Forward pass, caching what the backward pass needs per layer.
        printed = np.empty_like(rho)
        printed[0] = rho[0]
        min_jac_cache = [None] * n_layers
        max_jac_cache = [None] * n_layers
        for i in range(1, n_layers):
            stack = self._layer_footprint_stack(printed[i - 1])
            support, max_weights = self._ks_max(stack, axis=0)
            max_jac_cache[i] = self._ks_max_jacobian(stack, support, max_weights, p)
            support = support[0]
            pair = np.stack([-rho[i], -support], axis=0)
            neg_min, min_weights = self._ks_max(pair, axis=0)
            min_jac_cache[i] = self._ks_max_jacobian(pair, neg_min, min_weights, p)
            printed[i] = -neg_min[0]

        # Backward pass: accumulate d(loss)/d(rho) and d(loss)/d(printed[i-1]).
        d_rho = np.zeros_like(rho)
        d_printed_prev = np.zeros(n)  # gradient flowing into printed[i-1]
        for i in range(n_layers - 1, 0, -1):
            g = grad[i] + d_printed_prev
            min_jac = min_jac_cache[i]  # shape (2, n): jac of neg_min wrt [-rho[i], -support]
            # printed[i] = -neg_min(-rho[i], -support)
            # d(printed[i])/d(rho[i])  =  d(-neg_min)/d(-rho[i]) * d(-rho[i])/d(rho[i]) * (-1)... simplifies to min_jac[0]
            d_rho[i] += g * min_jac[0]
            d_support = g * min_jac[1]

            max_jac = max_jac_cache[i]  # shape (2w+1, n)
            # support = ks_max(stack of shifted printed[i-1])
            d_prev = np.zeros(n + 2 * w)
            for k in range(2 * w + 1):
                d_prev[k : k + n] += d_support * max_jac[k]
            d_printed_prev = d_prev[w : w + n]

        d_rho[0] += grad[0] + d_printed_prev
        return d_rho

    def _footprint_stack_2d(self, prev_layer: np.ndarray) -> np.ndarray:
        """Stack the ``(2w+1)^2`` in-plane-shifted copies of a 2D layer
        ``(ny, nx)`` into shape ``((2w+1)^2, ny, nx)``, zero padded (void
        outside the plate is never free support)."""
        w = self.footprint
        ny, nx = prev_layer.shape
        padded = np.pad(prev_layer, ((w, w), (w, w)), mode="constant", constant_values=0.0)
        copies = [
            padded[a : a + ny, b : b + nx]
            for a in range(2 * w + 1)
            for b in range(2 * w + 1)
        ]
        return np.stack(copies, axis=0)

    def _project_3d(self, rho: np.ndarray) -> np.ndarray:
        """3D Langelaar filter: layer axis is axis 0, each layer a 2D plane."""
        n_layers = rho.shape[0]
        printed = np.empty_like(rho)
        printed[0] = rho[0]
        for i in range(1, n_layers):
            stack = self._footprint_stack_2d(printed[i - 1])
            support, _ = self._ks_max(stack, axis=0)
            support = support[0]
            pair = np.stack([-rho[i], -support], axis=0)
            neg_min, _ = self._ks_max(pair, axis=0)
            printed[i] = -neg_min[0]
        return printed

    def _vjp_3d(self, rho: np.ndarray, grad: np.ndarray) -> np.ndarray:
        n_layers = rho.shape[0]
        w = self.footprint
        ny, nx = rho.shape[1], rho.shape[2]
        p = self.p

        printed = np.empty_like(rho)
        printed[0] = rho[0]
        min_jac_cache = [None] * n_layers
        max_jac_cache = [None] * n_layers
        for i in range(1, n_layers):
            stack = self._footprint_stack_2d(printed[i - 1])
            support, max_weights = self._ks_max(stack, axis=0)
            max_jac_cache[i] = self._ks_max_jacobian(stack, support, max_weights, p)
            support = support[0]
            pair = np.stack([-rho[i], -support], axis=0)
            neg_min, min_weights = self._ks_max(pair, axis=0)
            min_jac_cache[i] = self._ks_max_jacobian(pair, neg_min, min_weights, p)
            printed[i] = -neg_min[0]

        d_rho = np.zeros_like(rho)
        d_printed_prev = np.zeros((ny, nx))
        for i in range(n_layers - 1, 0, -1):
            g = grad[i] + d_printed_prev
            min_jac = min_jac_cache[i]
            d_rho[i] += g * min_jac[0]
            d_support = g * min_jac[1]

            max_jac = max_jac_cache[i]  # ((2w+1)^2, ny, nx)
            d_prev = np.zeros((ny + 2 * w, nx + 2 * w))
            c = 0
            for a in range(2 * w + 1):
                for b in range(2 * w + 1):
                    d_prev[a : a + ny, b : b + nx] += d_support * max_jac[c]
                    c += 1
            d_printed_prev = d_prev[w : w + ny, w : w + nx]

        d_rho[0] += grad[0] + d_printed_prev
        return d_rho

    def project(self, field: Field) -> Field:
        rho = np.moveaxis(field.values, self.build_axis, 0)
        printed = self._project_2d(rho) if rho.ndim == 2 else self._project_3d(rho)
        out = np.moveaxis(printed, 0, self.build_axis)
        return field.like(out)

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        rho = np.moveaxis(field.values, self.build_axis, 0)
        g = np.moveaxis(grad, self.build_axis, 0)
        d_rho = self._vjp_2d(rho, g) if rho.ndim == 2 else self._vjp_3d(rho, g)
        return np.moveaxis(d_rho, 0, self.build_axis)

    def report(self, field: Field) -> dict:
        printed = np.clip(self.project(field).values, 0.0, None)
        rho = np.clip(field.values, 0.0, None)
        deficit = np.clip(rho - printed, 0.0, None)
        total = rho.sum()
        unsupported_fraction = (
            float(deficit.sum() / total) if total > 0.0 else 0.0
        )
        return {
            "unsupported_volume_fraction": unsupported_fraction,
            "max_density_deficit": float(deficit.max()) if deficit.size else 0.0,
        }


class MinWallThickness(ManufacturabilityConstraint):
    """Enforce a minimum solid wall thickness by a differentiable morphological
    *opening* (erosion followed by dilation): solid features thinner than the
    structuring element are removed, while thicker walls and the void phase are
    left essentially unchanged.

    Each morphological step is a smooth log-sum-exp pool over a box neighbourhood
    of side ``2*min_thickness_voxels + 1``:

        erode(rho)_i  = -1/p * log( mean_{j in N(i)} exp(-p * rho_j) )   (soft-min)
        open(rho)      =  1/p * log( mean_{j in N(i)} exp( p * erode_j) ) (soft-max)

    As ``p -> infinity`` this recovers the exact min-pool / max-pool morphology;
    finite ``p`` keeps it differentiable for gradient-based topology
    optimization. The box average uses zero padding (``uniform_filter`` in
    ``constant`` mode), which is a *self-adjoint* linear operator, so the vector
    Jacobian product is the exact reverse chain through the two pools. FD-gated.

    (The audit's single-pool ``(mean rho^p)^(1/p)`` with large positive ``p`` is a
    soft-max / dilation, which would *grow* solid rather than open it; opening
    needs the soft-min erosion first, as implemented here.)
    """

    def __init__(self, min_thickness_voxels: int = 1, p_norm: float = 20.0) -> None:
        self.radius = int(min_thickness_voxels)
        self.size = 2 * self.radius + 1
        self.p = float(p_norm)

    def _boxmean(self, x: np.ndarray) -> np.ndarray:
        # Zero-padded box average: a self-adjoint linear operator (symmetric
        # kernel, constant padding), so it serves as its own VJP.
        return ndimage.uniform_filter(x, size=self.size, mode="constant", cval=0.0)

    def _open(self, rho: np.ndarray):
        p = self.p
        A = np.exp(-p * rho)
        B = self._boxmean(A)
        e = -np.log(B) / p          # soft erosion
        C = np.exp(p * e)
        D = self._boxmean(C)
        out = np.log(D) / p         # soft dilation of the erosion = opening
        return out, (A, B, C, D)

    def project(self, field: Field) -> Field:
        out, _ = self._open(np.clip(field.values, 0.0, 1.0))
        return field.like(out)

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        p = self.p
        rho = np.clip(field.values, 0.0, 1.0)
        _, (A, B, C, D) = self._open(rho)
        # Reverse chain (every _boxmean is self-adjoint):
        dD = grad / (p * D)
        dC = self._boxmean(dD)
        de = dC * p * C
        dB = -de / (p * B)
        dA = self._boxmean(dB)
        d_rho = dA * (-p) * A
        return d_rho

    def report(self, field: Field) -> dict:
        opened = self._open(np.clip(field.values, 0.0, 1.0))[0]
        rho = np.clip(field.values, 0.0, 1.0)
        removed = np.clip(rho - opened, 0.0, None)
        total = float(rho.sum())
        return {
            "thin_wall_volume_fraction": float(removed.sum() / total) if total > 0 else 0.0,
            "max_wall_removal": float(removed.max()) if removed.size else 0.0,
        }


class PowderRemoval(ManufacturabilityConstraint):
    """Penalise enclosed voids that cannot drain loose powder after printing.

    A connected-component test of the void phase is not differentiable, so this
    uses a diffusion proxy. Solve, on the grid,

        (-laplacian + kappa * rho) phi = 0,    phi = 1 on the drain boundary,

    where the solid phase (high ``rho``) acts as a distributed absorber. Void
    that is connected to a drain stays near ``phi = 1``; void sealed off by solid
    decays to ``phi ~ 0``. The (differentiable, scalar) penalty is

        P = sum_i (1 - rho_i) * (1 - phi_i),

    large only for enclosed (un-drainable) void. ``project`` is the identity (this
    is a penalty/diagnostic, not a geometry filter); ``value`` and ``vjp`` give the
    penalty and its exact adjoint sensitivity ``dP/drho``. FD-gated.
    """

    _EDGES = ("left", "right", "top", "bottom", "front", "back")

    def __init__(self, drain_edges=("bottom",), penalty_weight: float = 1.0,
                 kappa: float = 10.0) -> None:
        self.drain_edges = tuple(drain_edges)
        self.penalty_weight = float(penalty_weight)
        self.kappa = float(kappa)

    def _drain_mask(self, shape) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        ndim = len(shape)
        # axis, lo/hi -> edge name (2D: y axis is rows=top/bottom, x axis cols=left/right)
        names = {
            (0, 0): "top", (0, 1): "bottom",
            (1, 0): "left", (1, 1): "right",
            (2, 0): "front", (2, 1): "back",
        }
        for axis in range(ndim):
            for side, hi in ((0, False), (1, True)):
                if names.get((axis, side)) in self.drain_edges:
                    sl = [slice(None)] * ndim
                    sl[axis] = -1 if hi else 0
                    m[tuple(sl)] = True
        return m

    def _solve_phi(self, rho: np.ndarray, h: float):
        from morphos.physics.operators import interior_laplacian

        shape = rho.shape
        drain = self._drain_mask(shape)
        # Negative Laplacian (SPD) on the full grid via the shared kron-sum
        # assembly is interior-only; build a full-grid operator with Dirichlet
        # rows on the drain nodes instead. Simple 5/7-point assembly here.
        n = rho.size
        L = _full_laplacian(shape, h)
        A = (L + sparse.diags(self.kappa * rho.ravel())).tolil()
        b = np.zeros(n)
        drain_flat = np.where(drain.ravel())[0]
        for idx in drain_flat:
            A.rows[idx] = [idx]
            A.data[idx] = [1.0]
            b[idx] = 1.0
        A = A.tocsr()
        phi = spsolve(A, b)
        return phi.reshape(shape), A, drain.ravel()

    def value(self, field: Field) -> float:
        rho = np.clip(field.values, 0.0, 1.0)
        phi, _, _ = self._solve_phi(rho, field.spacing[0])
        return self.penalty_weight * float(np.sum((1.0 - rho) * (1.0 - phi)))

    def vjp(self, field: Field, grad: np.ndarray = None) -> np.ndarray:
        # Adjoint of P(rho) = w * sum (1-rho)(1-phi(rho)), A(rho) phi = b.
        rho = np.clip(field.values, 0.0, 1.0)
        w = self.penalty_weight
        phi, A, drain = self._solve_phi(rho, field.spacing[0])
        phi_flat = phi.ravel()
        rho_flat = rho.ravel()

        # Explicit dP/drho: -w (1 - phi).
        dP_drho = -w * (1.0 - phi_flat)
        # dP/dphi: -w (1 - rho); zero on drain rows (phi fixed there).
        dP_dphi = -w * (1.0 - rho_flat)
        dP_dphi[drain] = 0.0
        # Adjoint solve A^T lam = dP/dphi.
        lam = spsolve(A.T.tocsc(), dP_dphi)
        # dA/drho_e affects only the diagonal absorber: d(A phi)/drho_e = kappa*phi_e.
        # Total: dP/drho_e -= lam_e * kappa * phi_e (drain rows excluded: their
        # diagonal was overwritten to 1 and no longer depends on rho).
        coupling = self.kappa * lam * phi_flat
        coupling[drain] = 0.0
        dP_drho = dP_drho - coupling
        return dP_drho.reshape(rho.shape)

    def report(self, field: Field) -> dict:
        rho = np.clip(field.values, 0.0, 1.0)
        phi, _, _ = self._solve_phi(rho, field.spacing[0])
        enclosed = (1.0 - rho) * (1.0 - phi)
        return {
            "enclosed_void_penalty": self.penalty_weight * float(enclosed.sum()),
            "max_enclosed_void": float(enclosed.max()) if enclosed.size else 0.0,
        }


def _full_laplacian(shape, h: float) -> "sparse.csr_matrix":
    """Full-grid negative Laplacian (Neumann interior, 5/7-point) via a
    Kronecker sum of 1D second-difference operators with reflective ends."""
    mats = []
    ndim = len(shape)
    for axis, n in enumerate(shape):
        main = 2.0 * np.ones(n)
        main[0] = 1.0
        main[-1] = 1.0  # Neumann ends so the operator stays well-defined
        off = -1.0 * np.ones(n - 1)
        L1 = sparse.diags([off, main, off], [-1, 0, 1]) / (h * h)
        eyes = [sparse.identity(m) for m in shape]
        eyes[axis] = L1
        K = eyes[0]
        for e in eyes[1:]:
            K = sparse.kron(K, e)
        mats.append(K)
    return sum(mats).tocsr()


class Connectivity(ManufacturabilityConstraint):
    """Reports whether the solid region is a single connected component.

    This is a diagnostic constraint: its projection and vjp are identity. The
    solid region is taken as values at or above ``threshold``.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def report(self, field: Field) -> dict:
        mask = field.values >= self.threshold
        _, num = ndimage.label(mask)
        return {"num_components": int(num), "connected": bool(num == 1)}
