"""Manufacturability constraints and projections.

A constraint can project a design Field onto the manufacturable set (for example
removing features smaller than a printer can resolve), provide the vector
Jacobian product of that projection so the optimizer can chain gradients through
it, and report manufacturability diagnostics for the final result.
"""

from __future__ import annotations

from abc import ABC

import numpy as np
from scipy import ndimage

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

    ``project`` and ``vjp`` are implemented for 2D fields only: the recursive
    layer-by-layer construction generalizes to 3D by replacing each 1D
    "row of columns" with a 2D in-plane layer and the footprint with a disc
    of radius ``w`` in that plane, but that needs a 2D smooth-max over a
    disc-shaped neighbourhood (not just a small fixed-width strip) and a
    matching reverse pass, which is more than a drop-in change. Deferred;
    2D (build direction = one grid axis) is the scope of this class.
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

    def project(self, field: Field) -> Field:
        rho = np.moveaxis(field.values, self.build_axis, 0)
        printed = self._project_2d(rho)
        out = np.moveaxis(printed, 0, self.build_axis)
        return field.like(out)

    def vjp(self, field: Field, grad: np.ndarray) -> np.ndarray:
        rho = np.moveaxis(field.values, self.build_axis, 0)
        g = np.moveaxis(grad, self.build_axis, 0)
        d_rho = self._vjp_2d(rho, g)
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
