"""Reference geometry kernel: pure numpy signed distance primitives.

This backend exists so the whole engine pipeline can run and be tested without
any heavy dependency. It is not meant to replace a production implicit kernel
(see :class:`morphos.geometry.picogk.PicoGKKernel`); it produces the same Field
representation, so swapping in the production kernel is an interface change, not
a rewrite.
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field
from morphos.geometry.kernel import GeometryKernel
from morphos.geometry.tpms import diamond_sdf, gyroid_sdf, schwartz_p_sdf


class VoxelKernel(GeometryKernel):
    """Builds signed distance fields for a small set of primitives."""

    def build(self, spec: dict) -> Field:
        primitive = spec.get("primitive")
        if primitive == "sphere":
            values = self._sphere(spec["center"], float(spec["radius"]))
        elif primitive == "box":
            values = self._box(spec["center"], spec["half_extent"])
        elif primitive == "slab":
            values = self._slab(int(spec["axis"]), float(spec["lo"]), float(spec["hi"]))
        elif primitive in ("gyroid", "schwartz_p", "diamond"):
            return self._tpms(primitive, float(spec["period"]), float(spec["thickness"]))
        else:
            raise ValueError(f"unknown primitive: {primitive!r}")
        return Field(values, self.spacing)

    def _tpms(self, primitive: str, period: float, thickness: float) -> Field:
        blank = Field(np.zeros(self.grid_shape), self.spacing)
        fn = {"gyroid": gyroid_sdf, "schwartz_p": schwartz_p_sdf, "diamond": diamond_sdf}[primitive]
        return fn(blank, period=period, thickness=thickness)

    def _sphere(self, center, radius: float) -> np.ndarray:
        grids = self.coordinate_grids()
        sq = sum((g - c) ** 2 for g, c in zip(grids, center))
        return np.sqrt(sq) - radius

    def _box(self, center, half_extent) -> np.ndarray:
        grids = self.coordinate_grids()
        q = [np.abs(g - c) - h for g, c, h in zip(grids, center, half_extent)]
        outside = np.sqrt(sum(np.maximum(qi, 0.0) ** 2 for qi in q))
        inside = np.minimum(np.maximum.reduce([qi for qi in q]), 0.0)
        return outside + inside

    def _slab(self, axis: int, lo: float, hi: float) -> np.ndarray:
        grids = self.coordinate_grids()
        x = grids[axis]
        return np.maximum(lo - x, x - hi)
