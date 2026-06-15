"""GeometryKernel: turns design parameters into a geometry Field.

Convention: geometry is represented as a signed distance field. Values are
negative inside the solid, positive outside, and near zero on the surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from morphos.field import Field


class GeometryKernel(ABC):
    """Abstract base for anything that produces geometry as a Field.

    A kernel is bound to a grid (shape and spacing). Concrete kernels implement
    :meth:`build`, mapping a parameter spec to a signed distance ``Field`` on
    that grid.
    """

    def __init__(self, grid_shape, spacing) -> None:
        self.grid_shape = tuple(int(n) for n in grid_shape)
        ndim = len(self.grid_shape)
        if np.isscalar(spacing):
            self.spacing = tuple(float(spacing) for _ in range(ndim))
        else:
            self.spacing = tuple(float(s) for s in spacing)
            if len(self.spacing) != ndim:
                raise ValueError("spacing length must match grid_shape")

    def coordinate_grids(self):
        """Return one coordinate array per axis, broadcast over the grid.

        Coordinates run from 0 with the kernel spacing, using matrix ('ij')
        indexing so axis order matches the field array.
        """
        axes = [np.arange(n) * s for n, s in zip(self.grid_shape, self.spacing)]
        return np.meshgrid(*axes, indexing="ij")

    @abstractmethod
    def build(self, spec: dict) -> Field:
        """Build a signed distance Field from a parameter spec."""
        raise NotImplementedError
