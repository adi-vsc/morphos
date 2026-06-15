"""The Field: a scalar sampled on a regular Cartesian (voxel) grid.

Every layer of the engine speaks this one data type. A field can carry a signed
distance (geometry), a material density in [0, 1] (topology optimization), or any
physical quantity. The representation is shared by implicit geometry kernels and
by grid based physics solvers, so geometry passes to physics with no meshing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Field:
    """A scalar field on a uniform Cartesian grid.

    Parameters
    ----------
    values:
        An ``numpy.ndarray`` of sample values. Its number of dimensions defines
        the spatial dimension of the field (2 or 3 in practice).
    spacing:
        Physical distance between adjacent samples. A scalar applies to every
        axis; a sequence gives one spacing per axis and must match ``values.ndim``.
    """

    __slots__ = ("values", "_spacing")

    def __init__(self, values: np.ndarray, spacing) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        self.values = values
        self._spacing = self._normalize_spacing(spacing, values.ndim)

    @staticmethod
    def _normalize_spacing(spacing, ndim: int) -> tuple:
        if np.isscalar(spacing):
            spacing_tuple = tuple(float(spacing) for _ in range(ndim))
        else:
            spacing_tuple = tuple(float(s) for s in spacing)
            if len(spacing_tuple) != ndim:
                raise ValueError(
                    f"spacing has {len(spacing_tuple)} entries but field is "
                    f"{ndim}-dimensional"
                )
        if any(s <= 0.0 for s in spacing_tuple):
            raise ValueError("spacing must be positive on every axis")
        return spacing_tuple

    @property
    def shape(self) -> tuple:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def spacing(self) -> tuple:
        return self._spacing

    @property
    def voxel_volume(self) -> float:
        """Volume (or area in 2D) of a single grid cell."""
        v = 1.0
        for s in self._spacing:
            v *= s
        return v

    @property
    def physical_size(self) -> tuple:
        """Physical extent of the grid along each axis."""
        return tuple(n * s for n, s in zip(self.shape, self._spacing))

    def like(self, values: np.ndarray) -> "Field":
        """Return a new field on this grid carrying ``values``."""
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if values.shape != self.shape:
            raise ValueError(
                f"values shape {values.shape} does not match grid {self.shape}"
            )
        return Field(values, self._spacing)

    def copy(self) -> "Field":
        return Field(self.values.copy(), self._spacing)

    def __repr__(self) -> str:
        return f"Field(shape={self.shape}, spacing={self._spacing})"
