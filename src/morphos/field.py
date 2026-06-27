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
        if values.ndim not in (2, 3):
            raise TypeError(
                f"values must be a 2-D or 3-D array; got ndim={values.ndim}"
            )
        _sh = values.shape
        _ratio = max(_sh) / min(_sh)
        if _ratio > 20:
            raise ValueError(
                f"Extreme aspect ratio {_ratio:.1f}:1 (shape {_sh}). "
                "Use a coarser voxel size or a larger domain on the short axis."
            )
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

    @classmethod
    def validate(cls, values: np.ndarray, spacing) -> list:
        """Run all Field invariant checks and return a list of violation strings.

        Returns an empty list when valid. Intended for the feasibility layer to
        call before committing to a run; does NOT raise, only collects violations.

        Checks performed (superset of what ``__init__`` enforces):

        * ``values`` must be 2-D or 3-D.
        * Aspect ratio must not exceed 20:1.
        * Every axis must have at least 4 voxels (this check is here only, not in
          ``__init__``, so that existing tests that use 3×N grids are not broken).
        * Spacing entries must be positive and match ``values.ndim``.
        """
        violations: list = []

        if not isinstance(values, np.ndarray):
            violations.append(
                f"values must be a numpy.ndarray; got {type(values).__name__}"
            )
            return violations  # remaining checks assume ndarray

        if values.ndim not in (2, 3):
            violations.append(
                f"values must be 2-D or 3-D; got ndim={values.ndim}"
            )
        else:
            sh = values.shape
            ratio = max(sh) / min(sh)
            if ratio > 20:
                violations.append(
                    f"Extreme aspect ratio {ratio:.1f}:1 (shape {sh}). "
                    "Use a coarser voxel size or a larger domain on the short axis."
                )
            if any(s < 4 for s in sh):
                violations.append(
                    f"Every axis must have at least 4 voxels; got shape {sh}. "
                    "Refine the grid or choose a larger domain."
                )

        # Spacing checks
        try:
            ndim = values.ndim if isinstance(values, np.ndarray) else 0
            if np.isscalar(spacing):
                spacing_seq = [float(spacing)] * ndim
            else:
                spacing_seq = [float(s) for s in spacing]
                if len(spacing_seq) != ndim:
                    violations.append(
                        f"spacing has {len(spacing_seq)} entries but field is "
                        f"{ndim}-dimensional"
                    )
                    spacing_seq = []  # skip value check
            if any(s <= 0.0 for s in spacing_seq):
                violations.append("spacing must be positive on every axis")
        except (TypeError, ValueError) as exc:
            violations.append(f"spacing is invalid: {exc}")

        return violations

    def __repr__(self) -> str:
        return f"Field(shape={self.shape}, spacing={self._spacing})"
