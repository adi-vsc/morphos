"""A real PDE physics backend: steady-state heat conduction.

This backend exists to prove the architecture: a genuine partial differential
equation solver, with an exact adjoint gradient, slots in behind the same
:class:`~morphos.physics.oracle.PhysicsOracle` interface as the analytic toy,
and the same engine and optimizer drive it unchanged.

It solves the 2D steady heat equation on the grid with the design field acting
as a distributed heat source and the boundary held at zero temperature:

    -k laplacian(T) = s   inside,    T = 0   on the boundary.

The figure of merit is the negative weighted squared error of the temperature
against a target field. The gradient with respect to the source is computed by
the adjoint method (one extra linear solve), which is what makes inverse design
scale to many design variables.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult


class HeatConductionOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(self, target: np.ndarray, weight: float = 1.0) -> None:
        target = np.asarray(target, dtype=float)
        if target.ndim != 2:
            raise ValueError("HeatConductionOracle supports 2D grids only")
        self.target = target
        self.weight = float(weight)
        self.shape = target.shape
        self._boundary = self._boundary_mask(self.shape)
        self._A = None
        self._A_spacing = None

    @staticmethod
    def _boundary_mask(shape) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        m[0, :] = True
        m[-1, :] = True
        m[:, 0] = True
        m[:, -1] = True
        return m

    def _assemble(self, h: float) -> sparse.csr_matrix:
        ny, nx = self.shape
        n = ny * nx
        inv_h2 = 1.0 / (h * h)
        bnd = self._boundary
        A = sparse.lil_matrix((n, n))

        def idx(i, j):
            return i * nx + j

        for i in range(ny):
            for j in range(nx):
                k = idx(i, j)
                if bnd[i, j]:
                    A[k, k] = 1.0
                else:
                    A[k, k] = 4.0 * inv_h2
                    A[k, idx(i - 1, j)] = -inv_h2
                    A[k, idx(i + 1, j)] = -inv_h2
                    A[k, idx(i, j - 1)] = -inv_h2
                    A[k, idx(i, j + 1)] = -inv_h2
        return A.tocsr()

    def _matrix(self, field: Field) -> sparse.csr_matrix:
        spacing = field.spacing
        if abs(spacing[0] - spacing[1]) > 1e-12:
            raise ValueError("HeatConductionOracle assumes isotropic spacing")
        h = spacing[0]
        if self._A is None or self._A_spacing != h:
            self._A = self._assemble(h)
            self._A_spacing = h
        return self._A

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match target "
                f"shape {self.shape}"
            )
        A = self._matrix(field)
        bnd_flat = self._boundary.ravel()

        # Source, with the boundary forced to zero (Dirichlet).
        s = field.values.ravel().copy()
        s[bnd_flat] = 0.0

        T_flat = spsolve(A, s)
        T = T_flat.reshape(self.shape)

        diff = T - self.target
        value = -float(self.weight * np.sum(diff ** 2))

        # Adjoint: boundary temperature is fixed, so those objective terms are
        # constant in the source and must not flow into the gradient.
        dJdT = (-2.0 * self.weight * diff).ravel()
        dJdT[bnd_flat] = 0.0
        lam = spsolve(A.T.tocsr(), dJdT)
        grad = lam.copy()
        grad[bnd_flat] = 0.0
        gradient = grad.reshape(self.shape)

        return PhysicsResult(value=value, gradient=gradient, aux={"temperature": T})

    def residual(self, source: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Return A @ T - s_eff, which should be near zero for a valid solve."""
        if self._A is None:
            raise RuntimeError("call solve() before residual() to assemble the matrix")
        bnd_flat = self._boundary.ravel()
        s = np.asarray(source, dtype=float).ravel().copy()
        s[bnd_flat] = 0.0
        r = self._A @ temperature.ravel() - s
        return r.reshape(self.shape)
