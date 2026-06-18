"""A real PDE physics backend: steady-state heat conduction.

This backend exists to prove the architecture: a genuine partial differential
equation solver, with an exact adjoint gradient, slots in behind the same
:class:`~morphos.physics.oracle.PhysicsOracle` interface as the analytic toy,
and the same engine and optimizer drive it unchanged.

It solves the steady heat equation (2D or 3D) on the grid with the design field
acting as a distributed heat source and the boundary held at zero temperature:

    -k laplacian(T) = s   inside,    T = 0   on the boundary.

The figure of merit is the negative weighted squared error of the temperature
against a target field. The gradient with respect to the source is computed by
the adjoint method (one extra linear solve), which is what makes inverse design
scale to many design variables.

The discrete operator is the shared symmetric positive-definite interior
Laplacian from :mod:`morphos.physics.operators`. Because it is SPD, the linear
solve can run either by a direct sparse factorization (``solver="direct"``,
exact, best for small/moderate grids) or by a Jacobi-preconditioned conjugate
gradient (``solver="cg"``, matrix-friendly, scales to grids where direct LU
fill-in is prohibitive). Both produce the same answer to tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

from morphos.field import Field
from morphos.physics.operators import interior_laplacian
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_SOLVERS = ("direct", "cg")
_PRECONDITIONERS = ("jacobi", "amg")


class HeatConductionOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        target: np.ndarray,
        weight: float = 1.0,
        solver: str = "direct",
        preconditioner: str = "jacobi",
        cg_rtol: float = 1e-10,
        cg_maxiter: int | None = None,
    ) -> None:
        target = np.asarray(target, dtype=float)
        if target.ndim < 2:
            raise ValueError("HeatConductionOracle needs a 2D or 3D grid")
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
        if preconditioner not in _PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {_PRECONDITIONERS}, got {preconditioner!r}"
            )
        self.target = target
        self.weight = float(weight)
        self.solver = solver
        self.preconditioner = preconditioner
        self.cg_rtol = float(cg_rtol)
        self.cg_maxiter = cg_maxiter
        self.shape = target.shape
        self._interior = ~self._boundary_mask(self.shape)
        self._A = None
        self._A_spacing = None
        self._precond = None

    @staticmethod
    def _boundary_mask(shape) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        for axis in range(len(shape)):
            lo = [slice(None)] * len(shape)
            hi = [slice(None)] * len(shape)
            lo[axis] = 0
            hi[axis] = -1
            m[tuple(lo)] = True
            m[tuple(hi)] = True
        return m

    def _matrix(self, field: Field) -> sparse.csr_matrix:
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("HeatConductionOracle assumes isotropic spacing")
        h = spacing[0]
        if self._A is None or self._A_spacing != h:
            self._A = interior_laplacian(self.shape, h)
            self._A_spacing = h
            self._precond = self._build_preconditioner(self._A)
        return self._A

    def _build_preconditioner(self, A: sparse.csr_matrix):
        if self.preconditioner == "jacobi":
            # Diagonal scaling: cheap, but iteration count grows with grid size.
            return sparse.diags(1.0 / A.diagonal())
        # Algebraic multigrid: near grid-independent iteration count, which is
        # what makes CG actually beat direct LU at scale. Imported lazily so the
        # dependency is only required when this path is used.
        import pyamg

        return pyamg.smoothed_aggregation_solver(A.tocsr()).aspreconditioner()

    def _solve_linear(self, A: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
        if self.solver == "direct":
            return spsolve(A, b)
        # A is symmetric positive-definite, so CG is valid and self-adjoint.
        x, info = cg(A, b, rtol=self.cg_rtol, atol=0.0, maxiter=self.cg_maxiter, M=self._precond)
        if info != 0:
            raise RuntimeError(f"CG failed to converge (info={info})")
        return x

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match target "
                f"shape {self.shape}"
            )
        A = self._matrix(field)
        interior = self._interior.ravel()

        # Source on the interior unknowns; the boundary is Dirichlet zero.
        s_int = field.values.ravel()[interior]

        T_int = self._solve_linear(A, s_int)
        T_flat = np.zeros(field.values.size)
        T_flat[interior] = T_int
        T = T_flat.reshape(self.shape)

        diff = T - self.target
        value = -float(self.weight * np.sum(diff ** 2))

        # Adjoint: A is self-adjoint, so the co-state solves the same operator.
        # Boundary temperature is fixed, so its objective terms stay out of the
        # gradient (they live only on boundary DOFs, which we never solve for).
        dJdT_int = (-2.0 * self.weight * diff).ravel()[interior]
        lam_int = self._solve_linear(A, dJdT_int)
        grad_flat = np.zeros(field.values.size)
        grad_flat[interior] = lam_int
        gradient = grad_flat.reshape(self.shape)

        return PhysicsResult(value=value, gradient=gradient, aux={"temperature": T})

    def residual(self, source: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Return the residual of the linear system, near zero for a valid solve.

        Interior DOFs report ``A @ T_int - s_int``; boundary DOFs report the
        deviation of the temperature from the Dirichlet zero condition.
        """
        if self._A is None:
            raise RuntimeError("call solve() before residual() to assemble the matrix")
        interior = self._interior.ravel()
        s_int = np.asarray(source, dtype=float).ravel()[interior]
        T = np.asarray(temperature, dtype=float).ravel()
        r = np.zeros(T.size)
        r[interior] = self._A @ T[interior] - s_int
        r[~interior] = T[~interior]
        return r.reshape(self.shape)
