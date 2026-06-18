"""A modal (eigenvalue) physics backend: vibration of a variable-mass membrane.

This backend solves the generalized symmetric eigenproblem

    K phi = lambda M(rho) phi

on the interior of the grid, where ``K`` is the SPD interior Laplacian (the
stiffness of a membrane clamped on the Dirichlet boundary) and ``M`` is a
diagonal mass matrix whose entries are the design density field ``rho``. The
eigenvalue ``lambda = omega^2`` is a squared natural frequency, so optimizing a
chosen ``lambda`` shapes the density toward target resonances or band gaps.

It exists to demonstrate that a *sparse eigensolver* (ARPACK ``eigsh`` or
``lobpcg``) slots in behind the same :class:`~morphos.physics.oracle.PhysicsOracle`
interface as the linear-solve backends, and that its eigenvalue sensitivity has an
exact adjoint.

Eigenvalue sensitivity. For a simple eigenvalue the Rayleigh quotient
``lambda = phi^T K phi / phi^T M phi`` is stationary at the eigenvector, so the
eigenvector-derivative term drops out and the sensitivity is the explicit term
only:

    d lambda / d rho_e = - lambda * phi_e^2 / (phi^T M phi)

(``K`` is independent of ``rho`` and ``dM/drho_e`` is the unit at node ``e``).
With an M-normalized eigenvector this is ``-lambda * phi_e^2``. This is gated by a
directional finite difference in the tests, not trusted on theory alone.

Limitation: this sensitivity is only valid for a *simple* eigenvalue. Repeated
(degenerate) eigenvalues -- common on symmetric domains, e.g. modes 1 and 2 of a
square membrane coincide -- are not differentiable in the ordinary sense and need
the subspace/multi-eigenvalue treatment, which this backend does not implement.
Optimize a simple mode (the fundamental is safe) or break the symmetry first.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg

from morphos.field import Field
from morphos.physics.operators import interior_laplacian
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_EIGENSOLVERS = ("eigsh", "lobpcg")


class ModalOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape,
        mode: int = 0,
        eigensolver: str = "eigsh",
        tol: float = 0.0,
    ) -> None:
        if len(shape) < 2:
            raise ValueError("ModalOracle needs a 2D or 3D grid")
        if mode < 0:
            raise ValueError("mode must be non-negative")
        if eigensolver not in _EIGENSOLVERS:
            raise ValueError(
                f"eigensolver must be one of {_EIGENSOLVERS}, got {eigensolver!r}"
            )
        self.shape = tuple(shape)
        self.mode = int(mode)
        self.eigensolver = eigensolver
        self.tol = float(tol)
        self._interior = ~self._boundary_mask(self.shape)
        self._K = None
        self._K_spacing = None

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

    def _stiffness(self, field: Field) -> sparse.csr_matrix:
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("ModalOracle assumes isotropic spacing")
        h = spacing[0]
        if self._K is None or self._K_spacing != h:
            self._K = interior_laplacian(self.shape, h)
            self._K_spacing = h
        return self._K

    def _smallest_eigenpair(self, K, M, mass_int):
        """Return (lambda, phi_int) for the mode-th smallest generalized pair."""
        n = K.shape[0]
        k = min(self.mode + 2, n - 1)
        if self.eigensolver == "eigsh":
            # Shift-invert at sigma=0 finds the eigenvalues nearest zero, i.e. the
            # smallest. K is SPD so the factorization at sigma=0 is well posed.
            vals, vecs = eigsh(K, k=k, M=M, sigma=0.0, which="LM", tol=self.tol)
        else:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((n, k))
            vals, vecs = lobpcg(
                K, X, B=M, largest=False, tol=max(self.tol, 1e-10), maxiter=2000
            )
        order = np.argsort(vals)
        idx = order[self.mode]
        lam = float(vals[idx])
        phi = vecs[:, idx]
        # M-normalize so phi^T M phi = 1 (makes the sensitivity -lam*phi^2).
        phi = phi / np.sqrt(phi @ (mass_int * phi))
        return lam, phi

    def solve(self, field: Field) -> PhysicsResult:
        if tuple(field.values.shape) != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape}"
            )
        K = self._stiffness(field)
        interior = self._interior.ravel()
        mass_int = field.values.ravel()[interior]
        if np.any(mass_int <= 0.0):
            raise ValueError("ModalOracle requires strictly positive density (mass)")
        M = sparse.diags(mass_int)

        lam, phi = self._smallest_eigenpair(K, M, mass_int)

        # Adjoint: simple-eigenvalue sensitivity, explicit term only.
        grad_int = -lam * phi**2
        grad_flat = np.zeros(field.values.size)
        grad_flat[interior] = grad_int
        gradient = grad_flat.reshape(self.shape)

        mode_field = np.zeros(field.values.size)
        mode_field[interior] = phi
        return PhysicsResult(
            value=lam,
            gradient=gradient,
            aux={"eigenvalue": lam, "mode_shape": mode_field.reshape(self.shape)},
        )
