"""PDE (Helmholtz) density filter for topology optimization.

The box/cone density filter (``scipy.ndimage.uniform_filter``) used elsewhere
in this package is directionally biased: its square support weights
axis-aligned neighbors the same as diagonal ones at a much larger physical
distance, so the effective filter radius is anisotropic. The PDE filter
(Lazarov & Sigmund 2016) instead solves the Helmholtz-type equation

    (-r^2 * laplacian + I) rho_tilde = rho

on the full design domain with homogeneous Neumann (no-flux) boundary
conditions. Its Green's function is isotropic (radially symmetric in the
continuum limit), so it removes the directional bias of the box filter while
keeping the same self-adjoint, linear-operator structure the optimizers rely
on for the VJP.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _neumann_laplacian_1d(m: int, h: float) -> sparse.csr_matrix:
    """1D Neumann (reflecting) negative-Laplacian on ``m`` points.

    Interior rows are the usual ``(1/h^2) * [-1, 2, -1]`` stencil. The two
    boundary rows use a one-sided difference consistent with a zero-flux
    (ghost cell mirrors the interior cell) boundary condition: row 0 is
    ``(1/h^2) * [1, -1]`` and row ``m-1`` is ``(1/h^2) * [-1, 1]``.
    """
    inv_h2 = 1.0 / (h * h)
    A = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="lil") * inv_h2
    A[0, 0] = inv_h2
    A[m - 1, m - 1] = inv_h2
    return A.tocsr()


def _neumann_laplacian(shape, h: float) -> sparse.csr_matrix:
    """SPD discrete negative-Laplacian with Neumann BCs on the full grid.

    Same Kronecker-sum construction as ``interior_laplacian`` in
    ``morphos.physics.operators``, but built over every grid point (not just
    the interior) since the density filter has no Dirichlet border to
    eliminate.
    """
    Ls = [_neumann_laplacian_1d(n, h) for n in shape]
    eyes = [sparse.eye(n, format="csr") for n in shape]
    A = None
    for axis in range(len(shape)):
        factors = [Ls[axis] if i == axis else eyes[i] for i in range(len(shape))]
        term = factors[0]
        for f in factors[1:]:
            term = sparse.kron(term, f, format="csr")
        A = term if A is None else A + term
    return A.tocsr()


class PDEFilter:
    """Helmholtz PDE density filter: solves ``(r^2 L + I) rho_tilde = rho``.

    ``L`` is the Neumann negative-Laplacian, so the operator ``K = r^2 L + I``
    is symmetric positive definite. Its sparse LU factorization is computed
    once at construction and reused for every ``apply``/``vjp`` call, since
    ``K`` does not change across optimization iterations (shape, spacing and
    radius are fixed).
    """

    def __init__(self, shape, h: float, radius: float) -> None:
        self.shape = tuple(shape)
        self.h = float(h)
        self.radius = float(radius)
        n = int(np.prod(self.shape))
        L = _neumann_laplacian(self.shape, self.h)
        K = (self.radius**2) * L + sparse.eye(n, format="csr")
        self._lu = splu(K.tocsc())

    def apply(self, rho: np.ndarray) -> np.ndarray:
        rhs = rho.reshape(-1)
        sol = self._lu.solve(rhs)
        return sol.reshape(self.shape)

    def vjp(self, v: np.ndarray) -> np.ndarray:
        # K is symmetric, so the adjoint solve is the same factorized solve.
        rhs = v.reshape(-1)
        sol = self._lu.solve(rhs)
        return sol.reshape(self.shape)
