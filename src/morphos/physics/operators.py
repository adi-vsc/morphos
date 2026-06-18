"""Shared finite-difference operators on the regular Cartesian grid.

The implicit-geometry / grid-physics architecture (see project notes) leans on a
single voxel grid as the lingua franca, so the discrete operators that physics
backends need live here once rather than being re-derived per backend.

The central object is the interior Laplacian: the symmetric positive-definite
discretization of ``-laplacian`` on the interior unknowns of a grid held at zero
on the Dirichlet boundary. It is assembled by Kronecker products of the 1D
second-difference operator, which is fully vectorized (no Python element loop)
and therefore scales to large grids, and it is shared by both the steady
heat-conduction solve and the modal eigensolver.
"""

from __future__ import annotations

from scipy import sparse


def _second_difference_1d(m: int, h: float) -> sparse.csr_matrix:
    """1D Dirichlet second-difference operator on ``m`` interior points.

    Returns ``(1/h^2) * tridiag(-1, 2, -1)``, the SPD discretization of
    ``-d^2/dx^2`` with zero boundary values eliminated.
    """
    if m < 1:
        raise ValueError("need at least one interior point")
    inv_h2 = 1.0 / (h * h)
    main = 2.0 * inv_h2 * sparse.eye(m, format="csr")
    off = -inv_h2 * sparse.eye(m, k=1, format="csr")
    return (main + off + off.T).tocsr()


def interior_laplacian(shape, h: float) -> sparse.csr_matrix:
    """SPD discrete negative-Laplacian on the interior of an N-D grid.

    The operator is the Kronecker sum of the 1D second-difference along each
    axis, the dimension-general form of the 5-point (2D) / 7-point (3D) stencil:
    each axis contributes ``I (x) ... (x) L_axis (x) ... (x) I``. This is fully
    vectorized (no Python element loop) and works for any number of axes.

    Parameters
    ----------
    shape:
        Full grid shape including its one-cell Dirichlet border on every axis.
    h:
        Isotropic grid spacing.

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator of size ``prod(n_axis - 2)`` squared, acting on interior
        unknowns flattened row-major (C order, matching ``numpy.ravel``).
    """
    ms = [n - 2 for n in shape]
    if any(m < 1 for m in ms):
        raise ValueError("grid too small to have an interior")
    Ls = [_second_difference_1d(m, h) for m in ms]
    eyes = [sparse.eye(m, format="csr") for m in ms]
    A = None
    for axis in range(len(ms)):
        # row-major (C order) flatten: leading axes are the outer Kronecker
        # factors, so axis k carries L_k with identities on every other axis.
        factors = [Ls[axis] if i == axis else eyes[i] for i in range(len(ms))]
        term = factors[0]
        for f in factors[1:]:
            term = sparse.kron(term, f, format="csr")
        A = term if A is None else A + term
    return A.tocsr()
