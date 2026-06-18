"""Shared finite-difference / finite-element operators on the regular grid.

The implicit-geometry / grid-physics architecture (see project notes) leans on a
single voxel grid as the lingua franca, so the discrete operators that physics
backends need live here once rather than being re-derived per backend.

Two families of operator live here:

- The interior Laplacian: the symmetric positive-definite discretization of
  ``-laplacian`` on the interior unknowns of a grid held at zero on the
  Dirichlet boundary. Assembled by Kronecker products of the 1D
  second-difference operator (fully vectorized, no Python element loop), and
  shared by the steady heat-conduction solve and the modal eigensolver.
- ``q4_plane_stress_stiffness``: the bilinear-quad (Q4) plane-stress element
  stiffness matrix used by the elasticity backend, the finite-element analogue
  of the finite-difference Laplacian above (one constant local operator,
  scaled per element and assembled into a global sparse matrix by the caller).
"""

from __future__ import annotations

import numpy as np
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


def q4_plane_stress_stiffness(
    young_modulus: float, poisson_ratio: float, h: float
) -> np.ndarray:
    """Stiffness matrix of one bilinear-quad (Q4) plane-stress element.

    A square element of side ``h``, unit out-of-plane thickness, isotropic
    linear-elastic material (``young_modulus``, ``poisson_ratio``). Local node
    order is counterclockwise starting at the bottom-left corner -- ``(0,0)``,
    ``(h,0)``, ``(h,h)``, ``(0,h)`` -- and the 8x8 matrix's degrees of freedom
    are interleaved ``[u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y]``.

    Assembled by 2x2 Gauss quadrature (the standard exact integration for a
    bilinear quad's stiffness, since the strain-displacement product is at
    most bilinear in each natural coordinate), the textbook formula behind
    SIMP topology optimization codes such as Andreassen et al. 2011's 88-line
    implementation. This implementation derives the matrix from quadrature
    rather than transcribing a closed-form 8x8 array, and is checked in
    ``tests/test_elasticity.py`` for symmetry and the expected null space (the
    three planar rigid-body modes: two translations, one rotation).
    """
    E, nu = float(young_modulus), float(poisson_ratio)
    # Plane-stress constitutive matrix relating stress to engineering strain
    # (exx, eyy, gamma_xy).
    C = (E / (1.0 - nu**2)) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    # Natural coordinates of the four nodes, same order as the docstring.
    node_xi = np.array([-1.0, 1.0, 1.0, -1.0])
    node_eta = np.array([-1.0, -1.0, 1.0, 1.0])
    # Jacobian of the map from natural coords [-1,1]^2 to physical [0,h]^2 is
    # constant (h/2 on the diagonal) for this square element.
    j = h / 2.0
    det_j = j * j
    inv_j = 1.0 / j

    K = np.zeros((8, 8))
    for xi, eta in gauss_points:
        dN_dxi = 0.25 * node_xi * (1.0 + node_eta * eta)
        dN_deta = 0.25 * node_eta * (1.0 + node_xi * xi)
        dN_dx = inv_j * dN_dxi
        dN_dy = inv_j * dN_deta
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        # Gauss weights are 1 for the 2-point rule on each axis.
        K += (B.T @ C @ B) * det_j
    return K


def hex8_stiffness(young_modulus: float, poisson_ratio: float, h: float) -> np.ndarray:
    """Stiffness matrix of one trilinear hexahedral (Hex8) solid element.

    The 3D analogue of :func:`q4_plane_stress_stiffness`: a cube element of
    side ``h``, isotropic linear-elastic material, used by the 3D extension of
    the elasticity SIMP backend (the standard element behind 3D topology
    optimization codes such as Liu & Tovar's top3D). Local node order is the
    natural-coordinate corners ``(-1,-1,-1)`` through ``(-1,1,1)`` -- bottom
    face counterclockwise then top face counterclockwise, matching the Q4
    convention extruded along z -- and the 24x24 matrix's degrees of freedom
    are interleaved ``[u0x, u0y, u0z, u1x, u1y, u1z, ...]``.

    Assembled by 2x2x2 Gauss quadrature, exact for the trilinear shape
    functions' strain-displacement product. Checked in ``tests/test_operators.py``
    against an independent re-derivation of the same quadrature, for symmetry,
    and for the expected null space (the six spatial rigid-body modes: three
    translations, three rotations).
    """
    E, nu = float(young_modulus), float(poisson_ratio)
    # Isotropic 3D constitutive matrix relating stress to engineering strain
    # (exx, eyy, ezz, gamma_xy, gamma_yz, gamma_zx).
    lam_c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = lam_c * np.array(
        [
            [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
            [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
            [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ]
    )
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(a, b, c) for a in (-gp, gp) for b in (-gp, gp) for c in (-gp, gp)]
    # Natural coordinates of the eight nodes, same order as the docstring.
    node_xi = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    node_eta = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    node_zeta = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    # Jacobian of the map from natural coords [-1,1]^3 to physical [0,h]^3 is
    # constant (h/2 on the diagonal) for this cubic element.
    j = h / 2.0
    det_j = j * j * j
    inv_j = 1.0 / j

    K = np.zeros((24, 24))
    for xi, eta, zeta in gauss_points:
        dN_dxi = 0.125 * node_xi * (1.0 + node_eta * eta) * (1.0 + node_zeta * zeta)
        dN_deta = 0.125 * node_eta * (1.0 + node_xi * xi) * (1.0 + node_zeta * zeta)
        dN_dzeta = 0.125 * node_zeta * (1.0 + node_xi * xi) * (1.0 + node_eta * eta)
        dN_dx = inv_j * dN_dxi
        dN_dy = inv_j * dN_deta
        dN_dz = inv_j * dN_dzeta
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3 * i] = dN_dx[i]
            B[1, 3 * i + 1] = dN_dy[i]
            B[2, 3 * i + 2] = dN_dz[i]
            B[3, 3 * i] = dN_dy[i]
            B[3, 3 * i + 1] = dN_dx[i]
            B[4, 3 * i + 1] = dN_dz[i]
            B[4, 3 * i + 2] = dN_dy[i]
            B[5, 3 * i] = dN_dz[i]
            B[5, 3 * i + 2] = dN_dx[i]
        # Gauss weights are 1 for the 2-point rule on each axis.
        K += (B.T @ C @ B) * det_j
    return K
