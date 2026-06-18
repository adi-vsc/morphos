"""Tests for the shared finite-difference operators.

The interior Laplacian is the symmetric positive-definite core that both the
heat-conduction solve and the modal eigensolver build on. Its discrete spectrum
is known in closed form, which gives a machine-precision verification target.
"""

import numpy as np
import pytest
from scipy import sparse

from morphos.physics.operators import interior_laplacian, q4_plane_stress_stiffness


def analytic_dirichlet_spectrum(shape, h):
    """Exact eigenvalues of the discrete negative Dirichlet Laplacian.

    For an interior grid of (ny-2) by (nx-2) unknowns the 5-point operator has
    eigenvalues lambda_{p,q} = (2/h^2)(1-cos(p*pi/(nx-1)))
                             + (2/h^2)(1-cos(q*pi/(ny-1))).
    """
    ny, nx = shape
    p = np.arange(1, nx - 1)
    q = np.arange(1, ny - 1)
    lam_x = (2.0 / h**2) * (1.0 - np.cos(p * np.pi / (nx - 1)))
    lam_y = (2.0 / h**2) * (1.0 - np.cos(q * np.pi / (ny - 1)))
    return np.sort((lam_y[:, None] + lam_x[None, :]).ravel())


def test_returns_sparse_square_matrix_sized_to_the_interior():
    shape = (6, 7)
    A = interior_laplacian(shape, h=0.5)
    assert sparse.issparse(A)
    n_interior = (6 - 2) * (7 - 2)
    assert A.shape == (n_interior, n_interior)


def test_operator_is_symmetric():
    A = interior_laplacian((6, 7), h=0.5)
    asym = (A - A.T)
    assert abs(asym).max() < 1e-12


def test_operator_is_positive_definite():
    A = interior_laplacian((6, 7), h=0.5).toarray()
    eigvals = np.linalg.eigvalsh(A)
    assert eigvals.min() > 0.0


def test_spectrum_matches_exact_analytic_eigenvalues():
    shape = (6, 7)
    h = 0.5
    A = interior_laplacian(shape, h).toarray()
    numeric = np.sort(np.linalg.eigvalsh(A))
    analytic = analytic_dirichlet_spectrum(shape, h)
    assert np.allclose(numeric, analytic, atol=1e-10)


def test_stencil_values_are_the_five_point_laplacian():
    h = 0.5
    A = interior_laplacian((5, 5), h).tocsr()
    inv_h2 = 1.0 / h**2
    # every diagonal entry is 4/h^2
    assert np.allclose(A.diagonal(), 4.0 * inv_h2)
    # off-diagonal couplings are either 0 or -1/h^2
    off = A - sparse.diags(A.diagonal())
    vals = off.tocoo().data
    assert np.all(np.isclose(vals, -inv_h2))


def test_q4_stiffness_is_symmetric():
    K = q4_plane_stress_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    assert K.shape == (8, 8)
    assert np.allclose(K, K.T, atol=1e-12)


def test_q4_stiffness_null_space_is_the_three_rigid_body_modes():
    # Translation in x, translation in y, and small rotation are all strain
    # free for a Q4 element, so K must annihilate exactly those 3 modes (out
    # of 8 dof) and be positive semi-definite otherwise.
    K = q4_plane_stress_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    eigvals = np.linalg.eigvalsh(K)
    n_zero = np.sum(np.abs(eigvals) < 1e-9)
    assert n_zero == 3
    assert np.all(eigvals > -1e-9)

    translate_x = np.tile([1.0, 0.0], 4)
    translate_y = np.tile([0.0, 1.0], 4)
    assert np.allclose(K @ translate_x, 0.0, atol=1e-10)
    assert np.allclose(K @ translate_y, 0.0, atol=1e-10)


def test_q4_stiffness_scales_linearly_with_modulus():
    K1 = q4_plane_stress_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    K2 = q4_plane_stress_stiffness(young_modulus=2.5, poisson_ratio=0.3, h=1.0)
    assert np.allclose(K2, 2.5 * K1, atol=1e-12)
