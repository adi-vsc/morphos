"""Tests for the shared finite-difference operators.

The interior Laplacian is the symmetric positive-definite core that both the
heat-conduction solve and the modal eigensolver build on. Its discrete spectrum
is known in closed form, which gives a machine-precision verification target.
"""

import numpy as np
import pytest
from scipy import sparse

from morphos.physics.operators import (
    hex8_diffusion_stiffness,
    hex8_stiffness,
    interior_laplacian,
    q4_diffusion_stiffness,
    q4_plane_stress_stiffness,
)


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


def test_hex8_stiffness_is_symmetric():
    K = hex8_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T, atol=1e-12)


def test_hex8_stiffness_null_space_is_the_six_rigid_body_modes():
    # 3 translations + 3 small rotations are strain-free for a trilinear hex,
    # so K must annihilate exactly those 6 (of 24) modes and be PSD otherwise.
    K = hex8_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    eigvals = np.linalg.eigvalsh(K)
    n_zero = np.sum(np.abs(eigvals) < 1e-8)
    assert n_zero == 6
    assert np.all(eigvals > -1e-8)

    translate_x = np.tile([1.0, 0.0, 0.0], 8)
    translate_y = np.tile([0.0, 1.0, 0.0], 8)
    translate_z = np.tile([0.0, 0.0, 1.0], 8)
    assert np.allclose(K @ translate_x, 0.0, atol=1e-9)
    assert np.allclose(K @ translate_y, 0.0, atol=1e-9)
    assert np.allclose(K @ translate_z, 0.0, atol=1e-9)


def test_hex8_stiffness_scales_linearly_with_modulus():
    K1 = hex8_stiffness(young_modulus=1.0, poisson_ratio=0.3, h=1.0)
    K2 = hex8_stiffness(young_modulus=2.5, poisson_ratio=0.3, h=1.0)
    assert np.allclose(K2, 2.5 * K1, atol=1e-12)


def test_hex8_stiffness_matches_independent_quadrature():
    # Independently re-derive the same physics (trilinear shape functions, 2x2x2
    # Gauss quadrature, isotropic 3D constitutive matrix) without touching any
    # of the production code, mirroring the Q4 single-element hand check in
    # test_elasticity.py.
    E, nu, h = 3.0, 0.25, 0.7
    lam_c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = lam_c * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
    ])
    node_xi = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    node_eta = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
    node_zeta = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    gp = 1.0 / np.sqrt(3.0)
    pts = [(a, b, c) for a in (-gp, gp) for b in (-gp, gp) for c in (-gp, gp)]
    inv_j = 2.0 / h
    det_j = (h / 2.0) ** 3
    K = np.zeros((24, 24))
    for xi, eta, zeta in pts:
        dN_dxi = 0.125 * node_xi * (1 + node_eta * eta) * (1 + node_zeta * zeta)
        dN_deta = 0.125 * node_eta * (1 + node_xi * xi) * (1 + node_zeta * zeta)
        dN_dzeta = 0.125 * node_zeta * (1 + node_xi * xi) * (1 + node_eta * eta)
        dN_dx, dN_dy, dN_dz = inv_j * dN_dxi, inv_j * dN_deta, inv_j * dN_dzeta
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
        K += (B.T @ C @ B) * det_j

    got = hex8_stiffness(young_modulus=E, poisson_ratio=nu, h=h)
    assert np.allclose(got, K, atol=1e-10)


def test_q4_diffusion_stiffness_is_symmetric():
    K = q4_diffusion_stiffness(conductivity=1.0, h=1.0)
    assert K.shape == (4, 4)
    assert np.allclose(K, K.T, atol=1e-12)


def test_q4_diffusion_stiffness_null_space_is_the_constant_mode():
    # A uniform potential has zero gradient everywhere, so K must annihilate
    # exactly the constant vector (1 of 4 modes) and be PSD otherwise.
    K = q4_diffusion_stiffness(conductivity=1.0, h=1.0)
    eigvals = np.linalg.eigvalsh(K)
    n_zero = np.sum(np.abs(eigvals) < 1e-9)
    assert n_zero == 1
    assert np.all(eigvals > -1e-9)
    assert np.allclose(K @ np.ones(4), 0.0, atol=1e-10)


def test_q4_diffusion_stiffness_scales_linearly_with_conductivity():
    K1 = q4_diffusion_stiffness(conductivity=1.0, h=1.0)
    K2 = q4_diffusion_stiffness(conductivity=2.5, h=1.0)
    assert np.allclose(K2, 2.5 * K1, atol=1e-12)


def test_q4_diffusion_stiffness_matches_independent_quadrature():
    k, h = 1.7, 0.6
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    node_xi = np.array([-1, 1, 1, -1], dtype=float)
    node_eta = np.array([-1, -1, 1, 1], dtype=float)
    inv_j = 2.0 / h
    det_j = (h / 2.0) ** 2
    K = np.zeros((4, 4))
    for xi, eta in pts:
        dN_dxi = 0.25 * node_xi * (1 + node_eta * eta)
        dN_deta = 0.25 * node_eta * (1 + node_xi * xi)
        G = np.vstack([inv_j * dN_dxi, inv_j * dN_deta])  # (2,4)
        K += k * (G.T @ G) * det_j
    got = q4_diffusion_stiffness(conductivity=k, h=h)
    assert np.allclose(got, K, atol=1e-10)


def test_hex8_diffusion_stiffness_is_symmetric():
    K = hex8_diffusion_stiffness(conductivity=1.0, h=1.0)
    assert K.shape == (8, 8)
    assert np.allclose(K, K.T, atol=1e-12)


def test_hex8_diffusion_stiffness_null_space_is_the_constant_mode():
    K = hex8_diffusion_stiffness(conductivity=1.0, h=1.0)
    eigvals = np.linalg.eigvalsh(K)
    n_zero = np.sum(np.abs(eigvals) < 1e-8)
    assert n_zero == 1
    assert np.all(eigvals > -1e-8)
    assert np.allclose(K @ np.ones(8), 0.0, atol=1e-9)


def test_hex8_diffusion_stiffness_matches_independent_quadrature():
    k, h = 1.3, 0.8
    node_xi = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    node_eta = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
    node_zeta = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    gp = 1.0 / np.sqrt(3.0)
    pts = [(a, b, c) for a in (-gp, gp) for b in (-gp, gp) for c in (-gp, gp)]
    inv_j = 2.0 / h
    det_j = (h / 2.0) ** 3
    K = np.zeros((8, 8))
    for xi, eta, zeta in pts:
        dN_dxi = 0.125 * node_xi * (1 + node_eta * eta) * (1 + node_zeta * zeta)
        dN_deta = 0.125 * node_eta * (1 + node_xi * xi) * (1 + node_zeta * zeta)
        dN_dzeta = 0.125 * node_zeta * (1 + node_xi * xi) * (1 + node_eta * eta)
        G = np.vstack([inv_j * dN_dxi, inv_j * dN_deta, inv_j * dN_dzeta])  # (3,8)
        K += k * (G.T @ G) * det_j
    got = hex8_diffusion_stiffness(conductivity=k, h=h)
    assert np.allclose(got, K, atol=1e-10)
