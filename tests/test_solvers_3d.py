"""3D extension of the shared SPD operator and the physics backends.

The operator, heat-conduction solve, and modal eigensolver were first written
and verified in 2D. The discretization is dimension-general (the interior
Laplacian is a Kronecker sum over axes), so the same machine-precision and
FD-gate discipline must hold on a 3D grid. These tests pin that down.
"""

import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.operators import interior_laplacian


def analytic_spectrum(shape, h):
    """Exact eigenvalues of the discrete negative Dirichlet Laplacian, any ndim.

    lambda = sum_axis (2/h^2)(1 - cos(p_axis * pi / (n_axis - 1))).
    """
    per_axis = []
    for n in shape:
        p = np.arange(1, n - 1)
        per_axis.append((2.0 / h**2) * (1.0 - np.cos(p * np.pi / (n - 1))))
    total = per_axis[0]
    for a in per_axis[1:]:
        total = total[..., None] + a
    return np.sort(total.ravel())


# --- operator -------------------------------------------------------------

def test_3d_operator_is_sized_to_the_interior():
    shape = (5, 6, 7)
    A = interior_laplacian(shape, h=0.5)
    n_interior = (5 - 2) * (6 - 2) * (7 - 2)
    assert A.shape == (n_interior, n_interior)


def test_3d_operator_is_symmetric_positive_definite():
    A = interior_laplacian((5, 5, 6), h=0.5).toarray()
    assert np.abs(A - A.T).max() < 1e-12
    assert np.linalg.eigvalsh(A).min() > 0.0


def test_3d_spectrum_matches_exact_analytic_eigenvalues():
    shape = (5, 6, 5)
    h = 0.5
    A = interior_laplacian(shape, h).toarray()
    numeric = np.sort(np.linalg.eigvalsh(A))
    analytic = analytic_spectrum(shape, h)
    assert np.allclose(numeric, analytic, atol=1e-10)


def test_3d_stencil_is_the_seven_point_laplacian():
    from scipy import sparse

    h = 0.5
    A = interior_laplacian((5, 5, 5), h).tocsr()
    inv_h2 = 1.0 / h**2
    # each interior diagonal couples to 6 neighbours: 2*ndim/h^2
    assert np.allclose(A.diagonal(), 6.0 * inv_h2)
    off = A - sparse.diags(A.diagonal())
    assert np.all(np.isclose(off.tocoo().data, -inv_h2))


# --- heat conduction ------------------------------------------------------

def test_heat_3d_adjoint_passes_fd_gate():
    from morphos.physics.heat import HeatConductionOracle

    shape = (6, 6, 6)
    rng = np.random.default_rng(2)
    target = rng.normal(size=shape) * 0.1
    o = HeatConductionOracle(target=target)
    x0 = rng.normal(size=shape) * 0.1
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-5)


def test_heat_3d_solution_satisfies_the_linear_system():
    from morphos.physics.heat import HeatConductionOracle

    shape = (6, 6, 6)
    o = HeatConductionOracle(target=np.zeros(shape), solver="cg")
    rng = np.random.default_rng(4)
    s = rng.normal(size=shape)
    r = o.solve(Field(s, spacing=1.0))
    residual = o.residual(s, r.aux["temperature"])
    assert np.max(np.abs(residual)) < 1e-6


# --- modal ----------------------------------------------------------------

def test_modal_3d_fundamental_matches_analytic_uniform_density():
    from morphos.physics.modal import ModalOracle

    shape = (7, 7, 7)
    h = 1.0
    rho0 = 1.5
    o = ModalOracle(shape=shape, mode=0)
    value = o.solve(Field(np.full(shape, rho0), spacing=h)).value
    analytic = analytic_spectrum(shape, h)[0] / rho0
    assert value == pytest.approx(analytic, rel=1e-7)


def test_modal_3d_adjoint_passes_fd_gate():
    from morphos.physics.modal import ModalOracle

    shape = (6, 6, 6)
    h = 1.0
    o = ModalOracle(shape=shape, mode=0)
    rng = np.random.default_rng(6)
    x0 = 1.0 + 0.3 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=h)).gradient
    f = lambda x: o.solve(Field(x, spacing=h)).value
    fd_gate(f, grad, x0, rel=1e-4)
