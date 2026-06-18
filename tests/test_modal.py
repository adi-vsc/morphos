"""The modal (eigenvalue) physics backend.

ModalOracle solves the generalized vibration eigenproblem

    K phi = lambda M(rho) phi

where K is the stiffness (the SPD interior Laplacian) and M is a diagonal mass
matrix set by the design density field rho. The eigenvalue lambda = omega^2 is a
squared natural frequency, so the backend lets the engine shape a density field
toward target resonances (resonators, band gaps).

Two checks pin it down:
  * with uniform density the smallest eigenvalues reduce to the analytic discrete
    Laplacian spectrum divided by the density, verified to machine precision;
  * the eigenvalue sensitivity dlambda/drho is gated by an aggregate directional
    finite difference (per the symbolic-sensitivity discipline).
"""

import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.operators import interior_laplacian


def analytic_dirichlet_spectrum(shape, h):
    ny, nx = shape
    p = np.arange(1, nx - 1)
    q = np.arange(1, ny - 1)
    lam_x = (2.0 / h**2) * (1.0 - np.cos(p * np.pi / (nx - 1)))
    lam_y = (2.0 / h**2) * (1.0 - np.cos(q * np.pi / (ny - 1)))
    return np.sort((lam_y[:, None] + lam_x[None, :]).ravel())


def test_modal_is_a_physics_oracle():
    from morphos.physics.modal import ModalOracle

    o = ModalOracle(shape=(7, 7))
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_fundamental_eigenvalue_matches_analytic_uniform_density():
    from morphos.physics.modal import ModalOracle

    shape = (9, 9)
    h = 0.5
    rho0 = 2.0
    o = ModalOracle(shape=shape, mode=0)
    field = Field(np.full(shape, rho0), spacing=h)
    value = o.solve(field).value
    analytic = analytic_dirichlet_spectrum(shape, h)[0] / rho0
    assert value == pytest.approx(analytic, rel=1e-8)


def test_higher_modes_match_analytic_uniform_density():
    from morphos.physics.modal import ModalOracle

    shape = (9, 9)
    h = 1.0
    rho0 = 1.0
    analytic = analytic_dirichlet_spectrum(shape, h)
    field = Field(np.full(shape, rho0), spacing=h)
    for mode in range(4):
        value = ModalOracle(shape=shape, mode=mode).solve(field).value
        assert value == pytest.approx(analytic[mode] / rho0, rel=1e-7)


def test_eigenvalue_adjoint_passes_aggregate_fd_gate():
    from morphos.physics.modal import ModalOracle

    shape = (8, 8)
    h = 1.0
    o = ModalOracle(shape=shape, mode=0)
    rng = np.random.default_rng(3)
    x0 = 1.0 + 0.3 * rng.uniform(size=shape)  # strictly positive density

    grad = o.solve(Field(x0, spacing=h)).gradient
    f = lambda x: o.solve(Field(x, spacing=h)).value
    fd_gate(f, grad, x0, rel=1e-4)


def test_lobpcg_matches_eigsh():
    from morphos.physics.modal import ModalOracle

    shape = (10, 10)
    h = 1.0
    rng = np.random.default_rng(5)
    field = Field(1.0 + 0.2 * rng.uniform(size=shape), spacing=h)
    v_eigsh = ModalOracle(shape=shape, mode=0, eigensolver="eigsh").solve(field).value
    v_lobpcg = ModalOracle(shape=shape, mode=0, eigensolver="lobpcg").solve(field).value
    assert v_lobpcg == pytest.approx(v_eigsh, rel=1e-6)
