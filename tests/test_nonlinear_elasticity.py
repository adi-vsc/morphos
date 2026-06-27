"""Tests for CorotationalQ4Oracle: geometric nonlinear (co-rotational) Q4 FEM.

Gates the analytic self-adjoint sensitivity against central finite
differences (the engine's standing rule: never trust a textbook sensitivity
formula without an independent FD check), and checks the small-load /
large-load limits against ElasticityOracle.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.nonlinear_elasticity import CorotationalQ4Oracle


def cantilever_bcs(shape):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    return fixed, loads


def make_corot(shape, load=-1.0, **kw):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): load}
    return CorotationalQ4Oracle(shape=shape, fixed_dofs=fixed, loads=loads, **kw)


def make_linear(shape, load=-1.0, **kw):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): load}
    return ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads, **kw)


def fd_gradient(oracle, field, eps=1e-5):
    grad = np.zeros_like(field.values)
    for idx in np.ndindex(field.values.shape):
        fp = field.copy()
        fp.values[idx] += eps
        fm = field.copy()
        fm.values[idx] -= eps
        grad[idx] = (oracle.solve(fp).value - oracle.solve(fm).value) / (2 * eps)
    return grad


def test_corot_is_physics_oracle():
    o = make_corot((4, 8))
    assert isinstance(o, PhysicsOracle)


def test_corot_small_displacement_matches_linear():
    shape = (4, 8)
    rng = np.random.default_rng(0)
    rho = 0.3 + 0.6 * rng.uniform(size=shape)
    field = Field(rho, spacing=1.0)

    corot = make_corot(shape, load=-0.001)
    linear = make_linear(shape, load=-0.001)

    r_corot = corot.solve(field)
    r_linear = linear.solve(field)

    assert r_corot.value == pytest.approx(r_linear.value, rel=1e-2)


def test_corot_value_is_negative_compliance():
    shape = (4, 8)
    field = Field(np.full(shape, 0.5), spacing=1.0)
    o = make_corot(shape)
    r = o.solve(field)
    assert r.value < 0.0
    assert r.value == pytest.approx(-r.aux["compliance"])


def test_corot_gradient_matches_fd():
    # Use a small load so the nonlinear solver converges quickly and the
    # adjoint sensitivity is evaluated at the true equilibrium state.
    # (Large loads on this flexible cantilever geometry exceed the radius of
    # convergence of the simplified co-rotational tangent in one load step.)
    shape = (3, 4)
    rng = np.random.default_rng(1)
    rho = 0.4 + 0.5 * rng.uniform(size=shape)
    field = Field(rho, spacing=1.0)

    o = make_corot(shape, load=-0.002, nl_tol=1e-6)
    result = o.solve(field)
    assert result.aux["nr_iterations"] < o.max_nl_iter, "NR did not converge — gradient unreliable"
    analytic = result.gradient

    fd = fd_gradient(o, field, eps=1e-5)

    rel_err = np.abs(analytic - fd) / (np.abs(fd) + 1e-8)
    assert np.max(rel_err) < 0.05, f"max relative error {np.max(rel_err):.4f}"


def test_corot_provides_gradient_true():
    o = make_corot((3, 4))
    assert o.provides_gradient is True


def test_corot_larger_load_diverges_from_linear():
    shape = (4, 8)
    rho = np.full(shape, 0.6)
    field = Field(rho, spacing=1.0)

    corot = make_corot(shape, load=-10.0)
    linear = make_linear(shape, load=-10.0)

    r_corot = corot.solve(field)
    r_linear = linear.solve(field)

    assert abs(r_corot.value - r_linear.value) > 1e-4


def test_corot_newton_raphson_converges():
    # Use a stiff, lightly-loaded problem so NR converges well inside the cap.
    # rho=1.0 (fully solid), load=-0.05 → linear tip deflection ≈ 0.1 << beam
    # length, keeping deformations in the genuinely small-rotation regime where
    # the simplified co-rotational tangent has fast convergence.
    shape = (3, 4)
    field = Field(np.full(shape, 1.0), spacing=1.0)
    o = make_corot(shape, load=-0.05, max_nl_iter=20)
    r = o.solve(field)
    assert r.aux["nr_iterations"] < o.max_nl_iter


def test_corot_zero_internal_force_under_rigid_rotation():
    """A pure rigid-body rotation of all nodes produces zero internal force.
    This confirms the co-rotational formulation is frame-indifferent (objective)."""
    shape = (2, 4)
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = CorotationalQ4Oracle(shape=shape, fixed_dofs=fixed, loads=loads)

    h = 1.0
    x_ref_coords = oracle._node_ref_coords(h)   # (n_node, 2)
    x_ref = x_ref_coords.ravel()                # (n_dof,), interleaved [x0,y0,x1,y1,...]

    # Build a rigid rotation of 10 degrees about the origin
    theta = np.deg2rad(10.0)
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    I2 = np.eye(2)
    # u_rigid_node_i = (R2 - I2) @ x_ref_node_i
    # Vectorised: x_ref_coords is (n_node, 2) with row vectors
    u_rigid_coords = x_ref_coords @ (R2 - I2).T   # (n_node, 2)
    u_rigid = u_rigid_coords.ravel()              # (n_dof,)

    rho_flat = np.ones(int(np.prod(shape)))
    _, f_int = oracle._assemble(u_rigid, rho_flat, h, x_ref)

    assert np.linalg.norm(f_int) < 1e-10, (
        f"Internal force norm {np.linalg.norm(f_int):.2e} is not zero under "
        "pure rigid rotation — co-rotational formulation is NOT frame-indifferent"
    )
