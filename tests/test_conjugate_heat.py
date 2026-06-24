"""Tests for the conjugate-heat-transfer oracle (advection-diffusion with a
frozen velocity field, SUPG-stabilised, SIMP conductivity)."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.conjugate_heat import ConjugateHeatOracle
from morphos.physics.operators import q4_diffusion_stiffness
from morphos.physics.oracle import PhysicsOracle


def _inlet_nodes(nny, nnx):
    return [iy * nnx + 0 for iy in range(nny)]


def _outlet_nodes(nny, nnx):
    return [iy * nnx + (nnx - 1) for iy in range(nny)]


def test_is_a_physics_oracle():
    shape = (4, 6)
    nny, nnx = shape[0] + 1, shape[1] + 1
    o = ConjugateHeatOracle(
        shape=shape,
        velocity=np.zeros((nny, nnx, 2)),
        source=np.ones(shape),
        fixed_nodes=_inlet_nodes(nny, nnx),
        fixed_values=[0.0] * nny,
    )
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_zero_velocity_recovers_pure_conduction():
    """With u=0 the oracle must reduce to the SIMP conduction system, verified
    against an independent finite-element assembly."""
    shape = (4, 5)
    nely, nelx = shape
    nny, nnx = nely + 1, nelx + 1
    h = 1.0
    rng = np.random.default_rng(1)
    rho = 0.3 + 0.6 * rng.uniform(size=shape)
    Q = rng.uniform(size=shape)
    fixed = _inlet_nodes(nny, nnx)
    o = ConjugateHeatOracle(
        shape=shape,
        velocity=np.zeros((nny, nnx, 2)),
        source=Q,
        fixed_nodes=fixed,
        fixed_values=[0.0] * len(fixed),
        k_solid=2.0,
        k_fluid=0.1,
        p_simp=3.0,
    )
    T_oracle = o.solve(Field(rho, spacing=h)).aux["temperature"]

    # Independent conduction reference.
    n_nodes = nny * nnx
    k_e = 0.1 + rho.ravel() ** 3 * (2.0 - 0.1)
    Ke0 = q4_diffusion_stiffness(1.0, h)
    rows, cols, vals, f = [], [], [], np.zeros(n_nodes)
    e = 0
    for iy in range(nely):
        for ix in range(nelx):
            nodes = [iy * nnx + ix, iy * nnx + ix + 1,
                     (iy + 1) * nnx + ix + 1, (iy + 1) * nnx + ix]
            Ke = k_e[e] * Ke0
            for a in range(4):
                f[nodes[a]] += Q.ravel()[e] * (h * h / 4.0)
                for b in range(4):
                    rows.append(nodes[a]); cols.append(nodes[b]); vals.append(Ke[a, b])
            e += 1
    K = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    free = np.setdiff1d(np.arange(n_nodes), fixed)
    Tref = np.zeros(n_nodes)
    Tref[free] = spsolve(K[np.ix_(free, free)].tocsc(), f[free])
    assert np.allclose(T_oracle.ravel(), Tref, atol=1e-9)


def test_convection_increases_outlet_temperature():
    """A hot inlet carried downstream: higher Peclet (faster flow) raises the
    mean temperature at the outlet."""
    shape = (4, 10)
    nely, nelx = shape
    nny, nnx = nely + 1, nelx + 1
    fixed = _inlet_nodes(nny, nnx)
    outlet = _outlet_nodes(nny, nnx)

    def run(u_x):
        vel = np.zeros((nny, nnx, 2))
        vel[:, :, 0] = u_x
        o = ConjugateHeatOracle(
            shape=shape, velocity=vel, source=np.zeros(shape),
            fixed_nodes=fixed, fixed_values=[1.0] * len(fixed),
            k_solid=1.0, k_fluid=1.0, rho_cp=1.0,
        )
        T = o.solve(Field(np.ones(shape), spacing=1.0)).aux["temperature"].ravel()
        return float(np.mean(T[outlet]))

    slow = run(0.5)
    fast = run(5.0)
    assert fast > slow


def test_gradient_passes_directional_fd_gate():
    shape = (4, 4)
    nely, nelx = shape
    nny, nnx = nely + 1, nelx + 1
    rng = np.random.default_rng(7)
    vel = np.zeros((nny, nnx, 2))
    vel[:, :, 0] = 0.8  # uniform crossflow
    vel[:, :, 1] = 0.3 * rng.uniform(size=(nny, nnx))
    fixed = _inlet_nodes(nny, nnx)
    o = ConjugateHeatOracle(
        shape=shape, velocity=vel, source=np.ones(shape),
        fixed_nodes=fixed, fixed_values=[0.0] * len(fixed),
        k_solid=2.0, k_fluid=0.1, rho_cp=1.0, p_simp=3.0,
    )
    x0 = 0.3 + 0.6 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-5)


def test_peclet_scaling_matches_analytic_1d():
    """Quasi-1D advection-diffusion with uniform flow has the analytic solution
    T(x) = (exp(Pe x/L) - 1)/(exp(Pe) - 1); SUPG is nodally exact for it."""
    nelx, nely = 12, 1
    nny, nnx = nely + 1, nelx + 1
    h = 1.0
    L = nelx * h
    U = 2.0
    k = 1.0
    Pe = U * L / k

    vel = np.zeros((nny, nnx, 2))
    vel[:, :, 0] = U
    inlet = _inlet_nodes(nny, nnx)
    outlet = _outlet_nodes(nny, nnx)
    o = ConjugateHeatOracle(
        shape=(nely, nelx), velocity=vel, source=np.zeros((nely, nelx)),
        fixed_nodes=inlet + outlet,
        fixed_values=[0.0] * len(inlet) + [1.0] * len(outlet),
        k_solid=k, k_fluid=k, rho_cp=1.0,
    )
    T = o.solve(Field(np.ones((nely, nelx)), spacing=h)).aux["temperature"]
    x = np.arange(nnx) * h
    analytic = (np.exp(Pe * x / L) - 1.0) / (np.exp(Pe) - 1.0)
    assert np.allclose(T[0, :], analytic, atol=2e-3)
