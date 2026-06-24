import numpy as np
import pytest

pytest.importorskip("skfem")

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.stokes import StokesFlowOracle


def parabola(x):
    """Unit-peak parabolic inflow in +x across a left edge of height Ly."""
    y = x[1]
    Ly = y.max()
    ux = 4.0 * y * (Ly - y) / (Ly**2)
    return np.stack([ux, np.zeros_like(y)])


def channel(shape, **kw):
    return StokesFlowOracle(
        shape=shape,
        inlet=("left", parabola),
        noslip_edges=("top", "bottom"),
        **kw,
    )


def test_is_a_physics_oracle():
    o = channel((6, 8))
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_channel_flow_is_physical():
    o = channel((8, 10))
    r = o.solve(Field(np.ones((8, 10)), spacing=1.0))
    assert r.aux["dissipation"] > 0.0
    assert r.value == pytest.approx(-r.aux["dissipation"])
    # inlet parabola peaks at 1.0; the field should not blow past it
    assert 0.5 < np.abs(r.aux["velocity"]).max() < 1.5


def test_no_slip_walls_are_zero():
    o = channel((8, 10))
    r = o.solve(Field(np.ones((8, 10)), spacing=1.0))
    v = r.aux["velocity"]  # (nvy, nvx, 2) on the velocity node grid
    assert np.allclose(v[0, :, :], 0.0, atol=1e-9)   # bottom wall
    assert np.allclose(v[-1, :, :], 0.0, atol=1e-9)  # top wall


def test_blocking_the_channel_costs_more_power():
    # An all-fluid channel dissipates less than one half-filled with solid
    # (low rho = high Brinkman drag) at the same prescribed throughput.
    shape = (8, 10)
    o = channel(shape)
    open_ = o.solve(Field(np.ones(shape), spacing=1.0)).aux["dissipation"]
    blocked = o.solve(Field(np.full(shape, 0.4), spacing=1.0)).aux["dissipation"]
    assert open_ < blocked


def test_dissipation_gradient_passes_directional_fd_gate():
    shape = (6, 8)
    o = channel(shape)
    rng = np.random.default_rng(23)
    x0 = 0.3 + 0.6 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)


def test_field_shape_mismatch_raises():
    o = channel((6, 8))
    with pytest.raises(ValueError):
        o.solve(Field(np.ones((5, 8)), spacing=1.0))


def test_iterative_solver_matches_direct_on_saddle_point_system():
    # The mixed velocity-pressure system is an indefinite saddle point, so
    # the iterative path here is ILU-preconditioned GMRES (not SA-AMG), see
    # StokesFlowOracle.__init__. Pin it against the direct solve.
    shape = (10, 10)
    rng = np.random.default_rng(31)
    rho = 0.3 + 0.6 * rng.uniform(size=shape)
    direct = channel(shape, solver="direct").solve(Field(rho, spacing=1.0))
    it = channel(shape, solver="iterative").solve(Field(rho, spacing=1.0))
    assert direct.solver_iterations == 0
    assert it.solver_iterations > 0
    assert it.residual_norm < 1e-6
    rel_err = abs(it.value - direct.value) / abs(direct.value)
    assert rel_err < 1e-6
