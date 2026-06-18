import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.darcy import DarcyFlowOracle
from morphos.objective.objective import MaximizeValue, PhysicalBound
from morphos.optimize.topopt import TopologyOptimizer
from morphos.spec import DesignSpec
from morphos.engine import Engine


def central_fd_gradient(oracle, field, eps=1e-6):
    grad = np.zeros_like(field.values)
    it = np.nditer(field.values, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        fp = field.copy()
        fp.values[idx] += eps
        fm = field.copy()
        fm.values[idx] -= eps
        grad[idx] = (oracle.solve(fp).value - oracle.solve(fm).value) / (2 * eps)
        it.iternext()
    return grad


def channel_bcs(shape, source=1.0):
    """Fix the left-edge nodes at zero pressure (outlet); inject flow at the
    bottom-right node (inlet) -- the channel-routing analogue of the
    elasticity cantilever test case."""
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j) for j in range(nny)]
    sources = {(nnx - 1, nny - 1): source}
    return fixed, sources


def test_darcy_oracle_is_a_physics_oracle():
    fixed, sources = channel_bcs((4, 8))
    o = DarcyFlowOracle(shape=(4, 8), fixed_nodes=fixed, sources=sources)
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_single_element_dissipation_matches_hand_calculation():
    k, h = 1.0, 1.0
    shape = (1, 1)
    fixed = [(0, 0), (0, 1)]
    sources = {(1, 0): 1.0, (1, 1): 1.0}
    o = DarcyFlowOracle(shape=shape, permeability0=k, fixed_nodes=fixed, sources=sources)
    field = Field(np.ones(shape), spacing=h)
    r = o.solve(field)

    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    node_xi = np.array([-1, 1, 1, -1], dtype=float)
    node_eta = np.array([-1, -1, 1, 1], dtype=float)
    inv_j, det_j = 2.0 / h, (h / 2.0) ** 2
    K4 = np.zeros((4, 4))
    for xi, eta in pts:
        dN_dxi = 0.25 * node_xi * (1 + node_eta * eta)
        dN_deta = 0.25 * node_eta * (1 + node_xi * xi)
        G = np.vstack([inv_j * dN_dxi, inv_j * dN_deta])
        K4 += k * (G.T @ G) * det_j
    # local node order (0,0),(1,0),(1,1),(0,1); fixed are local indices 0,3
    free = [1, 2]
    Kff = K4[np.ix_(free, free)]
    F = np.array([1.0, 1.0])
    p_free = np.linalg.solve(Kff, F)
    dissipation_hand = float(F @ p_free)

    assert r.value == pytest.approx(-dissipation_hand, rel=1e-8)


def test_fixed_nodes_have_zero_pressure():
    fixed, sources = channel_bcs((3, 6))
    o = DarcyFlowOracle(shape=(3, 6), fixed_nodes=fixed, sources=sources)
    field = Field(np.ones((3, 6)), spacing=1.0)
    r = o.solve(field)
    p = r.aux["pressure"]
    assert np.allclose(p[:, 0], 0.0, atol=1e-10)


def test_dissipation_is_positive_for_a_driven_flow():
    fixed, sources = channel_bcs((4, 8))
    o = DarcyFlowOracle(shape=(4, 8), fixed_nodes=fixed, sources=sources)
    r = o.solve(Field(np.ones((4, 8)), spacing=1.0))
    assert r.aux["dissipation"] > 0.0
    assert r.value == pytest.approx(-r.aux["dissipation"])


def test_void_permeability_floor_keeps_matrix_nonsingular():
    fixed, sources = channel_bcs((3, 5))
    o = DarcyFlowOracle(shape=(3, 5), fixed_nodes=fixed, sources=sources)
    field = Field(np.full((3, 5), 1e-6), spacing=1.0)
    r = o.solve(field)
    assert np.isfinite(r.value)
    assert np.all(np.isfinite(r.aux["pressure"]))


def test_higher_density_reduces_dissipation():
    fixed, sources = channel_bcs((4, 8))
    o = DarcyFlowOracle(shape=(4, 8), fixed_nodes=fixed, sources=sources)
    soft = o.solve(Field(np.full((4, 8), 0.3), spacing=1.0)).aux["dissipation"]
    open_ = o.solve(Field(np.full((4, 8), 1.0), spacing=1.0)).aux["dissipation"]
    assert open_ < soft


def test_simp_gradient_matches_finite_differences():
    fixed, sources = channel_bcs((4, 6))
    o = DarcyFlowOracle(shape=(4, 6), fixed_nodes=fixed, sources=sources)
    rng = np.random.default_rng(7)
    rho = 0.3 + 0.6 * rng.uniform(size=(4, 6))
    field = Field(rho, spacing=1.0)
    analytic = o.solve(field).gradient
    numeric = central_fd_gradient(o, field, eps=1e-6)
    assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_simp_gradient_passes_directional_fd_gate():
    fixed, sources = channel_bcs((5, 9))
    o = DarcyFlowOracle(shape=(5, 9), fixed_nodes=fixed, sources=sources)
    rng = np.random.default_rng(11)
    x0 = 0.3 + 0.6 * rng.uniform(size=(5, 9))
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)


def test_field_shape_mismatch_raises():
    fixed, sources = channel_bcs((4, 8))
    o = DarcyFlowOracle(shape=(4, 8), fixed_nodes=fixed, sources=sources)
    with pytest.raises(ValueError):
        o.solve(Field(np.ones((3, 8)), spacing=1.0))


def channel_bcs_3d(shape, source=1.0):
    nz, ny, nx = shape
    nnz, nny, nnx = nz + 1, ny + 1, nx + 1
    fixed = [(0, j, k) for j in range(nny) for k in range(nnz)]
    sources = {(nnx - 1, nny - 1, 0): source}
    return fixed, sources


def test_darcy_3d_fixed_nodes_have_zero_pressure():
    shape = (3, 4, 6)
    fixed, sources = channel_bcs_3d(shape)
    o = DarcyFlowOracle(shape=shape, fixed_nodes=fixed, sources=sources)
    r = o.solve(Field(np.ones(shape), spacing=1.0))
    p = r.aux["pressure"]
    assert np.allclose(p[:, :, 0], 0.0, atol=1e-10)


def test_engine_drives_simp_channel_optimization_and_reduces_dissipation():
    shape = (10, 20)
    fixed, sources = channel_bcs(shape, source=1.0)
    oracle = DarcyFlowOracle(shape=shape, fixed_nodes=fixed, sources=sources)
    volume_frac = 0.4
    initial = Field(np.full(shape, volume_frac), spacing=1.0)

    spec = DesignSpec(
        initial=initial,
        oracle=oracle,
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="open-channel-limit")),
        optimizer=TopologyOptimizer(
            step_size=2e-3, max_iter=60, tol=-1.0, bounds=(1e-3, 1.0)
        ),
        name="channel-routing-simp",
    )
    res = Engine().run(spec)
    assert res.history[-1] > res.history[0]
    assert res.figure_of_merit <= 0.0  # never beats the open-channel ceiling of 0


def test_darcy_3d_simp_gradient_passes_directional_fd_gate():
    shape = (3, 3, 4)
    fixed, sources = channel_bcs_3d(shape)
    o = DarcyFlowOracle(shape=shape, fixed_nodes=fixed, sources=sources)
    rng = np.random.default_rng(13)
    x0 = 0.3 + 0.6 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)
