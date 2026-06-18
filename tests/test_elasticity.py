import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.elasticity import ElasticityOracle
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


def cantilever_bcs(shape, load=-1.0):
    """Cantilever: fixed left edge nodes, downward point load at the bottom of
    the free (right) edge -- the textbook 2D compliance-minimization test case.
    """
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): load}
    return fixed, loads


def test_elasticity_oracle_is_a_physics_oracle():
    fixed, loads = cantilever_bcs((4, 8))
    o = ElasticityOracle(shape=(4, 8), fixed_dofs=fixed, loads=loads)
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_single_element_compliance_matches_hand_calculation():
    # A single Q4 element, unit square, E=1, nu=0.3, full density (rho=1).
    # Fix the left edge (nodes (0,0) and (0,1)); apply a unit horizontal pull
    # at the top-right node. This is a textbook axial-tension check: for a
    # square plane-stress panel pulled along one edge with the opposite edge
    # fixed, axial stiffness k_axial = E / (1 - nu^2) (unit thickness, unit
    # side), so compliance C = F^2 / k_axial for a single point load aligned
    # with that mode is NOT exact for a single Q4 under a point load (Q4 is not
    # a pure rod), so instead we check the symmetric case: pull both right
    # nodes equally in x, which closely approximates uniaxial tension and must
    # equal the rod stiffness E/(1-nu^2) to a known precision for this coarse
    # a mesh. We instead pin down the exact value via direct hand-assembled K.
    E, nu = 1.0, 0.3
    shape = (1, 1)
    fixed = [(0, 0, "x"), (0, 0, "y"), (0, 1, "x"), (0, 1, "y")]
    loads = {(1, 0, "x"): 1.0, (1, 1, "x"): 1.0}
    o = ElasticityOracle(shape=shape, young_modulus=E, poisson_ratio=nu,
                          fixed_dofs=fixed, loads=loads)
    field = Field(np.ones(shape), spacing=1.0)
    r = o.solve(field)

    # Independent hand computation: assemble the same Q4 stiffness matrix from
    # first principles (2x2 Gauss quadrature plane-stress Q4, a standard,
    # independently-known formula) and solve the same reduced system directly,
    # without using any of the oracle's internals.
    C = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2],
    ])
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    node_xi = np.array([-1, 1, 1, -1])
    node_eta = np.array([-1, -1, 1, 1])
    K8 = np.zeros((8, 8))
    h = 1.0
    J = np.array([[h / 2, 0], [0, h / 2]])
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    for (xi, eta) in pts:
        dN_dxi = 0.25 * node_xi * (1 + node_eta * eta)
        dN_deta = 0.25 * node_eta * (1 + node_xi * xi)
        dN_phys = Jinv @ np.vstack([dN_dxi, dN_deta])
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dN_phys[0, i]
            B[1, 2 * i + 1] = dN_phys[1, i]
            B[2, 2 * i] = dN_phys[1, i]
            B[2, 2 * i + 1] = dN_phys[0, i]
        K8 += (B.T @ C @ B) * detJ
    # Node order in this local K8 is (-1,-1),(1,-1),(1,1),(-1,1) i.e. global
    # nodes (0,0),(1,0),(1,1),(0,1) for our single element with node (x,y)
    # mapped to local coords (2x-1, 2y-1) on the unit square.
    # global dof vector order: [(0,0)x,(0,0)y,(1,0)x,(1,0)y,(1,1)x,(1,1)y,(0,1)x,(0,1)y]
    free = [2, 3, 4, 5]  # (1,0)x,(1,0)y,(1,1)x,(1,1)y
    Kff = K8[np.ix_(free, free)]
    F = np.zeros(4)
    F[0] = 1.0  # (1,0)x
    F[2] = 1.0  # (1,1)x
    u_free = np.linalg.solve(Kff, F)
    compliance_hand = float(F @ u_free)

    assert r.value == pytest.approx(-compliance_hand, rel=1e-8)


def test_fixed_dofs_have_zero_displacement():
    fixed, loads = cantilever_bcs((3, 6))
    o = ElasticityOracle(shape=(3, 6), fixed_dofs=fixed, loads=loads)
    field = Field(np.ones((3, 6)), spacing=1.0)
    r = o.solve(field)
    u = r.aux["displacement"]  # shape (nny, nnx, 2)
    ny, nx = 3, 6
    for j in range(ny + 1):
        assert u[j, 0, 0] == pytest.approx(0.0, abs=1e-10)
        assert u[j, 0, 1] == pytest.approx(0.0, abs=1e-10)


def test_compliance_is_positive_for_a_loaded_structure():
    fixed, loads = cantilever_bcs((4, 8))
    o = ElasticityOracle(shape=(4, 8), fixed_dofs=fixed, loads=loads)
    field = Field(np.ones((4, 8)), spacing=1.0)
    r = o.solve(field)
    assert r.aux["compliance"] > 0.0
    assert r.value == pytest.approx(-r.aux["compliance"])


def test_void_density_floor_keeps_matrix_nonsingular():
    fixed, loads = cantilever_bcs((3, 5))
    o = ElasticityOracle(shape=(3, 5), fixed_dofs=fixed, loads=loads)
    field = Field(np.full((3, 5), 1e-6), spacing=1.0)
    r = o.solve(field)  # must not raise / produce nan
    assert np.isfinite(r.value)
    assert np.all(np.isfinite(r.aux["displacement"]))


def test_higher_density_reduces_compliance():
    fixed, loads = cantilever_bcs((4, 8))
    o = ElasticityOracle(shape=(4, 8), fixed_dofs=fixed, loads=loads)
    soft = o.solve(Field(np.full((4, 8), 0.3), spacing=1.0)).aux["compliance"]
    stiff = o.solve(Field(np.full((4, 8), 1.0), spacing=1.0)).aux["compliance"]
    assert stiff < soft


def test_simp_gradient_matches_finite_differences():
    fixed, loads = cantilever_bcs((4, 6))
    o = ElasticityOracle(shape=(4, 6), fixed_dofs=fixed, loads=loads)
    rng = np.random.default_rng(7)
    rho = 0.3 + 0.6 * rng.uniform(size=(4, 6))
    field = Field(rho, spacing=1.0)
    analytic = o.solve(field).gradient
    numeric = central_fd_gradient(o, field, eps=1e-6)
    assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_simp_gradient_passes_directional_fd_gate():
    fixed, loads = cantilever_bcs((5, 9))
    o = ElasticityOracle(shape=(5, 9), fixed_dofs=fixed, loads=loads)
    rng = np.random.default_rng(11)
    x0 = 0.3 + 0.6 * rng.uniform(size=(5, 9))
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)


def test_field_shape_mismatch_raises():
    fixed, loads = cantilever_bcs((4, 8))
    o = ElasticityOracle(shape=(4, 8), fixed_dofs=fixed, loads=loads)
    with pytest.raises(ValueError):
        o.solve(Field(np.ones((3, 8)), spacing=1.0))


def test_engine_drives_simp_topology_optimization_and_reduces_compliance():
    shape = (10, 20)
    fixed, loads = cantilever_bcs(shape, load=-1.0)
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)
    volume_frac = 0.4
    initial = Field(np.full(shape, volume_frac), spacing=1.0)

    spec = DesignSpec(
        initial=initial,
        oracle=oracle,
        objective=MaximizeValue(bound=PhysicalBound(value=0.0, name="rigid-limit")),
        optimizer=TopologyOptimizer(
            step_size=2e-3, max_iter=60, tol=-1.0, bounds=(1e-3, 1.0)
        ),
        name="cantilever-simp",
    )
    res = Engine().run(spec)
    assert res.history[-1] > res.history[0]
    assert res.figure_of_merit <= 0.0  # never beats the rigid-body ceiling of 0
