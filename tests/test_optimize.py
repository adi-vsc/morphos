import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import TopologyOptimizer


class NoGradientOracle(PhysicsOracle):
    """Same response as AnalyticOracle but withholds the gradient."""

    provides_gradient = False

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def solve(self, field):
        diff = field.values - self.target
        return PhysicsResult(value=-float(np.sum(diff ** 2)), gradient=None)


def test_topology_optimizer_is_an_optimizer():
    assert isinstance(TopologyOptimizer(), Optimizer)


def test_recovers_target_with_gradient():
    target = np.array([[0.2, -0.4], [0.9, 0.1]])
    oracle = AnalyticOracle(target=target)
    obj = MaximizeValue()
    opt = TopologyOptimizer(step_size=0.2, max_iter=5000, tol=1e-12)
    res = opt.run(Field(np.zeros((2, 2)), spacing=1.0), oracle, obj)
    assert isinstance(res, OptimizeResult)
    assert res.fom == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.field.values, target, atol=1e-3)
    assert res.used_finite_differences is False


def test_objective_improves_monotonically_on_convex_problem():
    target = np.array([[1.0, 2.0]])
    opt = TopologyOptimizer(step_size=0.2, max_iter=200, tol=0.0)
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), AnalyticOracle(target), MaximizeValue()
    )
    assert res.history[-1] >= res.history[0]


def test_finite_difference_fallback_when_no_gradient():
    target = np.array([[0.3, -0.2]])
    opt = TopologyOptimizer(step_size=0.2, max_iter=4000, tol=1e-12)
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), NoGradientOracle(target), MaximizeValue()
    )
    assert res.used_finite_differences is True
    assert np.allclose(res.field.values, target, atol=1e-2)


def test_bounds_are_respected():
    target = np.array([[5.0, 5.0]])  # outside [0, 1]
    opt = TopologyOptimizer(step_size=0.2, max_iter=500, bounds=(0.0, 1.0))
    res = opt.run(
        Field(np.zeros((1, 2)), spacing=1.0), AnalyticOracle(target), MaximizeValue()
    )
    assert np.all(res.field.values <= 1.0 + 1e-9)
    assert np.all(res.field.values >= 0.0 - 1e-9)


# --- Step 1: SIMP p-continuation, density filtering, Heaviside projection ---


def _cantilever_oracle(shape=(6, 12)):
    from morphos.physics.elasticity import ElasticityOracle

    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    return ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads, penalty=3.0)


def test_p_continuation_improves_compliance_over_fixed_p():
    """A p-continuation schedule (p: 1 -> 3) should reach lower compliance
    (higher figure of merit) than holding p fixed at 3 for the same iteration
    budget, when material is capped below full density (here via an upper
    bound standing in for a volume budget) on a 2D cantilever. At p=3 fixed
    and a uniform mid density, the SIMP gradient scales as rho**(p-1), tiny
    near the cap, so the fixed-p run stalls almost immediately; ramping from
    a near-linear p=1 problem first builds up real structure under the same
    cap before sharpening, which is the entire point of continuation."""
    from morphos.objective.objective import MaximizeValue

    shape = (6, 12)
    init = Field(np.full(shape, 0.3), spacing=1.0)
    bounds = (1e-3, 0.5)

    fixed_p_opt = TopologyOptimizer(
        step_size=1e-3, max_iter=60, tol=0.0, bounds=bounds,
        p_start=3.0, p_end=3.0,
    )
    res_fixed = fixed_p_opt.run(init.copy(), _cantilever_oracle(shape), MaximizeValue())

    continuation_opt = TopologyOptimizer(
        step_size=1e-3, max_iter=60, tol=0.0, bounds=bounds,
        p_start=1.0, p_end=3.0, p_ramp_fraction=0.5,
    )
    res_cont = continuation_opt.run(init.copy(), _cantilever_oracle(shape), MaximizeValue())

    assert res_cont.fom > res_fixed.fom


def test_density_filter_removes_isolated_single_voxel_spikes():
    """A filter_radius > 0 must remove isolated single-voxel high-density
    spikes from the projected design (the classic checkerboard / single-pixel
    artifact that an un-filtered SIMP optimization produces)."""
    shape = (8, 8)
    raw = np.full(shape, 0.2)
    raw[4, 4] = 1.0  # isolated spike, surrounded by low density

    opt = TopologyOptimizer(filter_radius=2)
    filtered = opt._apply_filter(raw)

    assert filtered[4, 4] < 0.5
    # neighbourhood is smoothed, not just the center clipped
    assert filtered[4, 4] < raw[4, 4]


def test_filter_vjp_passes_fd_gate():
    """The density filter's vector-Jacobian product must match a directional
    finite difference of the filtered objective (the FD gate)."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from fd_gate import fd_gate

    rng = np.random.default_rng(0)
    shape = (5, 5)
    x0 = rng.uniform(0.1, 0.9, shape)

    opt = TopologyOptimizer(filter_radius=1)

    def f(flat_x):
        return float(np.sum(opt._apply_filter(flat_x.reshape(shape)) ** 2))

    filtered = opt._apply_filter(x0)
    grad_out = 2.0 * filtered  # d(sum filtered^2)/d(filtered)
    grad_in = opt._filter_vjp(grad_out)

    fd_gate(f, grad_in, x0, rel=1e-4)


def test_heaviside_projection_vjp_passes_fd_gate():
    """The smooth Heaviside projection's derivative must match a directional
    finite difference (the FD gate), at a representative beta."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from fd_gate import fd_gate

    rng = np.random.default_rng(1)
    shape = (4, 4)
    x0 = rng.uniform(0.05, 0.95, shape)

    opt = TopologyOptimizer()
    beta = 8.0

    def f(flat_x):
        return float(np.sum(opt._heaviside_project(flat_x, beta) ** 2))

    proj = opt._heaviside_project(x0, beta)
    grad_out = 2.0 * proj
    grad_in = opt._heaviside_vjp(x0, beta, grad_out)

    fd_gate(f, grad_in, x0, rel=1e-4)


def test_early_stop_patience_fires_before_max_iter():
    """When the figure of merit stops changing by more than tol for `patience`
    consecutive iterations, the optimizer should stop early and report
    converged=True with fewer iterations than max_iter."""
    target = np.array([[0.2, -0.4], [0.9, 0.1]])
    oracle = AnalyticOracle(target=target)
    obj = MaximizeValue()
    opt = TopologyOptimizer(
        step_size=0.2, max_iter=5000, tol=1e-6, patience=5,
    )
    res = opt.run(Field(np.zeros((2, 2)), spacing=1.0), oracle, obj)
    assert res.converged is True
    assert res.iterations < 5000
