"""Integration tests for MMAOptimizer with various physics oracles.

These exercise the full optimization loop (filter -> Heaviside -> volume
constraint -> oracle solve -> MMA subproblem) against every physics backend
that exposes the PhysicsOracle interface, rather than re-testing any single
oracle's gradient (that is the job of each oracle's own test module).
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.optimize.mma import MMAOptimizer
from morphos.objective.objective import MaximizeValue
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.heat import HeatConductionOracle
from morphos.physics.darcy import DarcyFlowOracle
from morphos.physics.thermoelastic import ThermoElasticOracle

try:
    from morphos.physics.stokes import StokesFlowOracle

    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False


def cantilever_bcs(shape, load=-1.0):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny - 1, "y"): load}
    return fixed, loads


def cantilever_oracle(shape=(10, 20), load=-1.0, **kw):
    fixed, loads = cantilever_bcs(shape, load=load)
    return ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads, **kw)


# ---------------------------------------------------------------------------
# 1. elasticity cantilever
# ---------------------------------------------------------------------------


def test_mma_elasticity_cantilever():
    shape = (10, 20)
    oracle = cantilever_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=15)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(result.history) >= 1
    # FOM should improve over the run (compliance objective starts negative,
    # optimization should push it up toward zero).
    assert result.history[-1] >= result.history[0]
    assert result.field.values.mean() <= 0.42


# ---------------------------------------------------------------------------
# 2. heat conduction
# ---------------------------------------------------------------------------


def test_mma_heat_conduction():
    shape = (8, 12)
    target = np.zeros(shape)
    # Encourage heat to flow from a left-edge "source" toward the right edge
    # (which is implicitly held at zero temperature by HeatConductionOracle's
    # finite-difference Dirichlet boundary). The design field doubles as the
    # distributed source term.
    target[:, -1] = 0.0
    oracle = HeatConductionOracle(target=target)

    initial_vals = np.full(shape, 0.4, dtype=float)
    initial_vals[:, 0] = 1.0  # left-edge heat source
    initial = Field(initial_vals, spacing=1.0)

    opt = MMAOptimizer(volume_fraction=0.4, max_iter=10)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(result.history) >= 1
    assert all(np.isfinite(h) for h in result.history)


# ---------------------------------------------------------------------------
# 3. darcy flow
# ---------------------------------------------------------------------------


def darcy_oracle(shape=(8, 16), **kw):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed_nodes = [(nnx - 1, j) for j in range(nny)]  # right edge = outlet (p=0)
    sources = {(0, nny // 2): 1.0}  # inlet on the left, mid-height
    return DarcyFlowOracle(shape=shape, fixed_nodes=fixed_nodes, sources=sources, **kw)


def test_mma_darcy_flow():
    shape = (8, 16)
    oracle = darcy_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=10)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(result.history) >= 1
    # convergence: improvement (or at least no blow-up) across the run
    assert result.history[-1] >= result.history[0] - 1e-6
    assert np.isfinite(result.history[-1])


# ---------------------------------------------------------------------------
# 4. stokes flow (optional dependency)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SKFEM, reason="scikit-fem not installed")
def test_mma_stokes_flow():
    def parabola(x):
        y = x[1]
        Ly = y.max()
        ux = 4.0 * y * (Ly - y) / (Ly ** 2)
        return np.stack([ux, np.zeros_like(y)])

    shape = (6, 8)
    oracle = StokesFlowOracle(
        shape=shape,
        inlet=("left", parabola),
        noslip_edges=("top", "bottom"),
    )
    initial = Field(np.full(shape, 0.5, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.5, max_iter=5)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(result.history) >= 1
    assert all(np.isfinite(h) for h in result.history)


# ---------------------------------------------------------------------------
# 5. thermoelastic
# ---------------------------------------------------------------------------


def thermoelastic_oracle(shape=(6, 10), **kw):
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed_dofs = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny // 2, "y"): -1.0}
    fixed_temps = [(0, j) for j in range(nny)]
    heat_sources = {(nnx - 1, nny - 1): 1.0}
    return ThermoElasticOracle(
        shape=shape,
        fixed_dofs=fixed_dofs,
        loads=loads,
        fixed_temps=fixed_temps,
        heat_sources=heat_sources,
        **kw,
    )


def test_mma_thermoelastic():
    shape = (6, 10)
    oracle = thermoelastic_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=5)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(result.history) >= 1
    assert all(np.isfinite(h) for h in result.history)


# ---------------------------------------------------------------------------
# 6. PDE filter
# ---------------------------------------------------------------------------


def test_mma_with_pde_filter():
    shape = (10, 20)
    oracle = cantilever_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=10, filter_radius=1.5)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert result.field.values.mean() <= 0.42 + 1e-2


# ---------------------------------------------------------------------------
# 7. volume constraint at every iteration
# ---------------------------------------------------------------------------


def test_mma_volume_constraint_all_iterations():
    """Track the post-filter/Heaviside design volume at every iteration the
    same way MMAOptimizer's own volume constraint sees it, via a subclass
    that records `_design_chain`'s output (the exact quantity the bisection
    in `_mma_step` targets)."""
    shape = (10, 20)
    oracle = cantilever_oracle(shape)
    initial = Field(np.full(shape, 0.5, dtype=float), spacing=1.0)

    seen_volumes = []

    class _RecordingOptimizer(MMAOptimizer):
        def _design_chain(self, x, beta, constraint):
            design, filtered = super()._design_chain(x, beta, constraint)
            seen_volumes.append(float(design.values.mean()))
            return design, filtered

    opt = _RecordingOptimizer(volume_fraction=0.5, max_iter=12)
    result = opt.run(initial, oracle, MaximizeValue())

    assert np.isfinite(result.fom)
    assert len(seen_volumes) >= 1
    for v in seen_volumes:
        assert v <= 0.53


# ---------------------------------------------------------------------------
# 8. history monotone-ish after warmup
# ---------------------------------------------------------------------------


def test_mma_history_monotone_after_warmup():
    shape = (10, 20)
    oracle = cantilever_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=5)
    warmup = opt.run(initial, oracle, MaximizeValue())

    # Continue from the warmup's best field for a longer run.
    opt2 = MMAOptimizer(volume_fraction=0.4, max_iter=15)
    result = opt2.run(warmup.field, oracle, MaximizeValue())

    initial_mag = abs(result.history[0]) + 1e-12
    for h in result.history:
        assert np.isfinite(h)
        assert abs(h) < 50.0 * initial_mag


# ---------------------------------------------------------------------------
# 9. converged flag
# ---------------------------------------------------------------------------


def test_mma_converged_flag():
    shape = (8, 16)
    oracle = cantilever_oracle(shape)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=200, tol=1e-6)
    result = opt.run(initial, oracle, MaximizeValue())

    assert isinstance(result.converged, bool)
    # Well-conditioned cantilever compliance problem should settle down given
    # 200 iterations to chase a tight tolerance.
    assert result.converged is True


# ---------------------------------------------------------------------------
# 10. best field is not necessarily the last
# ---------------------------------------------------------------------------


def test_mma_best_field_is_not_last():
    shape = (8, 16)
    fixed, loads = cantilever_bcs(shape)

    class NoisyElasticityOracle(ElasticityOracle):
        """Wrap ElasticityOracle's value with an oscillating perturbation so
        the raw figure of merit is non-monotone across iterations, exercising
        the optimizer's "track best, not last" bookkeeping."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._call = 0

        def solve(self, field):
            result = super().solve(field)
            self._call += 1
            # Even calls get a downward kick; this perturbs value (and hence
            # fom) without touching the gradient, so the optimizer keeps
            # moving sensibly while its FOM history oscillates.
            if self._call % 2 == 0:
                result.value -= 10.0 * abs(result.value) + 1.0
            return result

    oracle = NoisyElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads)
    initial = Field(np.full(shape, 0.4, dtype=float), spacing=1.0)
    opt = MMAOptimizer(volume_fraction=0.4, max_iter=12)
    result = opt.run(initial, oracle, MaximizeValue())

    # The best (returned) fom must be at least as good as every fom seen in
    # history -- the optimizer must keep the *best* field, not the *last*.
    assert all(result.fom >= h - 1e-9 for h in result.history)
    assert result.fom >= max(result.history) - 1e-9
