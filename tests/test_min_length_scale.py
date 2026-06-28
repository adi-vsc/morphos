"""Tests for the double-filter minimum length scale control (Guest et al. 2004).

The double-filter scheme inserts a second filter-project pass after the first
Heaviside projection in the design chain:

    raw_x -> [filter r_min] -> [Heaviside eta=0.5] -> [filter r_min]
           -> [Heaviside eta=1-min_length_eta] -> constraint -> design

This squeezes solid features thinner than ``min_length_scale`` voxels toward
zero, which is the standard manufacturability guarantee for SIMP topology
optimization. ``min_length_scale=0`` (the default) must be a complete no-op,
preserving the original single-filter chain exactly.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.elasticity import ElasticityOracle
from morphos.objective.objective import MaximizeValue
from morphos.optimize.topopt import TopologyOptimizer
from morphos.optimize.oc import OCOptimizer
from morphos.optimize.mma import MMAOptimizer


def _cantilever_setup(shape=(6, 12), volume_fraction=0.4):
    """Fixed left edge, downward point load at the bottom-right corner.

    Matches the fixture used elsewhere in the test suite (test_oc.py,
    test_optimize.py, test_elasticity.py).
    """
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed = []
    for j in range(nny):
        fixed.append((0, j, "x"))
        fixed.append((0, j, "y"))
    loads = {(nnx - 1, nny - 1, "y"): -1.0}
    oracle = ElasticityOracle(shape=shape, fixed_dofs=fixed, loads=loads, penalty=3.0)
    initial = Field(np.full(shape, volume_fraction, dtype=float), spacing=1.0)
    return oracle, initial, MaximizeValue()


# ---------------------------------------------------------------------------
# Test 1: thin solid features are suppressed
# ---------------------------------------------------------------------------

def test_double_filter_suppresses_thin_solid_features():
    """A 1-voxel-wide solid stripe in a 20x20 grid must be eroded to near-zero
    by the double filter when min_length_scale enforces a wider minimum
    feature size."""
    n = 20
    raw = np.zeros((n, n), dtype=float)
    stripe_col = n // 2
    raw[:, stripe_col] = 1.0  # single-voxel-wide vertical stripe

    field = Field(raw, spacing=1.0)
    opt = TopologyOptimizer(
        filter_radius=2.0,
        min_length_scale=2.0,
        beta_start=16.0,
        beta_end=16.0,
    )
    design, _ = opt._design_chain(field, beta=16.0, constraint=None)

    stripe_region = design.values[:, stripe_col]
    assert stripe_region.max() < 0.1, (
        f"thin stripe should be suppressed by double filter, got max="
        f"{stripe_region.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: chain VJP matches finite differences
# ---------------------------------------------------------------------------

def test_double_filter_vjp_matches_fd():
    """_chain_vjp with min_length_scale > 0 must match a finite-difference
    check of d(mean(design))/d(raw_x), to a tight relative tolerance."""
    rng = np.random.default_rng(42)
    shape = (5, 8)
    raw = rng.uniform(0.2, 0.8, size=shape)
    field = Field(raw, spacing=1.0)

    opt = TopologyOptimizer(
        filter_radius=1.5,
        min_length_scale=1.5,
        beta_start=4.0,
        beta_end=4.0,
    )
    beta = 4.0

    design, filtered = opt._design_chain(field, beta=beta, constraint=None)
    n = design.values.size
    grad_out = np.ones_like(design.values) / n  # d(mean)/d(design) = 1/n

    analytic = opt._chain_vjp(field, filtered, beta, grad_out, constraint=None)

    def mean_design(x_values):
        f = Field(x_values, spacing=1.0)
        d, _ = opt._design_chain(f, beta=beta, constraint=None)
        return float(np.mean(d.values))

    eps = 1e-5
    fd = np.zeros_like(raw)
    it = np.nditer(raw, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = raw.copy()
        xm = raw.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd[idx] = (mean_design(xp) - mean_design(xm)) / (2 * eps)
        it.iternext()

    denom = np.maximum(np.abs(fd), 1e-8)
    rel_err = np.abs(analytic - fd) / denom
    assert np.max(rel_err) < 1e-4 or np.allclose(analytic, fd, atol=1e-6), (
        f"chain_vjp does not match FD: max rel err = {np.max(rel_err):.2e}, "
        f"max abs err = {np.max(np.abs(analytic - fd)):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3: disabled by default (no regression)
# ---------------------------------------------------------------------------

def test_min_length_scale_disabled_by_default():
    """TopologyOptimizer() with min_length_scale=0 (the default) must behave
    exactly like the original single-filter chain: projected values stay in
    [0, 1] and disabling explicitly produces identical output to the default."""
    rng = np.random.default_rng(7)
    shape = (6, 6)
    raw = rng.uniform(0.1, 0.9, size=shape)
    field = Field(raw, spacing=1.0)

    default_opt = TopologyOptimizer(filter_radius=1.0, beta_start=8.0, beta_end=8.0)
    explicit_opt = TopologyOptimizer(
        filter_radius=1.0, beta_start=8.0, beta_end=8.0, min_length_scale=0.0
    )

    design_default, filtered_default = default_opt._design_chain(
        field, beta=8.0, constraint=None
    )
    design_explicit, filtered_explicit = explicit_opt._design_chain(
        field, beta=8.0, constraint=None
    )

    assert design_default.values.min() >= 0.0
    assert design_default.values.max() <= 1.0
    assert np.allclose(design_default.values, design_explicit.values)
    assert np.allclose(filtered_default, filtered_explicit)
    assert default_opt.min_length_scale == 0.0


# ---------------------------------------------------------------------------
# Test 4: OCOptimizer accepts min_length_scale
# ---------------------------------------------------------------------------

def test_oc_optimizer_accepts_min_length_scale():
    """OCOptimizer with min_length_scale > 0 must run without error on a
    small cantilever and improve (or at least not regress) the fom.

    p_start == p_end (no SIMP continuation) here: the double filter's fixed
    erosion threshold already shifts the achievable volume well below the
    raw target, and stacking that with a simultaneous p ramp (which sharpens
    penalization from a near-convex p=1 starting point) creates a transient
    that overwhelms 5 iterations -- the same transient the un-modified
    optimizer shows over a short ramp window. Holding p fixed isolates the
    behavior this test actually targets: that min_length_scale wires through
    OC's update loop correctly and converges normally.
    """
    volume_fraction = 0.4
    oracle, initial, objective = _cantilever_setup(
        shape=(6, 8), volume_fraction=volume_fraction
    )
    opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=5,
        filter_radius=1.0,
        min_length_scale=1.0,
        p_start=3.0,
        p_end=3.0,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    assert len(result.history) > 0
    assert result.history[-1] > result.history[0], (
        f"history should improve: {result.history}"
    )


def test_oc_min_length_not_degenerate():
    """OCOptimizer with min_length_scale > 0 must produce a load-bearing
    design, not an empty/disconnected one.

    Regression test: the original OC double-filter eroded at eta=0.75 first,
    which on a gray (~volume_fraction) density field collapses the whole
    design to void before the physics ever sees structure. The optimized
    compliance must end up STRICTLY BETTER than the uniform-gray baseline --
    a degenerate (near-empty) design has compliance orders of magnitude
    worse than the baseline, so this cleanly separates a working design
    from a collapsed one. Run with the natural defaults (p-continuation on).
    """
    volume_fraction = 0.4
    shape = (8, 16)
    oracle, initial, objective = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    baseline_compliance = oracle.solve(initial).aux["compliance"]

    opt = OCOptimizer(
        volume_fraction=volume_fraction,
        max_iter=30,
        filter_radius=1.5,
        min_length_scale=2.0,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    best_compliance = -result.fom
    assert best_compliance < baseline_compliance, (
        f"OC+min_length design (compliance {best_compliance:.3g}) is worse "
        f"than the uniform baseline ({baseline_compliance:.3g}); the design "
        f"likely collapsed to void."
    )


# ---------------------------------------------------------------------------
# Test 5: MMAOptimizer accepts min_length_scale
# ---------------------------------------------------------------------------

def test_mma_optimizer_accepts_min_length_scale():
    """MMAOptimizer with min_length_scale > 0 must run without error on a
    small cantilever and improve (or at least not regress) the fom.

    Same rationale as the OC acceptance test above: p is held fixed (no
    continuation) and beta is kept modest so the double filter's erosion
    transient at the very first iteration is recoverable within 5 steps.
    """
    volume_fraction = 0.4
    oracle, initial, objective = _cantilever_setup(
        shape=(6, 8), volume_fraction=volume_fraction
    )
    opt = MMAOptimizer(
        volume_fraction=volume_fraction,
        max_iter=5,
        filter_radius=1.0,
        min_length_scale=1.0,
        p_start=3.0,
        p_end=3.0,
        beta_start=1.0,
        beta_end=2.0,
        tol=1e-12,
    )
    result = opt.run(initial.copy(), oracle, objective)
    assert len(result.history) > 0
    assert result.history[-1] > result.history[0], (
        f"history should improve: {result.history}"
    )


# ---------------------------------------------------------------------------
# Test 6: volume fraction conserved with min_length_scale enabled
# ---------------------------------------------------------------------------

def test_min_length_scale_volume_conserved():
    """TopologyOptimizer with min_length_scale > 0 must still hold the
    target volume fraction within 0.05 after 10 iterations, via the
    VolumeConstraint projection."""
    from morphos.manufacturing.constraints import VolumeConstraint

    volume_fraction = 0.4
    shape = (8, 16)
    oracle, initial, objective = _cantilever_setup(
        shape=shape, volume_fraction=volume_fraction
    )
    constraint = VolumeConstraint(volume_fraction)
    opt = TopologyOptimizer(
        step_size=1e-2,
        max_iter=10,
        tol=1e-12,
        bounds=(1e-3, 1.0),
        filter_radius=1.5,
        min_length_scale=1.5,
        beta_start=2.0,
        beta_end=4.0,
        p_start=1.0,
        p_end=3.0,
        p_ramp_fraction=0.5,
    )
    result = opt.run(initial.copy(), oracle, objective, constraint=constraint)
    final_vf = float(np.mean(result.field.values))
    assert abs(final_vf - volume_fraction) < 0.05, (
        f"final volume fraction {final_vf:.4f} not within 0.05 of "
        f"{volume_fraction}"
    )
