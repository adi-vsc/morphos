"""Tests for morphos.physics._linsolve: AMG/ILU cache behaviour and solve accuracy."""

import numpy as np
import pytest
from scipy import sparse


def test_amg_rebuild_every_default_is_10():
    from morphos.physics._linsolve import _AMGCache
    assert _AMGCache().rebuild_every == 10


def test_ilu_rebuild_every_default_is_10():
    from morphos.physics._linsolve import _ILUCache
    assert _ILUCache().rebuild_every == 10


def test_solve_linear_direct_accurate():
    """Direct solver produces a correct solution."""
    from morphos.physics._linsolve import solve_linear

    n = 20
    diag = 2.0 * np.ones(n)
    A = sparse.diags([diag, -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]).tocsr()
    b = np.ones(n)

    x, res, iters = solve_linear(A, b, "direct")
    assert np.linalg.norm(A @ x - b) < 1e-10
    assert iters == 0


def test_solve_linear_different_matrices_both_accurate():
    """Build two different 4x4 SPD matrices, alternate solve_linear calls with
    rebuild_amg_every=2, assert residual < 1e-10 for both."""
    pyamg = pytest.importorskip("pyamg")  # skip if pyamg not installed

    from morphos.physics._linsolve import solve_linear

    n = 20
    # Matrix 1: tridiagonal with diagonal 2, off-diagonal -1
    diag = 2.0 * np.ones(n)
    A1 = sparse.diags([diag, -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]).tocsr() * 1.0
    b1 = np.ones(n)
    # Matrix 2: scaled version
    A2 = A1 * 3.0
    b2 = np.ones(n) * 2.0

    cache = [None]
    # Remove the sentinel so cache_holder starts empty
    cache = []

    x1, res1, _ = solve_linear(A1, b1, "iterative", cache_holder=cache, rebuild_amg_every=2)
    x2, res2, _ = solve_linear(A2, b2, "iterative", cache_holder=cache, rebuild_amg_every=2)
    x1b, res1b, _ = solve_linear(A1, b1, "iterative", cache_holder=cache, rebuild_amg_every=2)

    assert np.linalg.norm(A1 @ x1 - b1) < 1e-10
    assert np.linalg.norm(A2 @ x2 - b2) < 1e-10
    assert np.linalg.norm(A1 @ x1b - b1) < 1e-10


def test_amg_cache_rebuild_every_respected():
    """_AMGCache with rebuild_every=3 rebuilds on calls 1, 3, 6, ..."""
    pyamg = pytest.importorskip("pyamg")

    from morphos.physics._linsolve import _AMGCache

    n = 10
    diag = 2.0 * np.ones(n)
    A = sparse.diags([diag, -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]).tocsr()

    cache = _AMGCache(rebuild_every=3)
    pre1 = cache.get(A)  # call 1 — build
    pre2 = cache.get(A)  # call 2 — reuse
    pre3 = cache.get(A)  # call 3 — rebuild (3 % 3 == 0)

    # On call 1 and call 3 the preconditioner is rebuilt (new object).
    # On call 2 it is reused (same object).
    assert pre1 is pre2   # reused
    assert pre2 is not pre3  # rebuilt on call 3
