"""Shared linear-solve strategy for the physics oracles: direct LU or
AMG-preconditioned Krylov, with preconditioner caching across calls.

Every oracle in this package reduces its physics to ``A x = b`` on the free /
interior degrees of freedom after Dirichlet elimination. Direct sparse LU
(``scipy.sparse.linalg.spsolve``) is exact and simple but its fill-in makes it
memory- and time-prohibitive once the interior DOF count gets into the tens of
thousands (3D grids get there fast). This module centralizes the alternative:
build a smoothed-aggregation AMG hierarchy once with :mod:`pyamg` and reuse it
as a preconditioner for CG (symmetric positive-definite systems) or GMRES
(non-symmetric / saddle-point systems), across as many solves as the matrix's
sparsity pattern stays the same -- which, for SIMP topology optimization, is
every iteration (only cell densities change, never the mesh connectivity).

``pyamg`` is optional: callers that request ``"iterative"`` or ``"auto"`` on a
large problem without it installed get a clear ImportError; ``"direct"``
never touches it.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, gmres, spilu, spsolve

#: DOF threshold for "auto": below this, direct LU is faster and exact;
#: at/above it, the AMG-preconditioned Krylov path is selected instead.
AUTO_ITERATIVE_THRESHOLD = 20000

SolverMode = Literal["direct", "iterative", "auto"]
_MODES = ("direct", "iterative", "auto")


def resolve_mode(solver: str, dof_count: int) -> str:
    """Turn a requested solver mode into a concrete ``"direct"``/``"iterative"``
    choice.

    Parameters
    ----------
    solver:
        One of ``"direct"``, ``"iterative"``, ``"auto"``.
    dof_count:
        Number of unknowns in the linear system (free / interior DOFs).
    """
    if solver not in _MODES:
        raise ValueError(f"solver must be one of {_MODES}, got {solver!r}")
    if solver == "auto":
        return "iterative" if dof_count >= AUTO_ITERATIVE_THRESHOLD else "direct"
    return solver


class _AMGCache:
    """Holds one cached AMG hierarchy, keyed on the matrix sparsity pattern.

    Rebuilding a smoothed-aggregation hierarchy costs roughly as much as a
    handful of CG iterations, so it is only worth paying once per distinct
    sparsity pattern. SIMP topology optimization changes element *values*
    (densities) every iteration but never the mesh connectivity, so the
    pattern (``indptr``/``indices``) is stable across an entire optimization
    run; this cache rebuilds only when that pattern changes (a new grid, a
    different boundary-condition set, etc.) and otherwise reuses the existing
    hierarchy even though the underlying values drifted. This is a pragmatic
    approximation -- an AMG hierarchy built on stale values is still a valid,
    if slightly less sharp, preconditioner for CG/GMRES, which only need an
    approximate inverse to converge; correctness of the final answer is
    governed by the Krylov residual tolerance, not by the preconditioner.

    ``rebuild_every`` forces a periodic rebuild so that, during SIMP
    continuation, the hierarchy does not get arbitrarily stale as densities
    evolve from near-uniform gray to near-binary (9 decades of stiffness
    contrast).  A value of 10 means the hierarchy is rebuilt at most once
    every 10 forward solves, which is cheap relative to the CG convergence
    improvement it buys.
    """

    __slots__ = ("indptr", "indices", "shape", "ml", "preconditioner", "_call_count", "rebuild_every")

    def __init__(self, rebuild_every: int = 10) -> None:
        self.indptr = None
        self.indices = None
        self.shape = None
        self.ml = None
        self.preconditioner = None
        self._call_count = 0
        self.rebuild_every = int(rebuild_every)

    def get(self, A: sparse.csr_matrix):
        """Return a cached preconditioner for ``A``, rebuilding if the
        sparsity pattern changed or the periodic rebuild interval elapsed."""
        import pyamg

        A = A.tocsr()
        self._call_count += 1
        pattern_changed = (
            self.indptr is None
            or self.shape != A.shape
            or self.indptr.shape != A.indptr.shape
            or self.indices.shape != A.indices.shape
            or not np.array_equal(self.indptr, A.indptr)
            or not np.array_equal(self.indices, A.indices)
            or (self._call_count > 1 and self._call_count % self.rebuild_every == 0)
        )
        if pattern_changed:
            self.ml = pyamg.smoothed_aggregation_solver(A)
            self.preconditioner = self.ml.aspreconditioner()
            self.indptr = A.indptr.copy()
            self.indices = A.indices.copy()
            self.shape = A.shape
        return self.preconditioner


def solve_linear(
    A: sparse.spmatrix,
    b: np.ndarray,
    solver: SolverMode,
    dof_count: Optional[int] = None,
    symmetric: bool = True,
    cache_holder: Optional[list] = None,
    rtol: float = 1e-10,
    maxiter: Optional[int] = None,
    indefinite: bool = False,
    rebuild_amg_every: int = 10,
) -> Tuple[np.ndarray, float, int]:
    """Solve ``A x = b`` by direct LU or preconditioned Krylov.

    Parameters
    ----------
    A:
        The (square) system matrix.
    b:
        Right-hand side.
    solver:
        ``"direct"`` (spsolve), ``"iterative"`` (preconditioned CG/GMRES), or
        ``"auto"`` (direct below :data:`AUTO_ITERATIVE_THRESHOLD` DOFs,
        iterative at or above it).
    dof_count:
        Number of unknowns, used only to resolve ``"auto"``; defaults to
        ``b.size``.
    symmetric:
        Whether ``A`` is symmetric positive-definite (CG) or not (GMRES,
        e.g. advection-coupled conjugate heat). Ignored when ``indefinite``
        is set (GMRES always applies there).
    indefinite:
        Set for saddle-point / indefinite systems (e.g. a mixed
        velocity-pressure Stokes block). Smoothed-aggregation AMG assumes an
        M-matrix-like, diagonally dominant SPD structure and empirically
        fails to converge on the raw saddle-point operator (verified: it
        hits the iteration cap without reducing the residual). For this case
        the iterative path uses an incomplete-LU-preconditioned GMRES
        instead of SA-AMG, which is robust on indefinite systems and was
        checked to match the direct solve to ~1e-13 relative error on a
        representative Stokes problem.
    cache_holder:
        A single-element mutable container (e.g. a list) used to persist the
        preconditioner (AMG hierarchy or ILU factorization) across calls on
        the same oracle instance. Pass the same list object on every call
        from a given oracle. If ``None``, no caching happens (a fresh
        preconditioner is built every call).
    rtol, maxiter:
        Convergence tolerance and iteration cap forwarded to CG/GMRES.
    rebuild_amg_every:
        How many forward solves between forced AMG hierarchy rebuilds (passed
        to :class:`_AMGCache` and :class:`_ILUCache`). Default is 10.

    Returns
    -------
    (x, residual_norm, iterations):
        The solution, ``||A x - b||`` (0.0 for the direct path, exact to
        round-off), and the iteration count (0 for the direct path).
    """
    if dof_count is None:
        dof_count = b.size
    mode = resolve_mode(solver, dof_count)

    if mode == "direct":
        x = spsolve(A.tocsc(), b)
        return x, 0.0, 0

    A = A.tocsr()
    iters = 0

    def _count(_):
        nonlocal iters
        iters += 1

    if indefinite:
        # SA-AMG is not a valid preconditioner for an indefinite saddle-point
        # operator; use ILU + GMRES instead (see docstring).
        M = _get_ilu_preconditioner(A, cache_holder, rebuild_amg_every)
        x, info = gmres(
            A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M,
            callback=_count, callback_type="legacy",
        )
        if info != 0:
            raise RuntimeError(f"GMRES (ILU) failed to converge (info={info})")
    elif symmetric:
        M = _get_amg_preconditioner(A, cache_holder, rebuild_amg_every)
        x, info = cg(A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=_count)
        if info != 0:
            raise RuntimeError(f"CG failed to converge (info={info})")
    else:
        M = _get_amg_preconditioner(A, cache_holder, rebuild_amg_every)
        x, info = gmres(
            A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M,
            callback=_count, callback_type="legacy",
        )
        if info != 0:
            raise RuntimeError(f"GMRES failed to converge (info={info})")

    residual_norm = float(np.linalg.norm(A @ x - b))
    return x, residual_norm, iters


def _get_amg_preconditioner(
    A: sparse.csr_matrix,
    cache_holder: Optional[list],
    rebuild_every: int = 10,
):
    if cache_holder is not None:
        if not cache_holder:
            cache_holder.append(_AMGCache(rebuild_every=rebuild_every))
        return cache_holder[0].get(A)
    import pyamg

    return pyamg.smoothed_aggregation_solver(A).aspreconditioner()


def _get_ilu_preconditioner(
    A: sparse.csr_matrix,
    cache_holder: Optional[list],
    rebuild_every: int = 10,
):
    if cache_holder is not None:
        if not cache_holder:
            cache_holder.append(_ILUCache(rebuild_every=rebuild_every))
        return cache_holder[0].get(A)
    return _build_ilu(A)


def _build_ilu(A: sparse.csr_matrix) -> LinearOperator:
    ilu = spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20)
    return LinearOperator(A.shape, ilu.solve)


class _ILUCache:
    """Like :class:`_AMGCache` but for the ILU preconditioner used on
    indefinite (saddle-point) systems: rebuilds only when the sparsity
    pattern changes or the periodic rebuild interval elapses."""

    __slots__ = ("indptr", "indices", "shape", "preconditioner", "_call_count", "rebuild_every")

    def __init__(self, rebuild_every: int = 10) -> None:
        self.indptr = None
        self.indices = None
        self.shape = None
        self.preconditioner = None
        self._call_count = 0
        self.rebuild_every = int(rebuild_every)

    def get(self, A: sparse.csr_matrix) -> LinearOperator:
        A = A.tocsr()
        self._call_count += 1
        pattern_changed = (
            self.indptr is None
            or self.shape != A.shape
            or self.indptr.shape != A.indptr.shape
            or self.indices.shape != A.indices.shape
            or not np.array_equal(self.indptr, A.indptr)
            or not np.array_equal(self.indices, A.indices)
            or (self._call_count > 1 and self._call_count % self.rebuild_every == 0)
        )
        if pattern_changed:
            self.preconditioner = _build_ilu(A)
            self.indptr = A.indptr.copy()
            self.indices = A.indices.copy()
            self.shape = A.shape
        return self.preconditioner
