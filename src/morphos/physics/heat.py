"""A real PDE physics backend: steady-state heat conduction.

This backend exists to prove the architecture: a genuine partial differential
equation solver, with an exact adjoint gradient, slots in behind the same
:class:`~morphos.physics.oracle.PhysicsOracle` interface as the analytic toy,
and the same engine and optimizer drive it unchanged.

It solves the steady heat equation (2D or 3D) on the grid with the design field
acting as a distributed heat source and the boundary held at zero temperature:

    -k laplacian(T) = s   inside,    T = 0   on the boundary.

The figure of merit is the negative weighted squared error of the temperature
against a target field. The gradient with respect to the source is computed by
the adjoint method (one extra linear solve), which is what makes inverse design
scale to many design variables.

The discrete operator is the shared symmetric positive-definite interior
Laplacian from :mod:`morphos.physics.operators`. Because it is SPD, the linear
solve can run either by a direct sparse factorization (``solver="direct"``,
exact, best for small/moderate grids) or by a preconditioned conjugate
gradient: ``solver="cg"`` with an explicit ``preconditioner`` choice
(``"jacobi"`` or ``"amg"``, legacy, hand-rolled), or the newer shared
``solver="iterative"``/``"auto"`` path (:mod:`morphos.physics._linsolve`),
which always uses SA-AMG-preconditioned CG and caches the AMG hierarchy across
solves on this oracle as long as the grid (hence the matrix sparsity pattern)
does not change. ``"auto"`` resolves to direct below
:data:`morphos.physics._linsolve.AUTO_ITERATIVE_THRESHOLD` interior DOFs and to
iterative at or above it. All paths produce the same answer to tolerance.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.operators import (
    hex8_diffusion_stiffness_aniso,
    interior_laplacian,
    q4_diffusion_stiffness_aniso,
)
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_SOLVERS = ("direct", "cg", "iterative", "auto")
_PRECONDITIONERS = ("jacobi", "amg")

# (dx, dy[, dz]) offsets of each element's local nodes, in the same natural-
# coordinate corner order as q4_diffusion_stiffness_aniso / hex8_diffusion_stiffness_aniso
# (and matching morphos.physics.elasticity._LOCAL_NODE_OFFSETS).
_LOCAL_NODE_OFFSETS = {
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}


class HeatConductionOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        target: np.ndarray,
        weight: float = 1.0,
        solver: str = "direct",
        preconditioner: str = "jacobi",
        cg_rtol: float = 1e-10,
        cg_maxiter: int | None = None,
    ) -> None:
        target = np.asarray(target, dtype=float)
        if target.ndim < 2:
            raise ValueError("HeatConductionOracle needs a 2D or 3D grid")
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
        if preconditioner not in _PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {_PRECONDITIONERS}, got {preconditioner!r}"
            )
        self.target = target
        self.weight = float(weight)
        self.solver = solver
        self.preconditioner = preconditioner
        self.cg_rtol = float(cg_rtol)
        self.cg_maxiter = cg_maxiter
        self.shape = target.shape
        self._interior = ~self._boundary_mask(self.shape)
        self._A = None
        self._A_spacing = None
        self._precond = None
        # Cache holder for the shared solve_linear() AMG hierarchy (used by
        # the "iterative"/"auto" modes only); a one-element list so
        # _linsolve._AMGCache can be mutated in place across calls.
        self._amg_cache: list = []

    @staticmethod
    def _boundary_mask(shape) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        for axis in range(len(shape)):
            lo = [slice(None)] * len(shape)
            hi = [slice(None)] * len(shape)
            lo[axis] = 0
            hi[axis] = -1
            m[tuple(lo)] = True
            m[tuple(hi)] = True
        return m

    def _matrix(self, field: Field) -> sparse.csr_matrix:
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("HeatConductionOracle assumes isotropic spacing")
        h = spacing[0]
        if self._A is None or self._A_spacing != h:
            self._A = interior_laplacian(self.shape, h)
            self._A_spacing = h
            self._precond = self._build_preconditioner(self._A)
        return self._A

    def _build_preconditioner(self, A: sparse.csr_matrix):
        if self.preconditioner == "jacobi":
            # Diagonal scaling: cheap, but iteration count grows with grid size.
            return sparse.diags(1.0 / A.diagonal())
        # Algebraic multigrid: near grid-independent iteration count, which is
        # what makes CG actually beat direct LU at scale. Imported lazily so the
        # dependency is only required when this path is used.
        import pyamg

        return pyamg.smoothed_aggregation_solver(A.tocsr()).aspreconditioner()

    def _solve_linear(self, A: sparse.csr_matrix, b: np.ndarray):
        """Return ``(x, residual_norm, iterations)`` for the configured solver.

        ``"direct"``/``"cg"`` are the original two paths (the latter with an
        explicit jacobi/amg preconditioner choice); ``"iterative"``/``"auto"``
        delegate to the shared, AMG-cached :func:`solve_linear` helper used by
        every oracle.
        """
        if self.solver in ("iterative", "auto"):
            return solve_linear(
                A,
                b,
                solver=self.solver,
                dof_count=b.size,
                symmetric=True,
                cache_holder=self._amg_cache,
                rtol=self.cg_rtol,
                maxiter=self.cg_maxiter,
            )
        if self.solver == "direct":
            return spsolve(A, b), 0.0, 0
        # "cg": A is symmetric positive-definite, so CG is valid and self-adjoint.
        x, info = cg(A, b, rtol=self.cg_rtol, atol=0.0, maxiter=self.cg_maxiter, M=self._precond)
        if info != 0:
            raise RuntimeError(f"CG failed to converge (info={info})")
        return x, float(np.linalg.norm(A @ x - b)), 0

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match target "
                f"shape {self.shape}"
            )
        A = self._matrix(field)
        interior = self._interior.ravel()

        # Source on the interior unknowns; the boundary is Dirichlet zero.
        s_int = field.values.ravel()[interior]

        T_int, res_fwd, iters_fwd = self._solve_linear(A, s_int)
        T_flat = np.zeros(field.values.size)
        T_flat[interior] = T_int
        T = T_flat.reshape(self.shape)

        diff = T - self.target
        value = -float(self.weight * np.sum(diff ** 2))

        # Adjoint: A is self-adjoint, so the co-state solves the same operator.
        # Boundary temperature is fixed, so its objective terms stay out of the
        # gradient (they live only on boundary DOFs, which we never solve for).
        dJdT_int = (-2.0 * self.weight * diff).ravel()[interior]
        lam_int, res_adj, iters_adj = self._solve_linear(A, dJdT_int)
        grad_flat = np.zeros(field.values.size)
        grad_flat[interior] = lam_int
        gradient = grad_flat.reshape(self.shape)

        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={"temperature": T},
            residual_norm=max(res_fwd, res_adj),
            solver_iterations=max(iters_fwd, iters_adj),
        )

    def residual(self, source: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Return the residual of the linear system, near zero for a valid solve.

        Interior DOFs report ``A @ T_int - s_int``; boundary DOFs report the
        deviation of the temperature from the Dirichlet zero condition.
        """
        if self._A is None:
            raise RuntimeError("call solve() before residual() to assemble the matrix")
        interior = self._interior.ravel()
        s_int = np.asarray(source, dtype=float).ravel()[interior]
        T = np.asarray(temperature, dtype=float).ravel()
        r = np.zeros(T.size)
        r[interior] = self._A @ T[interior] - s_int
        r[~interior] = T[~interior]
        return r.reshape(self.shape)


class FEMHeatOracle(PhysicsOracle):
    """SIMP topology-optimization oracle for steady heat conduction.

    Unlike :class:`HeatConductionOracle`, which discretizes the Laplacian by
    finite differences on the design field directly, this oracle is the
    thermal analogue of :class:`~morphos.physics.elasticity.ElasticityOracle`:
    the design field ``rho`` (in ``[0, 1]``, one value per grid cell) is a
    SIMP density that penalizes each cell's conductivity,

        k(rho) = k_min + rho**p * (k0 - k_min),

    assembled as a per-element Q4 (2D) / Hex8 (3D) diffusion stiffness
    (:func:`~morphos.physics.operators.q4_diffusion_stiffness_aniso` /
    :func:`~morphos.physics.operators.hex8_diffusion_stiffness_aniso`) into a
    global sparse conduction matrix, with arbitrary Dirichlet nodes (not just
    the full boundary) and anisotropic grid spacing. The PDE is

        -div(k(rho) grad(T)) = s,    T = 0 on the fixed (Dirichlet) nodes,

    with ``s`` a fixed (rho-independent) source distributed uniformly over
    elements -- a population of "heat sinks must absorb heat sourced
    everywhere" problems, e.g. heat-sink topology optimization.

    The figure of merit is heat compliance ``C = s^T T = T^T K T``, the
    thermal analogue of mechanical compliance, minimized by SIMP exactly as in
    :class:`~morphos.physics.elasticity.ElasticityOracle`: because ``s`` does
    not depend on ``rho`` and ``K`` is symmetric, the problem is
    self-adjoint --

        dC/drho_e = -p * rho_e**(p-1) * (k0 - k_min) * T_e^T @ k0_e @ T_e

    and, since ``value = -C``,

        d(value)/drho_e = +p * rho_e**(p-1) * (k0 - k_min) * T_e^T @ k0_e @ T_e

    where ``T_e`` is the element's local nodal-temperature vector and ``k0_e``
    is the unit-conductivity (``k0 - k_min`` scale folded out) diffusion
    element matrix. No extra linear solve is needed.
    """

    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        fixed_nodes: Iterable[Tuple[int, ...]] = None,
        conductivity: float = 1.0,
        penalty: float = 3.0,
        k_min_fraction: float = 1e-9,
        source=1.0,
        solver: str = "auto",
    ) -> None:
        """SIMP heat-compliance oracle: 2D (Q4) or 3D (Hex8) diffusion.

        Parameters
        ----------
        shape:
            Number of *elements* per axis: ``(ny, nx)`` for 2D or
            ``(nz, ny, nx)`` for 3D (the design field has this shape; the
            node grid is one larger on every axis).
        fixed_nodes:
            Iterable of ``(node_x, node_y)`` pairs (2D) or
            ``(node_x, node_y, node_z)`` triples (3D) naming nodes held at
            ``T = 0`` (the Dirichlet boundary condition). Node indices are
            0-based. Defaults to every node on the grid boundary (the same
            condition :class:`HeatConductionOracle` imposes implicitly) when
            not given.
        conductivity:
            Solid-material (``rho = 1``) conductivity, ``k0`` below.
        penalty:
            SIMP exponent ``p`` (typically 3).
        k_min_fraction:
            Void-region conductivity floor as a fraction of ``k0``, avoiding
            a singular global matrix when a cell's density is zero.
        source:
            Uniform heat source per element (a scalar) or an array matching
            ``shape`` giving a per-element source.
        solver:
            ``"direct"``, ``"iterative"``, or ``"auto"`` -- see
            :class:`~morphos.physics.elasticity.ElasticityOracle`.
        """
        if len(shape) not in (2, 3):
            raise ValueError("FEMHeatOracle needs a 2D (ny, nx) or 3D (nz, ny, nx) grid")
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        if solver not in ("direct", "iterative", "auto"):
            raise ValueError(f"solver must be one of direct/iterative/auto, got {solver!r}")
        self.ndim = len(shape)
        self.shape = tuple(int(s) for s in shape)
        self.conductivity = float(conductivity)
        self.penalty = float(penalty)
        self.k_min = float(k_min_fraction) * self.conductivity
        self.solver = solver
        self._amg_cache: list = []

        # Node-grid extent in the same axis order as `shape` (..., y, x).
        self._nn = tuple(s + 1 for s in self.shape)
        self._n_dof = int(np.prod(self._nn))  # one scalar dof (temperature) per node

        if fixed_nodes is None:
            fixed_nodes = self._default_boundary_nodes()
        self._fixed = sorted({self._dof_index(t) for t in fixed_nodes})
        if not self._fixed:
            raise ValueError("fixed_nodes must constrain at least one node")
        all_dofs = np.arange(self._n_dof)
        fixed_mask = np.zeros(self._n_dof, dtype=bool)
        fixed_mask[self._fixed] = True
        self._free = all_dofs[~fixed_mask]

        n_elem = int(np.prod(self.shape))
        if np.isscalar(source):
            self._source = np.full(self.shape, float(source))
        else:
            source = np.asarray(source, dtype=float)
            if source.shape != self.shape:
                raise ValueError(f"source shape {source.shape} does not match {self.shape}")
            self._source = source

        self._k0 = None  # unit-conductivity (k0 - k_min) element matrix, set per spacing
        self._k_floor = None
        self._spacing = None
        self._elem_dof = self._element_dof_table()

    def _default_boundary_nodes(self):
        """Every node on the grid boundary, the implicit condition that
        :class:`HeatConductionOracle` imposes via its finite-difference
        interior/boundary split."""
        for coords in np.ndindex(*reversed(self._nn)):  # coords already (x, y[, z])
            node_axis_order = tuple(reversed(coords))
            if any(c == 0 or c == n - 1 for c, n in zip(node_axis_order, self._nn)):
                yield coords

    def _dof_index(self, coords: Tuple[int, ...]) -> int:
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate(s), got {coords!r}")
        # coords given fastest-to-slowest (x, y[, z]); node-grid axes are
        # slowest-to-first (..., y, x), matching `shape`.
        coords_axis_order = tuple(reversed(coords))
        for c, n in zip(coords_axis_order, self._nn):
            if not (0 <= c < n):
                raise ValueError(f"node {coords} out of range for node grid {self._nn}")
        return int(np.ravel_multi_index(coords_axis_order, self._nn))

    def _element_dof_table(self) -> np.ndarray:
        """Row e -> the per-element global node (= scalar dof) indices, in
        local node order."""
        n_elem = int(np.prod(self.shape))
        n_local = 2 ** self.ndim
        table = np.zeros((n_elem, n_local), dtype=int)
        offsets = _LOCAL_NODE_OFFSETS[self.ndim]
        for idx in np.ndindex(*self.shape):  # idx in axis order (..., ey, ex)
            e = int(np.ravel_multi_index(idx, self.shape))
            nodes = []
            for off in offsets:  # off given fastest-to-slowest (dx, dy[, dz])
                off_axis_order = tuple(reversed(off))
                node_coord = tuple(i + o for i, o in zip(idx, off_axis_order))
                nodes.append(int(np.ravel_multi_index(node_coord, self._nn)))
            table[e] = nodes
        return table

    def _element_diffusion(self, k: float, spacing: Tuple[float, ...]) -> np.ndarray:
        if self.ndim == 2:
            hy, hx = spacing
            return q4_diffusion_stiffness_aniso(k, hx, hy)
        hz, hy, hx = spacing
        return hex8_diffusion_stiffness_aniso(k, hx, hy, hz)

    def _unit_diffusion(self, spacing: Tuple[float, ...]) -> np.ndarray:
        if self._k0 is None or self._spacing != spacing:
            self._k0 = self._element_diffusion(self.conductivity - self.k_min, spacing)
            self._k_floor = self._element_diffusion(self.k_min, spacing)
            self._spacing = spacing
        return self._k0

    def _global_matrix(self, rho_flat: np.ndarray, spacing: Tuple[float, ...]):
        k0 = self._unit_diffusion(spacing)
        k_floor = self._k_floor
        n_elem = rho_flat.size
        n_local = self._elem_dof.shape[1]
        scale = rho_flat ** self.penalty

        rows = np.repeat(self._elem_dof, n_local, axis=1).reshape(n_elem, n_local, n_local)
        cols = np.tile(self._elem_dof, (1, n_local)).reshape(n_elem, n_local, n_local)
        # K_e = k_floor + rho_e**p * k0 (k_floor independent of rho: the void
        # conductivity floor that keeps K nonsingular everywhere).
        vals = k_floor[None, :, :] + scale[:, None, None] * k0[None, :, :]

        K = sparse.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof, self._n_dof),
        ).tocsr()
        return K

    def _source_vector(self, field: Field) -> np.ndarray:
        """Distribute each element's uniform source equally to its nodes."""
        n_local = self._elem_dof.shape[1]
        elem_volume = field.voxel_volume
        contrib = self._source.ravel() * elem_volume / n_local
        F = np.zeros(self._n_dof)
        np.add.at(F, self._elem_dof.ravel(), np.repeat(contrib, n_local))
        return F

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (ny, nx elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing

        rho_flat = rho.ravel()  # row-major: element e = ey*nx + ex, matches _element_dof_table
        K = self._global_matrix(rho_flat, spacing)
        F = self._source_vector(field)

        free = self._free
        Kff = K[np.ix_(free, free)]
        Ff = F[free]

        Tf, residual_norm, iterations = solve_linear(
            Kff, Ff, solver=self.solver, dof_count=free.size,
            symmetric=True, cache_holder=self._amg_cache,
        )
        T = np.zeros(self._n_dof)
        T[free] = Tf

        compliance = float(F @ T)
        value = -compliance

        # Self-adjoint SIMP sensitivity: per element,
        # -p*rho^(p-1)*(k0-k_min)*T_e^T k0 T_e (heat-compliance sensitivity);
        # since value = -C, d(value)/drho_e = +that, sign flipped.
        k0 = self._k0
        T_e = T[self._elem_dof]  # (n_elem, n_local)
        energy = np.einsum("ei,ij,ej->e", T_e, k0, T_e)
        dE_drho = self.penalty * rho_flat ** (self.penalty - 1.0)
        grad_flat = dE_drho * energy
        gradient = grad_flat.reshape(self.shape)

        temperature = T.reshape(self._nn)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "compliance": compliance,
                "temperature": temperature,
            },
            residual_norm=residual_norm,
            solver_iterations=iterations,
        )
