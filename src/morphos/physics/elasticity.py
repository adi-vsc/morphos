"""A real PDE physics backend: linear elasticity for SIMP topology
optimization, 2D plane-stress (Q4) or 3D solid (Hex8) (the structural leg of
the classic structural/thermal/EM topology-optimization triad; this backend
completes it alongside :mod:`morphos.physics.heat` and the ``ceviche_em``
backend).

The design field ``rho`` (in ``[0, 1]``, one value per grid cell) is the SIMP
("solid isotropic material with penalization") density. Each cell is one
bilinear-quad (Q4) plane-stress finite element whose Young's modulus is
penalized by density:

    E(rho) = E_min + rho**p * (E0 - E_min)

``p`` (typically 3) pushes the optimizer toward 0/1 ("black and white")
designs by making intermediate densities structurally inefficient. ``E_min``
is a small floor (``1e-9 * E0`` by default) rather than zero, so that void
cells still contribute a (tiny) positive stiffness and the global matrix never
goes singular -- the standard fix from Andreassen et al. 2011's 88-line SIMP
topology optimization paper, which this backend follows for the FEM
formulation and the penalization, deriving the element stiffness matrix from
quadrature (:func:`morphos.physics.operators.q4_plane_stress_stiffness`)
rather than transcribing it.

The static equilibrium problem is

    K(rho) @ u = F

with ``K`` assembled from the per-element Q4 stiffness scaled by ``E(rho_e)``,
Dirichlet boundary conditions fixing a subset of nodal degrees of freedom
(supports), and ``F`` a sparse nodal force vector (point or distributed
loads). The figure of merit is compliance ``C = F^T u = u^T K u``, the
standard scalar measure of structural flexibility; SIMP topology optimization
minimizes it (a stiffer structure for the same material budget).

Self-adjoint gradient. Unlike :mod:`morphos.physics.heat`, which needs a
separate co-state (adjoint) solve because its quantity of interest is a
mismatch against a target rather than the energy of the governing equation
itself, compliance minimization in SIMP is the textbook *self-adjoint* case:
because ``F`` does not depend on ``rho`` and ``K`` is symmetric (derivation:
``K u = F`` => ``K du/drho + dK/drho u = 0`` => ``du/drho = -K^-1 dK/drho u``;
``dC/drho = F^T du/drho = u^T K du/drho = -u^T dK/drho u`` since ``F = K u``):

    dC/drho_e = -p * rho_e**(p-1) * (E0 - E_min) * u_e^T @ k0 @ u_e

and, since ``value = -C``,

    d(value)/drho_e = +p * rho_e**(p-1) * (E0 - E_min) * u_e^T @ k0 @ u_e

where ``u_e`` is the element's local displacement vector (extracted from the
global solution) and ``k0`` is the unit-density (``E=E0-E_min`` scale folded
out) Q4 stiffness matrix. No extra linear solve is needed -- the same
displacement solve that gives the compliance also gives every element's
sensitivity. This is, however, exactly the kind of textbook formula the
project's own discipline distrusts on theory alone: it is gated against a
central finite difference of the *actual* assembled, BC-reduced ``K`` in
``tests/test_elasticity.py`` rather than assumed correct because it matches a
paper.

Sign convention. Like ``HeatConductionOracle``, the optimizer in this engine
performs gradient *ascent*. Compliance minimization is therefore expressed as
maximizing ``value = -compliance``, exactly mirroring
``HeatConductionOracle.value = -weighted_squared_error``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Sequence, Tuple

import numpy as np
from scipy import sparse

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.operators import hex8_stiffness, q4_plane_stress_stiffness
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_AXES = {"x": 0, "y": 1, "z": 2}

# (dx, dy[, dz]) offsets of each element's local nodes, in the same natural-
# coordinate corner order as q4_plane_stress_stiffness / hex8_stiffness.
_LOCAL_NODE_OFFSETS = {
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}


class ElasticityOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        fixed_dofs: Iterable[Tuple[int, ...]],
        loads: Dict[Tuple[int, ...], float],
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        penalty: float = 3.0,
        e_min_fraction: float = 1e-9,
        solver: Literal["direct", "iterative", "auto"] = "auto",
    ) -> None:
        """SIMP compliance oracle: 2D plane-stress (Q4) or 3D solid (Hex8).

        Parameters
        ----------
        shape:
            Number of *elements* per axis: ``(ny, nx)`` for 2D plane-stress or
            ``(nz, ny, nx)`` for 3D solid (the design field has this shape;
            the node grid is one larger on every axis).
        fixed_dofs:
            Iterable of ``(node_x, node_y, axis)`` triples (2D) or
            ``(node_x, node_y, node_z, axis)`` quadruples (3D), ``axis`` in
            ``{"x", "y"}`` / ``{"x", "y", "z"}``, naming nodal degrees of
            freedom held at zero displacement (the support boundary
            condition). Node indices are 0-based.
        loads:
            Mapping of the same coordinate tuples to force, for every loaded
            degree of freedom; unlisted DOFs carry zero force.
        young_modulus, poisson_ratio:
            Solid-material (``rho = 1``) elastic properties, ``E0`` below.
        penalty:
            SIMP exponent ``p`` (typically 3).
        e_min_fraction:
            Void-region stiffness floor as a fraction of ``E0``, avoiding a
            singular global stiffness matrix when a cell's density is zero.
        solver:
            ``"direct"`` (sparse LU), ``"iterative"`` (AMG-preconditioned CG;
            ``Kff`` is symmetric positive-definite), or ``"auto"`` (direct
            below, iterative at/above,
            :data:`morphos.physics._linsolve.AUTO_ITERATIVE_THRESHOLD` free
            dofs). The AMG hierarchy is cached and reused across solves as
            long as the free-dof sparsity pattern (fixed dofs, grid shape)
            does not change.
        """
        if len(shape) not in (2, 3):
            raise ValueError(
                "ElasticityOracle needs a 2D (ny, nx) or 3D (nz, ny, nx) grid of elements"
            )
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        self.ndim = len(shape)
        self.shape = tuple(int(s) for s in shape)
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.penalty = float(penalty)
        self.e_min = float(e_min_fraction) * self.young_modulus
        self.solver = solver
        self._amg_cache: list = []

        # Node-grid extent in the same axis order as `shape` (..., y, x).
        self._nn = tuple(s + 1 for s in self.shape)
        self._n_dof = self.ndim * int(np.prod(self._nn))

        self._fixed = sorted({self._dof_index(*t) for t in fixed_dofs})
        if not self._fixed:
            raise ValueError("fixed_dofs must constrain at least one degree of freedom")
        all_dofs = np.arange(self._n_dof)
        fixed_mask = np.zeros(self._n_dof, dtype=bool)
        fixed_mask[self._fixed] = True
        self._free = all_dofs[~fixed_mask]

        F = np.zeros(self._n_dof)
        for t, force in loads.items():
            F[self._dof_index(*t)] += float(force)
        self._F = F
        if not np.any(F[self._free] != 0.0):
            raise ValueError("loads must apply nonzero force on at least one free dof")

        self._k0 = None  # unit-density (E0 - E_min) element stiffness, set per spacing
        self._h = None
        self._elem_dof = self._element_dof_table()

    def _dof_index(self, *args) -> int:
        *coords, axis = args
        if len(coords) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} coordinate(s) plus axis, got {args!r}"
            )
        if axis not in _AXES or _AXES[axis] >= self.ndim:
            raise ValueError(f"axis must be one of {list(_AXES)[: self.ndim]}, got {axis!r}")
        # coords are given fastest-to-slowest (x, y[, z]); node-grid axes are
        # slowest-to-first (..., y, x), matching `shape`.
        coords_axis_order = tuple(reversed(coords))
        for c, n in zip(coords_axis_order, self._nn):
            if not (0 <= c < n):
                raise ValueError(f"node {coords} out of range for node grid {self._nn}")
        node = int(np.ravel_multi_index(coords_axis_order, self._nn))
        return self.ndim * node + _AXES[axis]

    def _element_dof_table(self) -> np.ndarray:
        """Row e -> the per-element global dof indices, in local node order."""
        n_elem = int(np.prod(self.shape))
        n_local_dof = self.ndim * (2 ** self.ndim)
        table = np.zeros((n_elem, n_local_dof), dtype=int)
        offsets = _LOCAL_NODE_OFFSETS[self.ndim]
        for idx in np.ndindex(*self.shape):  # idx in axis order (..., ey, ex)
            e = int(np.ravel_multi_index(idx, self.shape))
            dofs = []
            for off in offsets:  # off given fastest-to-slowest (dx, dy[, dz])
                off_axis_order = tuple(reversed(off))
                node_coord = tuple(i + o for i, o in zip(idx, off_axis_order))
                node = int(np.ravel_multi_index(node_coord, self._nn))
                dofs.extend(self.ndim * node + k for k in range(self.ndim))
            table[e] = dofs
        return table

    def _element_stiffness(self, e_modulus: float, h: float) -> np.ndarray:
        if self.ndim == 2:
            return q4_plane_stress_stiffness(e_modulus, self.poisson_ratio, h)
        return hex8_stiffness(e_modulus, self.poisson_ratio, h)

    def _unit_stiffness(self, h: float) -> np.ndarray:
        if self._k0 is None or self._h != h:
            # k0 is the stiffness at unit (E0 - E_min) modulus; SIMP scales it
            # per element by (E_min + rho**p * (E0 - E_min)) / (E0 - E_min)... but
            # it is simpler and exactly equivalent to assemble at full E0 and
            # add the void floor as a separate uniform term. We instead build
            # k0 directly at modulus (E0 - E_min) so K_e = E_min*k_floor +
            # rho_e**p * k0; k_floor is k0 scaled by E_min/(E0-E_min) when E0
            # != E_min, assembled below from the same shape function call.
            self._k0 = self._element_stiffness(self.young_modulus - self.e_min, h)
            self._k_floor = self._element_stiffness(self.e_min, h)
            self._h = h
        return self._k0

    def _global_stiffness(self, rho_flat: np.ndarray, h: float):
        k0 = self._unit_stiffness(h)
        k_floor = self._k_floor
        n_elem = rho_flat.size
        n_local_dof = self._elem_dof.shape[1]
        scale = rho_flat ** self.penalty  # (n_elem,)

        rows = np.repeat(self._elem_dof, n_local_dof, axis=1).reshape(
            n_elem, n_local_dof, n_local_dof
        )
        cols = np.tile(self._elem_dof, (1, n_local_dof)).reshape(
            n_elem, n_local_dof, n_local_dof
        )
        # K_e = k_floor + rho_e**p * k0  (k_floor independent of rho: the void
        # stiffness floor that keeps K nonsingular everywhere).
        vals = k_floor[None, :, :] + scale[:, None, None] * k0[None, :, :]

        K = sparse.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof, self._n_dof),
        ).tocsr()
        return K

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (ny, nx elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("ElasticityOracle assumes isotropic spacing")
        h = spacing[0]

        rho_flat = rho.ravel()  # row-major: element e = ey*nx + ex, matches _element_dof_table
        K = self._global_stiffness(rho_flat, h)

        free = self._free
        Kff = K[np.ix_(free, free)]
        Ff = self._F[free]

        uf, residual_norm, iterations = solve_linear(
            Kff, Ff, solver=self.solver, dof_count=free.size,
            symmetric=True, cache_holder=self._amg_cache,
        )
        u = np.zeros(self._n_dof)
        u[free] = uf

        compliance = float(self._F @ u)
        value = -compliance

        # Self-adjoint SIMP sensitivity: per element, -p*rho^(p-1)*(E0-Emin)*u_e^T k0 u_e.
        k0 = self._k0
        u_e = u[self._elem_dof]  # (n_elem, 8)
        energy = np.einsum("ei,ij,ej->e", u_e, k0, u_e)
        # dC/drho_e = -dE/drho_e * u_e^T k0 u_e (compliance sensitivity); since
        # value = -C, d(value)/drho_e = +dE/drho_e * u_e^T k0 u_e.
        dE_drho = self.penalty * rho_flat ** (self.penalty - 1.0)
        grad_flat = dE_drho * energy
        gradient = grad_flat.reshape(self.shape)

        displacement = u.reshape(*self._nn, self.ndim)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "compliance": compliance,
                "displacement": displacement,
            },
            residual_norm=residual_norm,
            solver_iterations=iterations,
        )
