"""A real PDE physics backend: 2D linear elasticity for SIMP topology
optimization (the structural leg of the classic structural/thermal/EM
topology-optimization triad; this backend completes it alongside
:mod:`morphos.physics.heat` and the ``ceviche_em`` backend).

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

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from morphos.field import Field
from morphos.physics.operators import q4_plane_stress_stiffness
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_AXES = {"x": 0, "y": 1}


class ElasticityOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, int],
        fixed_dofs: Iterable[Tuple[int, int, str]],
        loads: Dict[Tuple[int, int, str], float],
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        penalty: float = 3.0,
        e_min_fraction: float = 1e-9,
    ) -> None:
        """Plane-stress SIMP compliance oracle.

        Parameters
        ----------
        shape:
            ``(ny, nx)`` number of *elements* (the design field has this
            shape; the node grid is ``(ny + 1, nx + 1)``).
        fixed_dofs:
            Iterable of ``(node_x, node_y, axis)`` triples, ``axis`` in
            ``{"x", "y"}``, naming nodal degrees of freedom held at zero
            displacement (the support boundary condition). Node indices are
            0-based, ``node_x`` in ``[0, nx]``, ``node_y`` in ``[0, ny]``.
        loads:
            Mapping ``(node_x, node_y, axis) -> force`` for every loaded
            degree of freedom; unlisted DOFs carry zero force.
        young_modulus, poisson_ratio:
            Solid-material (``rho = 1``) elastic properties, ``E0`` below.
        penalty:
            SIMP exponent ``p`` (typically 3).
        e_min_fraction:
            Void-region stiffness floor as a fraction of ``E0``, avoiding a
            singular global stiffness matrix when a cell's density is zero.
        """
        if len(shape) != 2:
            raise ValueError("ElasticityOracle needs a 2D (ny, nx) grid of elements")
        ny, nx = shape
        if ny < 1 or nx < 1:
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        self.shape = (int(ny), int(nx))
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.penalty = float(penalty)
        self.e_min = float(e_min_fraction) * self.young_modulus

        self._nny, self._nnx = ny + 1, nx + 1
        self._n_dof = 2 * self._nny * self._nnx

        self._fixed = sorted({self._dof_index(*t) for t in fixed_dofs})
        if not self._fixed:
            raise ValueError("fixed_dofs must constrain at least one degree of freedom")
        all_dofs = np.arange(self._n_dof)
        fixed_mask = np.zeros(self._n_dof, dtype=bool)
        fixed_mask[self._fixed] = True
        self._free = all_dofs[~fixed_mask]

        F = np.zeros(self._n_dof)
        for (i, j, axis), force in loads.items():
            F[self._dof_index(i, j, axis)] += float(force)
        self._F = F
        if not np.any(F[self._free] != 0.0):
            raise ValueError("loads must apply nonzero force on at least one free dof")

        self._k0 = None  # unit-density (E0 - E_min) element stiffness, set per spacing
        self._h = None
        self._elem_dof = self._element_dof_table()

    def _dof_index(self, node_x: int, node_y: int, axis: str) -> int:
        if axis not in _AXES:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        if not (0 <= node_x < self._nnx) or not (0 <= node_y < self._nny):
            raise ValueError(
                f"node ({node_x}, {node_y}) out of range for "
                f"{self._nny}x{self._nnx} node grid"
            )
        node = node_y * self._nnx + node_x
        return 2 * node + _AXES[axis]

    def _element_dof_table(self) -> np.ndarray:
        """Row e -> the 8 global dof indices of element e, in q4 local order."""
        ny, nx = self.shape
        table = np.zeros((ny * nx, 8), dtype=int)
        for ey in range(ny):
            for ex in range(nx):
                e = ey * nx + ex
                # local node order: (ex,ey), (ex+1,ey), (ex+1,ey+1), (ex,ey+1)
                nodes = [
                    (ex, ey),
                    (ex + 1, ey),
                    (ex + 1, ey + 1),
                    (ex, ey + 1),
                ]
                dofs = []
                for (nx_, ny_) in nodes:
                    node = ny_ * self._nnx + nx_
                    dofs.extend([2 * node, 2 * node + 1])
                table[e] = dofs
        return table

    def _unit_stiffness(self, h: float) -> np.ndarray:
        if self._k0 is None or self._h != h:
            # k0 is the stiffness at unit (E0 - E_min) modulus; SIMP scales it
            # per element by (E_min + rho**p * (E0 - E_min)) / (E0 - E_min)... but
            # it is simpler and exactly equivalent to assemble at full E0 and
            # add the void floor as a separate uniform term. We instead build
            # k0 directly at modulus (E0 - E_min) so K_e = E_min*k_floor +
            # rho_e**p * k0; k_floor is k0 scaled by E_min/(E0-E_min) when E0
            # != E_min, assembled below from the same shape function call.
            self._k0 = q4_plane_stress_stiffness(
                self.young_modulus - self.e_min, self.poisson_ratio, h
            )
            self._k_floor = q4_plane_stress_stiffness(
                self.e_min, self.poisson_ratio, h
            )
            self._h = h
        return self._k0

    def _global_stiffness(self, rho_flat: np.ndarray, h: float):
        k0 = self._unit_stiffness(h)
        k_floor = self._k_floor
        n_elem = rho_flat.size
        scale = rho_flat ** self.penalty  # (n_elem,)

        rows = np.repeat(self._elem_dof, 8, axis=1).reshape(n_elem, 8, 8)
        cols = np.tile(self._elem_dof, (1, 8)).reshape(n_elem, 8, 8)
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

        uf = spsolve(Kff.tocsc(), Ff)
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

        displacement = u.reshape(self._nny, self._nnx, 2)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "compliance": compliance,
                "displacement": displacement,
            },
        )
