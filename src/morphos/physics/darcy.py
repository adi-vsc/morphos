"""A real PDE physics backend: density-dependent Darcy (potential) flow for
channel-routing topology optimization -- the first fluid-adjacent oracle in
the structural/thermal/EM/fluid topology-optimization quartet.

Scope. This solves pressure-driven Darcy flow:

    -div(K(rho) grad(p)) = s   inside,   p = 0   on the fixed (outlet) nodes,

with the design field ``rho`` controlling a scalar permeability ``K(rho)``
(SIMP-style, exactly like ``ElasticityOracle``'s Young's modulus): high
density is an open, low-resistance channel, low density is a near-impermeable
wall. This is *not* the full viscous incompressible Stokes equations -- there
is no separate velocity field, no no-slip boundary layer, and no viscous
stress tensor. It is the fast, well-posed potential-flow surrogate used for
flow-network / channel-topology design (the fluid analogue of how a resistor
network approximates a heat sink), deliberately scoped this way because a
correct, stable mixed-FEM Stokes saddle-point solve is a materially larger
and riskier undertaking that deserves dedicated, supervised derivation rather
than an unattended first cut (see docs/strategy for the recommended follow-up).

Mathematically this oracle is `ElasticityOracle` with one degree of freedom
per node instead of `ndim`: the same SIMP permeability interpolation, the
same Q4/Hex8 finite-element assembly (now the scalar diffusion element from
:mod:`morphos.physics.operators` rather than the vector elasticity one), and
the same self-adjoint "compliance" trick, since the source ``s`` does not
depend on ``rho`` and ``A`` is symmetric:

    A(rho) p = s   =>   dissipation = s^T p = p^T A p
    d(dissipation)/drho_e = -p_e^T (dA_e/drho_e) p_e
    d(value)/drho_e = +p_e^T (dA_e/drho_e) p_e      (value = -dissipation)

where ``p_e`` is the element's local nodal pressure vector and ``dA_e/drho_e``
is the SIMP derivative of that element's diffusion stiffness. Gated against
central finite differences and the directional FD gate in ``tests/test_darcy.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from morphos.field import Field
from morphos.physics.operators import hex8_diffusion_stiffness, q4_diffusion_stiffness
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_LOCAL_NODE_OFFSETS = {
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}


class DarcyFlowOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        fixed_nodes: Iterable[Tuple[int, ...]],
        sources: Dict[Tuple[int, ...], float],
        permeability0: float = 1.0,
        penalty: float = 3.0,
        k_min_fraction: float = 1e-6,
    ) -> None:
        """SIMP-style Darcy-flow channel oracle, 2D (Q4) or 3D (Hex8).

        Parameters
        ----------
        shape:
            Number of *elements* per axis: ``(ny, nx)`` (2D) or
            ``(nz, ny, nx)`` (3D).
        fixed_nodes:
            Iterable of ``(node_x, node_y[, node_z])`` coordinate tuples held
            at zero pressure (the outlet / reference boundary condition).
        sources:
            Mapping of the same coordinate tuples to a nodal flow source;
            unlisted nodes carry zero source.
        permeability0:
            Open-channel (``rho = 1``) permeability, ``K0`` below.
        penalty:
            SIMP exponent ``p`` (typically 3).
        k_min_fraction:
            Closed-wall permeability floor as a fraction of ``K0``, avoiding
            a singular global matrix when a cell's density is zero.
        """
        if len(shape) not in (2, 3):
            raise ValueError("DarcyFlowOracle needs a 2D (ny, nx) or 3D (nz, ny, nx) grid")
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        self.ndim = len(shape)
        self.shape = tuple(int(s) for s in shape)
        self.permeability0 = float(permeability0)
        self.penalty = float(penalty)
        self.k_min = float(k_min_fraction) * self.permeability0

        self._nn = tuple(s + 1 for s in self.shape)
        self._n_dof = int(np.prod(self._nn))

        self._fixed = sorted({self._node_index(t) for t in fixed_nodes})
        if not self._fixed:
            raise ValueError("fixed_nodes must constrain at least one node")
        all_dofs = np.arange(self._n_dof)
        fixed_mask = np.zeros(self._n_dof, dtype=bool)
        fixed_mask[self._fixed] = True
        self._free = all_dofs[~fixed_mask]

        S = np.zeros(self._n_dof)
        for coords, value in sources.items():
            S[self._node_index(coords)] += float(value)
        self._S = S
        if not np.any(S[self._free] != 0.0):
            raise ValueError("sources must inject nonzero flow on at least one free node")

        self._k0 = None
        self._h = None
        self._elem_dof = self._element_dof_table()

    def _node_index(self, coords: Tuple[int, ...]) -> int:
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate(s), got {coords!r}")
        coords_axis_order = tuple(reversed(coords))
        for c, n in zip(coords_axis_order, self._nn):
            if not (0 <= c < n):
                raise ValueError(f"node {coords} out of range for node grid {self._nn}")
        return int(np.ravel_multi_index(coords_axis_order, self._nn))

    def _element_dof_table(self) -> np.ndarray:
        n_elem = int(np.prod(self.shape))
        n_local = 2 ** self.ndim
        table = np.zeros((n_elem, n_local), dtype=int)
        offsets = _LOCAL_NODE_OFFSETS[self.ndim]
        for idx in np.ndindex(*self.shape):
            e = int(np.ravel_multi_index(idx, self.shape))
            nodes = []
            for off in offsets:
                off_axis_order = tuple(reversed(off))
                node_coord = tuple(i + o for i, o in zip(idx, off_axis_order))
                nodes.append(int(np.ravel_multi_index(node_coord, self._nn)))
            table[e] = nodes
        return table

    def _element_stiffness(self, conductivity: float, h: float) -> np.ndarray:
        if self.ndim == 2:
            return q4_diffusion_stiffness(conductivity, h)
        return hex8_diffusion_stiffness(conductivity, h)

    def _unit_stiffness(self, h: float) -> np.ndarray:
        if self._k0 is None or self._h != h:
            self._k0 = self._element_stiffness(self.permeability0 - self.k_min, h)
            self._k_floor = self._element_stiffness(self.k_min, h)
            self._h = h
        return self._k0

    def _global_stiffness(self, rho_flat: np.ndarray, h: float):
        k0 = self._unit_stiffness(h)
        k_floor = self._k_floor
        n_elem = rho_flat.size
        n_local = self._elem_dof.shape[1]
        scale = rho_flat ** self.penalty

        rows = np.repeat(self._elem_dof, n_local, axis=1).reshape(n_elem, n_local, n_local)
        cols = np.tile(self._elem_dof, (1, n_local)).reshape(n_elem, n_local, n_local)
        vals = k_floor[None, :, :] + scale[:, None, None] * k0[None, :, :]

        return sparse.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof, self._n_dof),
        ).tocsr()

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("DarcyFlowOracle assumes isotropic spacing")
        h = spacing[0]

        rho_flat = rho.ravel()
        K = self._global_stiffness(rho_flat, h)

        free = self._free
        Kff = K[np.ix_(free, free)]
        Sf = self._S[free]

        pf = spsolve(Kff.tocsc(), Sf)
        p = np.zeros(self._n_dof)
        p[free] = pf

        dissipation = float(self._S @ p)
        value = -dissipation

        k0 = self._k0
        p_e = p[self._elem_dof]
        energy = np.einsum("ei,ij,ej->e", p_e, k0, p_e)
        dK_drho = self.penalty * rho_flat ** (self.penalty - 1.0)
        grad_flat = dK_drho * energy
        gradient = grad_flat.reshape(self.shape)

        pressure = p.reshape(self._nn)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={"dissipation": dissipation, "pressure": pressure},
        )
