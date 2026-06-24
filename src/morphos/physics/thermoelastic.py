"""A coupled multiphysics backend: steady thermo-elasticity for SIMP topology
optimization, 2D plane-stress (Q4) or 3D solid (Hex8).

This is the engine's first *coupled* oracle: two PDEs solved on the same design
field, where the solution of one drives the load of the other. A single SIMP
density ``rho`` (per element) interpolates both the thermal conductivity and the
Young's modulus, so the same geometry that conducts heat also carries load.

    A(rho) T = Q                          (steady heat conduction, sink T = 0)
    K(rho) u = F_mech + C(rho) (T - T_ref) (elasticity with thermal pre-stress)

``A`` is the SIMP diffusion stiffness (as in :mod:`morphos.physics.darcy`), ``K``
the SIMP elasticity stiffness (as in :mod:`morphos.physics.elasticity`), and
``C`` the thermo-elastic coupling assembled from
:func:`morphos.physics.operators.q4_thermoelastic_coupling` /
``hex8_thermoelastic_coupling``: a temperature rise expands the material, and
where that expansion is constrained it becomes a mechanical load. Conductivity,
stiffness, and coupling all scale by ``rho**p`` (void material neither conducts
nor expands), the standard thermoelastic-TO interpolation.

Figure of merit. A user-supplied linear functional ``J = l^T u`` (default
``l = F_mech``, i.e. mechanical compliance, the work done by the applied
mechanical loads). The optimizer ascends, so ``value = -J``.

Coupled adjoint. ``J`` depends on ``rho`` through ``K``, through ``C``, and
through ``T`` (which itself depends on ``rho`` via ``A``). Differentiating
``K u = F_mech + C theta`` and ``A T = Q`` (``theta = T - T_ref``) and
introducing a mechanical co-state ``mu`` (``K mu = l``) and a thermal co-state
``psi`` (``A psi = C^T mu``) gives a fully element-local sensitivity:

    d(value)/drho_e = p rho_e**(p-1) * [ + mu_e^T k0  u_e        (mech stiffness)
                                         - mu_e^T L0  theta_e     (coupling)
                                         + psi_e^T a0 T_e ]       (conduction)

with ``k0``/``L0``/``a0`` the unit-scale element stiffness/coupling/diffusion
matrices. Two extra linear solves (``mu``, ``psi``) on the already-factorisable
operators. When ``thermal_expansion = 0`` the coupling vanishes, ``mu`` collapses
to ``u`` and ``psi`` to zero, and this reduces *exactly* to
:class:`~morphos.physics.elasticity.ElasticityOracle` -- an anchor checked in
``tests/test_thermoelastic.py`` alongside the central-FD directional gate on the
full coupled gradient in 2D and 3D.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
from scipy import sparse

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.operators import (
    hex8_diffusion_stiffness,
    hex8_stiffness,
    hex8_thermoelastic_coupling,
    q4_diffusion_stiffness,
    q4_plane_stress_stiffness,
    q4_thermoelastic_coupling,
)
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_AXES = {"x": 0, "y": 1, "z": 2}

_LOCAL_NODE_OFFSETS = {
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
}


class ThermoElasticOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        fixed_dofs: Iterable[Tuple],
        fixed_temps: Iterable[Tuple[int, ...]],
        heat_sources: Dict[Tuple[int, ...], float],
        loads: Optional[Dict[Tuple, float]] = None,
        objective_dofs: Optional[Dict[Tuple, float]] = None,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        conductivity0: float = 1.0,
        thermal_expansion: float = 1.0,
        t_ref: float = 0.0,
        penalty: float = 3.0,
        e_min_fraction: float = 1e-9,
        k_min_fraction: float = 1e-6,
        solver: Literal["direct", "iterative", "auto"] = "auto",
    ) -> None:
        """Coupled thermo-elastic SIMP oracle, 2D (Q4) or 3D (Hex8).

        Parameters
        ----------
        shape:
            Number of *elements* per axis: ``(ny, nx)`` (2D) or
            ``(nz, ny, nx)`` (3D).
        fixed_dofs:
            Mechanical supports as ``(node_x, node_y, axis)`` triples (2D) or
            ``(node_x, node_y, node_z, axis)`` quadruples (3D), held at zero
            displacement.
        fixed_temps:
            Thermal Dirichlet nodes (the heat sink), held at zero temperature,
            given as node-coordinate tuples.
        heat_sources:
            Mapping of node-coordinate tuples to a nodal heat input ``Q``.
        loads:
            Mechanical point loads, mapping the same dof tuples as
            ``fixed_dofs`` to force. May be empty (a purely thermally driven
            problem) provided ``objective_dofs`` is given.
        objective_dofs:
            The linear-functional weights ``l`` for ``J = l^T u`` (same dof-tuple
            keys as ``loads``). Defaults to ``loads`` (mechanical compliance).
        young_modulus, poisson_ratio, conductivity0, thermal_expansion:
            Solid-material (``rho = 1``) properties.
        t_ref:
            Stress-free reference temperature.
        penalty:
            SIMP exponent ``p``.
        e_min_fraction, k_min_fraction:
            Void floors for stiffness and conductivity, keeping both global
            matrices nonsingular.
        solver:
            ``"direct"``, ``"iterative"`` (AMG-preconditioned CG for both the
            symmetric mechanical and thermal reduced systems), or ``"auto"``
            (threshold-based; see :mod:`morphos.physics._linsolve`). The
            mechanical and thermal operators each get their own cached AMG
            hierarchy (sparsity patterns differ), reused while their
            respective free-dof sets are stable.
        """
        if len(shape) not in (2, 3):
            raise ValueError("ThermoElasticOracle needs a 2D (ny, nx) or 3D (nz, ny, nx) grid")
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        self.ndim = len(shape)
        self.shape = tuple(int(s) for s in shape)
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.conductivity0 = float(conductivity0)
        self.thermal_expansion = float(thermal_expansion)
        self.t_ref = float(t_ref)
        self.penalty = float(penalty)
        self.e_min = float(e_min_fraction) * self.young_modulus
        self.k_min = float(k_min_fraction) * self.conductivity0
        self.solver = solver
        # Two operators (mechanical K, thermal A) with distinct sparsity
        # patterns get independent AMG caches.
        self._amg_cache_mech: list = []
        self._amg_cache_therm: list = []

        self._nn = tuple(s + 1 for s in self.shape)
        self._n_node = int(np.prod(self._nn))
        self._n_mech = self.ndim * self._n_node
        self._n_therm = self._n_node

        # Mechanical supports / free dofs.
        fixed_m = sorted({self._mech_dof_index(*t) for t in fixed_dofs})
        if not fixed_m:
            raise ValueError("fixed_dofs must constrain at least one mechanical dof")
        mask = np.zeros(self._n_mech, dtype=bool)
        mask[fixed_m] = True
        self._free_m = np.arange(self._n_mech)[~mask]

        # Thermal Dirichlet / free dofs.
        fixed_t = sorted({self._node_index(t) for t in fixed_temps})
        if not fixed_t:
            raise ValueError("fixed_temps must constrain at least one node")
        tmask = np.zeros(self._n_therm, dtype=bool)
        tmask[fixed_t] = True
        self._free_t = np.arange(self._n_therm)[~tmask]

        # Heat load.
        Q = np.zeros(self._n_therm)
        for coords, q in heat_sources.items():
            Q[self._node_index(coords)] += float(q)
        self._Q = Q
        if not np.any(Q[self._free_t] != 0.0):
            raise ValueError("heat_sources must inject nonzero heat on at least one free node")

        # Mechanical load.
        loads = loads or {}
        F = np.zeros(self._n_mech)
        for t, force in loads.items():
            F[self._mech_dof_index(*t)] += float(force)
        self._F = F

        # Objective functional l (defaults to the mechanical loads).
        obj = objective_dofs if objective_dofs is not None else loads
        l = np.zeros(self._n_mech)
        for t, w in obj.items():
            l[self._mech_dof_index(*t)] += float(w)
        self._l = l
        if not np.any(l[self._free_m] != 0.0):
            raise ValueError(
                "objective must be nonzero on a free mechanical dof "
                "(pass objective_dofs when there are no mechanical loads)"
            )

        self._mech_elem_dof = self._mech_element_dof_table()
        self._therm_elem_dof = self._therm_element_dof_table()
        self._cache_h = None

    # -- indexing -----------------------------------------------------------
    def _node_index(self, coords: Tuple[int, ...]) -> int:
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate(s), got {coords!r}")
        coords_axis_order = tuple(reversed(coords))
        for c, n in zip(coords_axis_order, self._nn):
            if not (0 <= c < n):
                raise ValueError(f"node {coords} out of range for node grid {self._nn}")
        return int(np.ravel_multi_index(coords_axis_order, self._nn))

    def _mech_dof_index(self, *args) -> int:
        *coords, axis = args
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate(s) plus axis, got {args!r}")
        if axis not in _AXES or _AXES[axis] >= self.ndim:
            raise ValueError(f"axis must be one of {list(_AXES)[: self.ndim]}, got {axis!r}")
        return self.ndim * self._node_index(tuple(coords)) + _AXES[axis]

    def _mech_element_dof_table(self) -> np.ndarray:
        n_elem = int(np.prod(self.shape))
        n_local = self.ndim * (2 ** self.ndim)
        table = np.zeros((n_elem, n_local), dtype=int)
        offsets = _LOCAL_NODE_OFFSETS[self.ndim]
        for idx in np.ndindex(*self.shape):
            e = int(np.ravel_multi_index(idx, self.shape))
            dofs = []
            for off in offsets:
                node_coord = tuple(i + o for i, o in zip(idx, tuple(reversed(off))))
                node = int(np.ravel_multi_index(node_coord, self._nn))
                dofs.extend(self.ndim * node + k for k in range(self.ndim))
            table[e] = dofs
        return table

    def _therm_element_dof_table(self) -> np.ndarray:
        n_elem = int(np.prod(self.shape))
        n_local = 2 ** self.ndim
        table = np.zeros((n_elem, n_local), dtype=int)
        offsets = _LOCAL_NODE_OFFSETS[self.ndim]
        for idx in np.ndindex(*self.shape):
            e = int(np.ravel_multi_index(idx, self.shape))
            nodes = []
            for off in offsets:
                node_coord = tuple(i + o for i, o in zip(idx, tuple(reversed(off))))
                nodes.append(int(np.ravel_multi_index(node_coord, self._nn)))
            table[e] = nodes
        return table

    # -- unit element operators (cached per spacing) ------------------------
    def _build_unit_operators(self, h: float) -> None:
        if self._cache_h == h:
            return
        E0, nu, alpha = self.young_modulus - self.e_min, self.poisson_ratio, self.thermal_expansion
        k0c = self.conductivity0 - self.k_min
        if self.ndim == 2:
            self._k0 = q4_plane_stress_stiffness(E0, nu, h)
            self._k_floor = q4_plane_stress_stiffness(self.e_min, nu, h)
            self._a0 = q4_diffusion_stiffness(k0c, h)
            self._a_floor = q4_diffusion_stiffness(self.k_min, h)
            self._L0 = q4_thermoelastic_coupling(E0, nu, alpha, h)
            self._L_floor = q4_thermoelastic_coupling(self.e_min, nu, alpha, h)
        else:
            self._k0 = hex8_stiffness(E0, nu, h)
            self._k_floor = hex8_stiffness(self.e_min, nu, h)
            self._a0 = hex8_diffusion_stiffness(k0c, h)
            self._a_floor = hex8_diffusion_stiffness(self.k_min, h)
            self._L0 = hex8_thermoelastic_coupling(E0, nu, alpha, h)
            self._L_floor = hex8_thermoelastic_coupling(self.e_min, nu, alpha, h)
        self._cache_h = h

    def _assemble(self, elem_dof_r, elem_dof_c, vals, n_rows, n_cols):
        n_elem, nr, nc = vals.shape
        rows = np.repeat(elem_dof_r, nc, axis=1).reshape(n_elem, nr, nc)
        cols = np.tile(elem_dof_c, (1, nr)).reshape(n_elem, nr, nc)
        return sparse.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n_rows, n_cols)
        ).tocsr()

    # -- solve --------------------------------------------------------------
    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("ThermoElasticOracle assumes isotropic spacing")
        h = spacing[0]
        self._build_unit_operators(h)

        rho_flat = rho.ravel()
        scale = rho_flat ** self.penalty

        K = self._assemble(
            self._mech_elem_dof, self._mech_elem_dof,
            self._k_floor[None] + scale[:, None, None] * self._k0[None],
            self._n_mech, self._n_mech,
        )
        A = self._assemble(
            self._therm_elem_dof, self._therm_elem_dof,
            self._a_floor[None] + scale[:, None, None] * self._a0[None],
            self._n_therm, self._n_therm,
        )
        C = self._assemble(
            self._mech_elem_dof, self._therm_elem_dof,
            self._L_floor[None] + scale[:, None, None] * self._L0[None],
            self._n_mech, self._n_therm,
        )

        free_m, free_t = self._free_m, self._free_t

        # 1. thermal forward solve
        Aff = A[np.ix_(free_t, free_t)].tocsc()
        Tf, res_t1, it_t1 = solve_linear(
            Aff, self._Q[free_t], solver=self.solver, dof_count=free_t.size,
            symmetric=True, cache_holder=self._amg_cache_therm,
        )
        T = np.zeros(self._n_therm)
        T[free_t] = Tf
        theta = T - self.t_ref

        # 2. mechanical forward solve (thermal pre-stress as a load)
        Kff = K[np.ix_(free_m, free_m)].tocsc()
        rhs = self._F + C @ theta
        uf, res_m1, it_m1 = solve_linear(
            Kff, rhs[free_m], solver=self.solver, dof_count=free_m.size,
            symmetric=True, cache_holder=self._amg_cache_mech,
        )
        u = np.zeros(self._n_mech)
        u[free_m] = uf

        J = float(self._l @ u)
        value = -J

        # 3. mechanical co-state  K mu = l
        muf, res_m2, it_m2 = solve_linear(
            Kff, self._l[free_m], solver=self.solver, dof_count=free_m.size,
            symmetric=True, cache_holder=self._amg_cache_mech,
        )
        mu = np.zeros(self._n_mech)
        mu[free_m] = muf

        # 4. thermal co-state  A psi = C^T mu
        ctmu = C.T @ mu
        psif, res_t2, it_t2 = solve_linear(
            Aff, ctmu[free_t], solver=self.solver, dof_count=free_t.size,
            symmetric=True, cache_holder=self._amg_cache_therm,
        )
        psi = np.zeros(self._n_therm)
        psi[free_t] = psif

        residual_norm = max(res_t1, res_m1, res_m2, res_t2)
        iterations = max(it_t1, it_m1, it_m2, it_t2)

        # 5. element-local coupled sensitivity (value = -J)
        dscale = self.penalty * rho_flat ** (self.penalty - 1.0)
        u_e = u[self._mech_elem_dof]
        mu_e = mu[self._mech_elem_dof]
        T_e = T[self._therm_elem_dof]
        th_e = theta[self._therm_elem_dof]
        psi_e = psi[self._therm_elem_dof]

        mech = np.einsum("ei,ij,ej->e", mu_e, self._k0, u_e)
        coup = np.einsum("ei,ij,ej->e", mu_e, self._L0, th_e)
        cond = np.einsum("ei,ij,ej->e", psi_e, self._a0, T_e)
        grad_flat = dscale * (mech - coup + cond)
        gradient = grad_flat.reshape(self.shape)

        displacement = u.reshape(*self._nn, self.ndim)
        temperature = T.reshape(self._nn)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "objective": J,
                "displacement": displacement,
                "temperature": temperature,
            },
            residual_norm=residual_norm,
            solver_iterations=iterations,
        )
