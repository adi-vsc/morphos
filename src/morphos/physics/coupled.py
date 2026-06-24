"""Monolithic (simultaneous) coupled thermo-elastic solver.

:class:`~morphos.physics.thermoelastic.ThermoElasticOracle` solves the thermal
field, then plugs the resulting thermal pre-stress into a *separate*
mechanical solve -- a staggered (operator-split) scheme. That scheme is exact
for this particular coupling (the thermal field never depends on the
mechanical displacement, so there is no feedback loop to iterate), but it is
the template for couplings that *do* feed back (e.g. viscous heating, contact,
or any case where the "downstream" physics also forces the "upstream" one).
For those, repeatedly re-solving each block to convergence is the staggered
fixed-point iteration, and it can converge slowly -- or only linearly -- when
the coupling is strong.

``MonolithicCoupledOracle`` instead assembles *one* block system

    [ K   -C ] [ u ]   [ F + C T_ref ]
    [ 0    A ] [ T ] = [ Q           ]

(mechanical block ``K``, thermal block ``A``, coupling block ``C``, both SIMP
penalised by the same density field) and solves it in a single linear solve.
The off-diagonal block makes the system block-triangular for this particular
physics (thermal does not depend on mechanical), so the monolithic and
staggered solutions are mathematically identical here; the value of the
monolithic path is that it generalises immediately to a coupling where the
upper-right block is nonzero too (full two-way coupling), which would make
the staggered iteration genuinely necessary and the monolithic solve the
single-shot alternative. The combined operator is non-symmetric (the
coupling block only appears in one off-diagonal slot), so the iterative path
uses GMRES via ``solve_linear(..., indefinite=True)`` rather than CG.

Field is the only data currency: ``solve`` takes a Field and returns a
PhysicsResult; the displacement and temperature live in ``aux``, matching
:class:`~morphos.physics.thermoelastic.ThermoElasticOracle` exactly so the two
are drop-in interchangeable for any consumer that only reads ``aux``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
from scipy import sparse

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.thermoelastic import ThermoElasticOracle


class MonolithicCoupledOracle(PhysicsOracle):
    """One-shot block-system solve of the same thermo-elastic problem
    :class:`ThermoElasticOracle` solves staggered.

    Takes exactly the same constructor arguments as ``ThermoElasticOracle``
    (it reuses that class internally for assembly: unit-element operators,
    DOF tables, and boundary conditions) but solves the combined
    mechanical+thermal system as a single sparse linear solve on the free
    DOFs of both fields stacked into one vector, instead of two sequential
    solves.
    """

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
        # Delegate all assembly (unit operators, DOF tables, BC bookkeeping)
        # to a private ThermoElasticOracle instance; this class only changes
        # *how* the assembled blocks are solved.
        self._te = ThermoElasticOracle(
            shape=shape,
            fixed_dofs=fixed_dofs,
            fixed_temps=fixed_temps,
            heat_sources=heat_sources,
            loads=loads,
            objective_dofs=objective_dofs,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            conductivity0=conductivity0,
            thermal_expansion=thermal_expansion,
            t_ref=t_ref,
            penalty=penalty,
            e_min_fraction=e_min_fraction,
            k_min_fraction=k_min_fraction,
            solver=solver,
        )
        self.shape = self._te.shape
        self.ndim = self._te.ndim
        self.solver = solver
        # Single coupled-operator AMG/ILU cache (one combined sparsity
        # pattern, unlike ThermoElasticOracle's two separate caches).
        self._cache: list = []
        #: number of outer (fixed-point) iterations the *equivalent* staggered
        #: scheme would need; always 1 here because the block solve is exact
        #: in a single linear solve regardless of coupling strength.
        self.last_outer_iterations = 1

    @property
    def penalty(self) -> float:
        return self._te.penalty

    @penalty.setter
    def penalty(self, value: float) -> None:
        self._te.penalty = float(value)

    def solve(self, field: Field) -> PhysicsResult:
        te = self._te
        if field.values.shape != te.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {te.shape} (elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("MonolithicCoupledOracle assumes isotropic spacing")
        h = spacing[0]
        te._build_unit_operators(h)

        rho_flat = rho.ravel()
        scale = rho_flat ** te.penalty

        K = te._assemble(
            te._mech_elem_dof, te._mech_elem_dof,
            te._k_floor[None] + scale[:, None, None] * te._k0[None],
            te._n_mech, te._n_mech,
        )
        A = te._assemble(
            te._therm_elem_dof, te._therm_elem_dof,
            te._a_floor[None] + scale[:, None, None] * te._a0[None],
            te._n_therm, te._n_therm,
        )
        C = te._assemble(
            te._mech_elem_dof, te._therm_elem_dof,
            te._L_floor[None] + scale[:, None, None] * te._L0[None],
            te._n_mech, te._n_therm,
        )

        free_m, free_t = te._free_m, te._free_t
        n_m, n_t = free_m.size, free_t.size

        Kff = K[np.ix_(free_m, free_m)]
        Aff = A[np.ix_(free_t, free_t)]
        Cft = C[np.ix_(free_m, free_t)]

        # Block system on the stacked free-DOF vector [u_free; T_free]:
        #   [ K   -C ] [u]   [F]
        #   [ 0    A ] [T] = [Q + C^T T_ref-folded already via theta=T-T_ref]
        # theta = T - t_ref is folded by shifting Q's effective RHS: since
        # C couples to theta = T - t_ref, write T = theta + t_ref and absorb
        # the constant t_ref shift into the mechanical RHS instead, keeping
        # the unknown vector as (u, theta) so the block structure is exact.
        zero_block = sparse.csr_matrix((n_t, n_m))
        top = sparse.hstack([Kff, -Cft], format="csr")
        bottom = sparse.hstack([zero_block, Aff], format="csr")
        M = sparse.vstack([top, bottom], format="csr")

        rhs = np.concatenate([te._F[free_m], te._Q[free_t]])

        x, residual_norm, iterations = solve_linear(
            M, rhs, solver=self.solver, dof_count=n_m + n_t,
            indefinite=True, cache_holder=self._cache,
        )
        uf = x[:n_m]
        thetaf = x[n_m:]

        u = np.zeros(te._n_mech)
        u[free_m] = uf
        theta = np.zeros(te._n_therm)
        theta[free_t] = thetaf
        T = theta + te.t_ref

        J = float(te._l @ u)
        value = -J

        # Co-state system: the block operator is not symmetric (only the
        # mechanical row carries the coupling), so the adjoint system is the
        # transpose of M, solved against the objective's mechanical weights
        # stacked with zero thermal forcing (J depends on the state only
        # through u, i.e. only through the top block of the stacked vector).
        adj_rhs = np.concatenate([te._l[free_m], np.zeros(n_t)])
        y, res_adj, it_adj = solve_linear(
            M.T.tocsr(), adj_rhs, solver=self.solver, dof_count=n_m + n_t,
            indefinite=True, cache_holder=self._cache,
        )
        mu_f = y[:n_m]
        psi_f = y[n_m:]
        mu = np.zeros(te._n_mech)
        mu[free_m] = mu_f
        psi = np.zeros(te._n_therm)
        psi[free_t] = psi_f

        residual_norm = max(residual_norm, res_adj)
        solver_iterations = max(iterations, it_adj)

        dscale = te.penalty * rho_flat ** (te.penalty - 1.0)
        u_e = u[te._mech_elem_dof]
        mu_e = mu[te._mech_elem_dof]
        T_e = T[te._therm_elem_dof]
        th_e = theta[te._therm_elem_dof]
        psi_e = psi[te._therm_elem_dof]

        mech = np.einsum("ei,ij,ej->e", mu_e, te._k0, u_e)
        coup = np.einsum("ei,ij,ej->e", mu_e, te._L0, th_e)
        cond = np.einsum("ei,ij,ej->e", psi_e, te._a0, T_e)
        grad_flat = dscale * (mech - coup + cond)
        gradient = grad_flat.reshape(self.shape)

        displacement = u.reshape(*te._nn, te.ndim)
        temperature = T.reshape(te._nn)

        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "objective": J,
                "displacement": displacement,
                "temperature": temperature,
            },
            residual_norm=residual_norm,
            solver_iterations=solver_iterations,
        )
