"""Geometric nonlinearity via the co-rotational Q4 formulation (Crisfield 1991;
Felippa & Haugen 2005).

:class:`~morphos.physics.elasticity.ElasticityOracle` assumes small strains
*and* small rotations: its strain-displacement operator ``B`` is evaluated
once on the undeformed mesh and never updated, which is the standard
linear-elasticity assumption. That assumption breaks for flexible,
large-rotation structures -- compliant mechanisms, bio-inspired morphing
skins, anything that bends a lot under load even while the *material* strain
stays small. The co-rotational formulation is the cheapest fix that keeps the
linear constitutive law: track, per element, a rigid rotation ``R_e``
extracted from the current (deformed) configuration, evaluate the *linear*
strain-displacement relation in that rotated frame, and rotate the resulting
element stiffness back into the global frame before assembly. Large
rotation + small strain is exactly the compliant-mechanism / morphing-skin
regime this oracle targets; for large *strains* a fully nonlinear
(Green-Lagrange) formulation would be needed instead.

Per element, every Newton-Raphson iteration:

1. Track current nodal positions ``x_current = x_ref + u_e``.
2. Extract the element's rigid rotation angle ``theta_e`` from the deformed
   shape of its sides (the 2D analogue of polar-decomposing the deformation
   gradient ``F = R U``).
3. Build ``R_e``, the 8x8 block-diagonal rotation matrix (four repeated 2x2
   blocks, one per node).
4. Material (SIMP-penalized) stiffness in the *reference* frame is the same
   :func:`~morphos.physics.operators.q4_plane_stress_stiffness` used by
   :class:`~morphos.physics.elasticity.ElasticityOracle`. The element's
   internal force is computed from the *local* (de-rotated) displacement
   ``u_local_e = R_e^T x_current_e - x_ref_e`` -- zero by construction for a
   pure rigid rotation -- as ``f_int_e = R_e (k_lin u_local_e)``; the
   (approximate, standard simplified) tangent stiffness used to pick the
   Newton search direction is the rotated linear stiffness
   ``K_e = R_e @ k_lin @ R_e.T``.
5. Assemble the global internal-force vector and tangent stiffness from
   these element contributions and Newton-Raphson iterate
   ``K_T(u) @ delta_u = f_ext - f_int(u)`` until the residual norm falls
   below ``nl_tol``.

This module reuses (does not re-derive) the unit Q4 stiffness from
:mod:`morphos.physics.operators`, and copies the DOF-indexing helpers from
:mod:`morphos.physics.elasticity` so the two oracles are drop-in compatible
(same ``shape``/``fixed_dofs``/``loads`` argument shapes).

Sensitivity. The figure of merit is compliance ``J = -f_ext^T u`` evaluated at
the converged nonlinear state, exactly as in the linear oracle. At
convergence the internal force is the assembled local-frame elastic force
rotated back to global, ``f_int_e = R_e (k_lin u_local_e)``, with
``u_local_e = R_e^T x_current_e - x_ref_e`` the de-rotated ("local")
displacement that the linear constitutive law actually sees -- by
construction zero for a pure rigid rotation, which is what makes the
co-rotational element exactly objective (frame-indifferent). The element
strain energy is therefore ``u_local_e^T k0_e u_local_e`` (not the raw global
``u_e^T k0_e u_e``, which is *not* frame-indifferent and is nonzero even
under a pure rotation). The same self-adjoint trick as linear SIMP compliance
then applies to the *explicit* density derivative of that energy, holding the
converged ``u_local_e`` fixed:

    dJ/drho_e = -p * rho_e**(p-1) * u_local_e^T @ k0 @ u_local_e

where ``k0`` is the unit-density Q4 stiffness. This is the same formula as
linear compliance with ``u_e`` replaced by ``u_local_e``; the co-rotational
twist is entirely in using the local (frame-indifferent) displacement rather
than in the sensitivity formula itself. Gated against central finite
differences in ``tests/test_nonlinear_elasticity.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from morphos.field import Field
from morphos.physics.operators import q4_plane_stress_stiffness
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_AXES = {"x": 0, "y": 1}

# (dx, dy) offsets of each element's local nodes, same natural-coordinate
# corner order as q4_plane_stress_stiffness / ElasticityOracle.
_LOCAL_NODE_OFFSETS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    """8x8 block-diagonal rotation matrix: four repeated 2x2 blocks, one per
    Q4 node, rotating the element's local (reference-aligned) displacement
    DOFs into the global frame by angle ``theta``."""
    c, s = np.cos(theta), np.sin(theta)
    r2 = np.array([[c, -s], [s, c]])
    R = np.zeros((8, 8))
    for i in range(4):
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = r2
    return R


class CorotationalQ4Oracle(PhysicsOracle):
    """Geometric nonlinear Q4 FEM using the co-rotational formulation.

    For SIMP topology optimization of structures undergoing large rotations
    but small strains (compliant mechanisms, flexible bio-inspired
    structures). Shares the same oracle interface as
    :class:`~morphos.physics.elasticity.ElasticityOracle`; can drop in as a
    replacement when geometric nonlinearity matters (strains beyond roughly
    5%, or rotations large enough that the linear small-angle assumption
    breaks down).
    """

    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, int],
        fixed_dofs: Iterable[Tuple[int, int, str]],
        loads: Dict[Tuple[int, int, str], float],
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        penalty: float = 3.0,
        rho_min: float = 1e-3,
        nl_tol: float = 1e-8,
        max_nl_iter: int = 20,
        load_steps: int = 1,
        objective_dofs: Optional[Iterable[Tuple[int, int, str]]] = None,
    ) -> None:
        """Co-rotational SIMP compliance oracle, 2D plane-stress (Q4) only.

        Parameters
        ----------
        shape:
            Number of *elements* per axis, ``(ny, nx)``.
        fixed_dofs:
            Iterable of ``(node_x, node_y, axis)`` triples, ``axis`` in
            ``{"x", "y"}``, naming nodal degrees of freedom held at zero
            displacement -- same format as ``ElasticityOracle``.
        loads:
            Mapping of the same coordinate tuples to applied force; unlisted
            DOFs carry zero force.
        young_modulus, poisson_ratio:
            Solid-material (``rho = 1``) elastic properties.
        penalty:
            SIMP exponent ``p`` (typically 3).
        rho_min:
            Void-region stiffness floor as a *fraction of* ``young_modulus``
            (interpolation ``E(rho) = rho_min + (1 - rho_min) * rho**p``,
            in units of ``young_modulus``), keeping the tangent stiffness
            nonsingular at zero density.
        nl_tol:
            Newton-Raphson relative convergence tolerance on the force
            residual.
        max_nl_iter:
            Maximum Newton-Raphson iterations per load step.
        load_steps:
            Number of load increments (1 = apply the full load in one shot;
            more steps help convergence for large loads / strongly nonlinear
            problems).
        objective_dofs:
            DOFs defining the linear functional ``J = l^T u`` (the figure of
            merit before negation); defaults to the loaded DOFs (mechanical
            compliance, mirroring ``ElasticityOracle``).
        """
        if len(shape) != 2:
            raise ValueError("CorotationalQ4Oracle only supports 2D (ny, nx) grids")
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        if load_steps < 1:
            raise ValueError("load_steps must be >= 1")

        self.ndim = 2
        self.shape = (int(shape[0]), int(shape[1]))
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.penalty = float(penalty)
        self.rho_min = float(rho_min)
        self.nl_tol = float(nl_tol)
        self.max_nl_iter = int(max_nl_iter)
        self.load_steps = int(load_steps)

        self._nn = (self.shape[0] + 1, self.shape[1] + 1)
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

        if objective_dofs is None:
            l = F.copy()
        else:
            l = np.zeros(self._n_dof)
            for t, w in objective_dofs.items():
                l[self._dof_index(*t)] += float(w)
        self._l = l

        self._elem_dof = self._element_dof_table()
        self._k0_unit = None  # unit (E=1) Q4 stiffness, cached per spacing
        self._h = None

    # -- DOF indexing (copied from ElasticityOracle's convention) -----------

    def _dof_index(self, *args) -> int:
        *coords, axis = args
        if len(coords) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate(s) plus axis, got {args!r}")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {list(_AXES)}, got {axis!r}")
        coords_axis_order = tuple(reversed(coords))  # (x, y) -> (y, x)
        for c, n in zip(coords_axis_order, self._nn):
            if not (0 <= c < n):
                raise ValueError(f"node {coords} out of range for node grid {self._nn}")
        node = int(np.ravel_multi_index(coords_axis_order, self._nn))
        return self.ndim * node + _AXES[axis]

    def _element_dof_table(self) -> np.ndarray:
        n_elem = int(np.prod(self.shape))
        n_local_dof = self.ndim * 4
        table = np.zeros((n_elem, n_local_dof), dtype=int)
        for idx in np.ndindex(*self.shape):  # idx = (ey, ex)
            e = int(np.ravel_multi_index(idx, self.shape))
            dofs = []
            for off in _LOCAL_NODE_OFFSETS:  # off = (dx, dy)
                off_axis_order = tuple(reversed(off))  # (dy, dx)
                node_coord = tuple(i + o for i, o in zip(idx, off_axis_order))
                node = int(np.ravel_multi_index(node_coord, self._nn))
                dofs.extend(self.ndim * node + k for k in range(self.ndim))
            table[e] = dofs
        return table

    def _node_ref_coords(self, h: float) -> np.ndarray:
        """Reference (undeformed) coordinates of every node, flattened in the
        same global-DOF order as displacements: shape ``(n_node, 2)``."""
        ny_n, nx_n = self._nn
        ys, xs = np.meshgrid(np.arange(ny_n) * h, np.arange(nx_n) * h, indexing="ij")
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1)  # (n_node, 2), row-major (y, x)
        return coords

    # -- element-level operators ----------------------------------------------

    def _unit_stiffness(self, h: float) -> np.ndarray:
        if self._k0_unit is None or self._h != h:
            self._k0_unit = q4_plane_stress_stiffness(1.0, self.poisson_ratio, h)
            self._h = h
        return self._k0_unit

    @staticmethod
    def _element_rotation(u_e: np.ndarray, x_ref_e: np.ndarray) -> float:
        """Extract the rigid-rotation angle of one Q4 element from its
        reference and current (deformed) nodal positions, by averaging the
        rotation of its bottom and right sides relative to the reference
        configuration -- the standard 2D co-rotational "two-side" rotation
        extractor (Crisfield 1991), which exactly recovers the rigid-body
        rotation for any pure rigid rotation of the element."""
        # Side 1-2 (bottom): nodes 0 -> 1.
        dx_ref = x_ref_e[2] - x_ref_e[0]
        dy_ref = x_ref_e[3] - x_ref_e[1]
        dx_cur = dx_ref + u_e[2] - u_e[0]
        dy_cur = dy_ref + u_e[3] - u_e[1]
        theta1 = np.arctan2(dy_cur, dx_cur) - np.arctan2(dy_ref, dx_ref)

        # Side 2-3 (right): nodes 1 -> 2.
        dx_ref = x_ref_e[4] - x_ref_e[2]
        dy_ref = x_ref_e[5] - x_ref_e[3]
        dx_cur = dx_ref + u_e[4] - u_e[2]
        dy_cur = dy_ref + u_e[5] - u_e[3]
        theta2 = np.arctan2(dy_cur, dx_cur) - np.arctan2(dy_ref, dx_ref)

        # Wrap each increment to (-pi, pi] before averaging so the mean is
        # well-defined across the branch cut.
        theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
        theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
        return 0.5 * (theta1 + theta2)

    def _simp_modulus(self, rho_e: np.ndarray) -> np.ndarray:
        return self.rho_min + (1.0 - self.rho_min) * rho_e ** self.penalty

    def _corot_element(
        self, e_modulus: float, k0_unit: np.ndarray, u_e: np.ndarray, x_ref_e: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(K_e_global, f_int_e_global, theta_e)`` for one element.

        ``K_e_global = R_e @ k_lin @ R_e^T`` is the (approximate, standard
        simplified) co-rotational tangent stiffness in the global frame --
        used only to pick the Newton search direction. The internal force is
        *not* ``K_e_global @ u_e``: applying the rotated linear stiffness to
        the raw global displacement spuriously generates force under a pure
        rigid rotation (the rotated stiffness is not itself self-equilibrated
        against the unrotated global displacement). Instead the internal
        force is built from the genuinely co-rotational quantity -- the
        *local* (de-rotated) displacement relative to the reference
        configuration, ``u_local = R_e^T @ x_current - x_ref`` -- which is
        exactly zero for a pure rigid rotation by construction, then mapped
        back to the global frame: ``f_int = R_e @ (k_lin @ u_local)``.
        """
        theta = self._element_rotation(u_e, x_ref_e)
        R_e = _rotation_matrix_2d(theta)
        k_lin = e_modulus * k0_unit

        x_cur_e = x_ref_e + u_e
        u_local = R_e.T @ x_cur_e - x_ref_e
        f_int_local = k_lin @ u_local
        f_int_global = R_e @ f_int_local

        K_e = R_e @ k_lin @ R_e.T
        return K_e, f_int_global, theta

    # -- global assembly -------------------------------------------------------

    def _assemble(self, u: np.ndarray, rho_flat: np.ndarray, h: float, x_ref: np.ndarray):
        """Assemble the global co-rotational tangent stiffness and internal
        force vector at the current displacement ``u``."""
        k0_unit = self._unit_stiffness(h)
        e_modulus = self.young_modulus * self._simp_modulus(rho_flat)  # (n_elem,)

        n_elem = rho_flat.size
        n_local_dof = self._elem_dof.shape[1]

        u_e_all = u[self._elem_dof]  # (n_elem, 8)
        x_ref_e_all = x_ref[self._elem_dof]  # (n_elem, 8)

        K_e_all = np.empty((n_elem, n_local_dof, n_local_dof))
        f_e_all = np.empty((n_elem, n_local_dof))
        for e in range(n_elem):
            K_e_all[e], f_e_all[e], _ = self._corot_element(
                e_modulus[e], k0_unit, u_e_all[e], x_ref_e_all[e]
            )

        rows = np.repeat(self._elem_dof, n_local_dof, axis=1).reshape(
            n_elem, n_local_dof, n_local_dof
        )
        cols = np.tile(self._elem_dof, (1, n_local_dof)).reshape(
            n_elem, n_local_dof, n_local_dof
        )
        K = sparse.coo_matrix(
            (K_e_all.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof, self._n_dof),
        ).tocsr()

        f_int = np.zeros(self._n_dof)
        np.add.at(f_int, self._elem_dof.ravel(), f_e_all.ravel())

        return K, f_int

    def _newton_iterate(self, u0: np.ndarray, f_target: np.ndarray, rho_flat, h, x_ref):
        """Run up to ``max_nl_iter`` full Newton-Raphson steps from ``u0``
        toward equilibrium under ``f_target``. Returns
        ``(u, n_iterations, converged, K_t, f_int)``.
        """
        free = self._free
        n_dof = self._n_dof
        u = u0.copy()
        f_norm = np.linalg.norm(f_target[free]) + 1e-12
        K_t, f_int = self._assemble(u, rho_flat, h, x_ref)

        for nr_iter in range(self.max_nl_iter):
            residual = f_target - f_int
            r_free = residual[free]
            if np.linalg.norm(r_free) < self.nl_tol * f_norm:
                return u, nr_iter, True, K_t, f_int
            K_free = K_t[np.ix_(free, free)].tocsc()
            delta_u = np.zeros(n_dof)
            delta_u[free] = spsolve(K_free, r_free)
            u = u + delta_u
            K_t, f_int = self._assemble(u, rho_flat, h, x_ref)
            r_post = (f_target - f_int)[free]
            if np.linalg.norm(r_post) < self.nl_tol * f_norm:
                return u, nr_iter + 1, True, K_t, f_int

        residual = f_target - f_int
        converged = np.linalg.norm(residual[free]) < self.nl_tol * f_norm
        return u, self.max_nl_iter, converged, K_t, f_int

    def _nr_solve(self, rho_flat: np.ndarray, h: float, x_ref: np.ndarray):
        """Newton-Raphson iteration for the co-rotational nonlinear solve,
        with adaptive load sub-stepping: each requested load increment (of
        ``self.load_steps``) is halved and retried (up to a fixed bisection
        depth) whenever Newton fails to converge within ``max_nl_iter``
        iterations -- the standard fallback for the simplified co-rotational
        tangent (material stiffness rotated into the current frame, without
        the additional geometric-stiffness correction), which loses its
        normal quadratic convergence once an increment drives an element
        through a large incremental rotation.

        Crucially, an increment is only ever *accepted* (advancing the
        current state and ``frac_done``) once Newton actually converges on
        it; an unconverged attempt is always discarded and retried at half
        the step, never used as the starting point for the next increment
        (accepting an unconverged intermediate state is what causes the
        bisection to spiral: each subsequent attempt starts further from
        equilibrium, needs an ever-smaller step, and the total work explodes).
        If the bisection depth bottoms out without converging, the solve
        stops where it is and reports the unconverged iteration count (the
        ``nr_iterations >= max_nl_iter`` signal callers use to detect this).

        Returns ``(u, n_iterations, K_t_final)`` where ``n_iterations`` is
        the iteration count of the *last* Newton solve attempted (the
        natural "did it converge before hitting the cap" diagnostic).
        """
        free = self._free
        u = np.zeros(self._n_dof)
        total_iters = 0
        K_t, _ = self._assemble(u, rho_flat, h, x_ref)

        max_bisections = 6
        base_fraction = 1.0 / self.load_steps
        frac_done = 0.0

        while frac_done < 1.0 - 1e-12:
            remaining = 1.0 - frac_done
            frac = min(base_fraction, remaining)
            for depth in range(max_bisections + 1):
                f_target = self._F * (frac_done + frac)
                u_new, n_iter, converged, K_t_new, _ = self._newton_iterate(
                    u, f_target, rho_flat, h, x_ref
                )
                total_iters = n_iter
                if converged:
                    u, K_t = u_new, K_t_new
                    frac_done += frac
                    break
                if depth == max_bisections:
                    # Smallest allowed sub-step still failed to converge:
                    # accept this best-effort state (so the caller gets a
                    # finite, finite-valued result) and stop advancing --
                    # the unconverged iteration count is the diagnostic.
                    u, K_t = u_new, K_t_new
                    return u, total_iters, K_t
                frac *= 0.5

        return u, total_iters, K_t

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (ny, nx elements)"
            )
        rho = np.clip(field.values, 0.0, 1.0)
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("CorotationalQ4Oracle assumes isotropic spacing")
        h = spacing[0]

        rho_flat = rho.ravel()  # row-major: element e = ey*nx + ex
        x_ref = self._node_ref_coords(h).ravel()  # (n_dof,), interleaved (x, y)

        u, nr_iterations, K_t = self._nr_solve(rho_flat, h, x_ref)

        compliance = float(self._l @ u)
        value = -compliance

        # Co-rotational adjoint sensitivity. For the nonlinear equilibrium
        # f_int(u;rho) = f_ext, the adjoint variable mu satisfies K_T^T mu = l
        # (l = f_ext for compliance). Because K_T is the material co-rotational
        # tangent (symmetric), mu = K_T^{-1} l — a fresh linear solve on the
        # converged tangent, NOT u itself (K_T u ≠ f_ext for nonlinear problems).
        # Sensitivity: dJ/drho_e = mu_e^T R_e (dE/drho_e k0) u_local_e.
        K_t_free = K_t[np.ix_(self._free, self._free)].tocsc()
        mu = np.zeros(self._n_dof)
        mu[self._free] = spsolve(K_t_free, self._l[self._free])

        k0_unit = self._unit_stiffness(h)
        u_e_all = u[self._elem_dof]      # (n_elem, 8)
        mu_e_all = mu[self._elem_dof]    # (n_elem, 8)
        x_ref_e_all = x_ref[self._elem_dof]  # (n_elem, 8)
        n_elem = rho_flat.size
        u_local_all = np.empty_like(u_e_all)
        mu_local_all = np.empty_like(mu_e_all)
        for e in range(n_elem):
            theta = self._element_rotation(u_e_all[e], x_ref_e_all[e])
            R_e = _rotation_matrix_2d(theta)
            x_cur_e = x_ref_e_all[e] + u_e_all[e]
            u_local_all[e] = R_e.T @ x_cur_e - x_ref_e_all[e]
            mu_local_all[e] = R_e.T @ mu_e_all[e]

        # energy[e] = mu_local_e^T k0 u_local_e (reduces to u^T k0 u when R_e=I)
        energy = np.einsum("ei,ij,ej->e", mu_local_all, k0_unit, u_local_all)
        dE_drho = (
            self.young_modulus
            * (1.0 - self.rho_min)
            * self.penalty
            * rho_flat ** (self.penalty - 1.0)
        )
        grad_flat = dE_drho * energy
        gradient = grad_flat.reshape(self.shape)

        displacement = u.reshape(*self._nn, self.ndim)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "compliance": compliance,
                "displacement": displacement,
                "nr_iterations": nr_iterations,
            },
            residual_norm=0.0,
            solver_iterations=nr_iterations,
        )
