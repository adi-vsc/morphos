"""Conjugate heat transfer (CHT): steady advection-diffusion with a frozen
velocity field -- the engine's first oracle that couples a fluid result into a
thermal solve.

It solves, on a structured 2D quad or 3D hex grid,

    -div(k(rho) grad T) + rho_cp * (u . grad T) = Q,

where ``u`` is a *frozen* velocity field (e.g. produced by
:class:`~morphos.physics.stokes.StokesFlowOracle`), ``Q`` a volumetric heat
source on element centres, and the conductive material distribution ``rho`` is
the design variable through a SIMP interpolation

    k(rho) = k_fluid + rho**p * (k_solid - k_fluid).

This is the operator-split / staggered approach: the velocity is fixed during the
thermal solve (full monolithic coupling is deferred). The element-Peclet number
can be large, so the advection term is stabilised with SUPG (streamline-upwind
Petrov-Galerkin) -- a plain Galerkin advection term is unstable for Pe > 2.

Design choices that keep the adjoint exact and cheap (and are physically honest):

* The advection operator ``rho_cp * (u . grad)`` and the SUPG stabilisation
  depend only on the frozen velocity and the (constant) fluid heat capacity, not
  on the conductive design field ``rho``. The SUPG parameter ``tau`` uses a fixed
  reference diffusivity ``k_solid`` (stabilisation is a numerical device, not
  physics, so a reference-Pe tau is standard practice). Hence the only
  ``rho``-dependence is in the symmetric diffusion block, and the sensitivity is
  the same element-local SIMP form the other oracles use.
* The figure of merit is ``value = -(c . T)`` where ``c`` averages temperature
  over a set of objective nodes (default: all nodes). Minimising mean temperature
  -> maximising ``value``. The adjoint of the (non-symmetric) operator solves
  ``K^T lambda = c``; the gradient is ``+SIMP'(rho_e) * lambda_e^T Ke0 T_e``
  per element. FD-gated in ``tests/test_conjugate_heat.py``.

3D follows the same recipe with the trilinear Hex8 element
(:func:`morphos.physics.operators.hex8_diffusion_stiffness`) in place of Q4:
8 nodes and trilinear shape functions instead of 4 nodes and bilinear ones, one
extra velocity/gradient component, and a node grid ``(nelz+1, nely+1, nelx+1)``
flattened row-major (z slowest, x fastest) -- exactly the layout
:class:`~morphos.physics.stokes.StokesFlowOracle` uses for its 3D
``aux["velocity"]`` grid, so a Stokes result feeds straight into this oracle's
``velocity`` argument with no reshaping. Dimension is selected by
``len(shape)``; the assembly loop, SUPG stabilisation, and adjoint sensitivity
are written once per dimension but follow an identical structure.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple

import numpy as np
from scipy import sparse

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.operators import hex8_diffusion_stiffness, q4_diffusion_stiffness
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

# Reference-element node ordering, shared with operators.q4_diffusion_stiffness:
# node 0=(-1,-1), 1=(1,-1), 2=(1,1), 3=(-1,1).
_NODE_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_NODE_ETA = np.array([-1.0, -1.0, 1.0, 1.0])
_GP = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]

# Reference-element node ordering, shared with operators.hex8_diffusion_stiffness:
# bottom face (zeta=-1) nodes 0..3 counterclockwise, then top face (zeta=1)
# nodes 4..7 counterclockwise, matching hex8_stiffness's convention.
_NODE_XI_3D = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
_NODE_ETA_3D = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
_NODE_ZETA_3D = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
_GAUSS_3D = [(a, b, c) for a in (-_GP, _GP) for b in (-_GP, _GP) for c in (-_GP, _GP)]


def _shape_and_grads(h: float):
    """Per-Gauss-point shape values ``N`` (4,) and physical gradients ``G``
    (2, 4) for a square Q4 element of side ``h``. Returns lists over the 4
    quadrature points plus the common ``detJ``."""
    inv_j = 2.0 / h
    det_j = (h / 2.0) ** 2
    Ns, Gs = [], []
    for xi, eta in _GAUSS:
        N = 0.25 * (1.0 + _NODE_XI * xi) * (1.0 + _NODE_ETA * eta)
        dN_dxi = 0.25 * _NODE_XI * (1.0 + _NODE_ETA * eta)
        dN_deta = 0.25 * _NODE_ETA * (1.0 + _NODE_XI * xi)
        G = np.vstack([inv_j * dN_dxi, inv_j * dN_deta])  # (2, 4)
        Ns.append(N)
        Gs.append(G)
    return Ns, Gs, det_j


def _shape_and_grads_3d(h: float):
    """Per-Gauss-point shape values ``N`` (8,) and physical gradients ``G``
    (3, 8) for a cubic Hex8 element of side ``h``. Returns lists over the 8
    quadrature points plus the common ``detJ``."""
    inv_j = 2.0 / h
    det_j = (h / 2.0) ** 3
    Ns, Gs = [], []
    for xi, eta, zeta in _GAUSS_3D:
        N = (
            0.125
            * (1.0 + _NODE_XI_3D * xi)
            * (1.0 + _NODE_ETA_3D * eta)
            * (1.0 + _NODE_ZETA_3D * zeta)
        )
        dN_dxi = 0.125 * _NODE_XI_3D * (1.0 + _NODE_ETA_3D * eta) * (1.0 + _NODE_ZETA_3D * zeta)
        dN_deta = 0.125 * _NODE_ETA_3D * (1.0 + _NODE_XI_3D * xi) * (1.0 + _NODE_ZETA_3D * zeta)
        dN_dzeta = 0.125 * _NODE_ZETA_3D * (1.0 + _NODE_XI_3D * xi) * (1.0 + _NODE_ETA_3D * eta)
        G = np.vstack([inv_j * dN_dxi, inv_j * dN_deta, inv_j * dN_dzeta])  # (3, 8)
        Ns.append(N)
        Gs.append(G)
    return Ns, Gs, det_j


class ConjugateHeatOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        velocity: np.ndarray,
        source: np.ndarray,
        fixed_nodes: Iterable[int] = (),
        fixed_values: Iterable[float] = (),
        k_solid: float = 1.0,
        k_fluid: float = 1e-3,
        rho_cp: float = 1.0,
        p_simp: float = 3.0,
        objective_nodes: Optional[Iterable[int]] = None,
        solver: Literal["direct", "iterative", "auto"] = "auto",
    ) -> None:
        """Conjugate-heat oracle on a structured 2D quad or 3D hex element grid.

        Parameters
        ----------
        shape:
            ``(nely, nelx)`` in 2D or ``(nelz, nely, nelx)`` in 3D, number of
            elements per axis. Nodes form a grid one larger per axis. Dimension
            is selected by ``len(shape)``.
        velocity:
            Frozen velocity field on the node grid: shape
            ``(nely+1, nelx+1, 2)`` in 2D or ``(nelz+1, nely+1, nelx+1, 3)`` in
            3D -- exactly the ``aux["velocity"]`` grid returned by
            ``StokesFlowOracle`` in the matching dimension.
        source:
            Volumetric heat source ``Q`` per element, shape matching ``shape``.
        fixed_nodes, fixed_values:
            Dirichlet temperature BCs by flat node index (row-major over the
            node grid; in 3D, z slowest, x fastest).
        k_solid, k_fluid:
            Conductivity of the solid and fluid phase (SIMP endpoints).
        rho_cp:
            Fluid volumetric heat capacity ``rho*cp``.
        p_simp:
            SIMP penalisation exponent.
        objective_nodes:
            Node indices whose mean temperature is the figure of merit. Default
            is all nodes.
        solver:
            ``"direct"``, ``"iterative"`` (AMG-preconditioned GMRES; the
            convection-diffusion operator is non-symmetric, so CG is not
            valid here), or ``"auto"`` (threshold-based; see
            :mod:`morphos.physics._linsolve`). The forward solve and its
            transposed adjoint solve share the same sparsity pattern (one is
            the transpose of the other) but get independent AMG caches,
            since SA-AMG is built from the matrix as given and a transpose
            is a different matrix to the solver.
        """
        if len(shape) not in (2, 3):
            raise ValueError(
                "ConjugateHeatOracle supports 2D (nely, nelx) or "
                "3D (nelz, nely, nelx) grids"
            )
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        self.ndim = len(shape)
        self.shape = tuple(int(s) for s in shape)

        if self.ndim == 2:
            nely, nelx = self.shape
            self.nny, self.nnx = nely + 1, nelx + 1
            self.n_nodes = self.nny * self.nnx
            vel_shape = (self.nny, self.nnx, 2)
        else:
            nelz, nely, nelx = self.shape
            self.nnz, self.nny, self.nnx = nelz + 1, nely + 1, nelx + 1
            self.n_nodes = self.nnz * self.nny * self.nnx
            vel_shape = (self.nnz, self.nny, self.nnx, 3)

        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != vel_shape:
            raise ValueError(
                f"velocity shape {velocity.shape} must be {vel_shape} "
                f"(node grid x {self.ndim})"
            )
        self.velocity = velocity

        source = np.asarray(source, dtype=float)
        if source.shape != self.shape:
            raise ValueError(f"source shape {source.shape} must match {self.shape}")
        self.source = source

        self.k_solid = float(k_solid)
        self.k_fluid = float(k_fluid)
        self.rho_cp = float(rho_cp)
        self.p = float(p_simp)

        fixed_nodes = np.asarray(list(fixed_nodes), dtype=int)
        fixed_values = np.asarray(list(fixed_values), dtype=float)
        if fixed_nodes.shape != fixed_values.shape:
            raise ValueError("fixed_nodes and fixed_values must have equal length")
        self.fixed_nodes = fixed_nodes
        self.fixed_values = fixed_values

        if objective_nodes is None:
            obj = np.arange(self.n_nodes)
        else:
            obj = np.asarray(list(objective_nodes), dtype=int)
        c = np.zeros(self.n_nodes)
        if obj.size:
            c[obj] = 1.0 / obj.size
        self._c = c  # averaging vector over objective nodes

        self._elem_nodes = self._element_node_table()
        self._cache_h = None
        self.solver = solver
        self._amg_cache_fwd: list = []
        self._amg_cache_adj: list = []

    def _element_node_table(self) -> np.ndarray:
        """(nelem, nodes_per_elem) global node indices per element, in the
        reference node order shared with operators.q4/hex8_diffusion_stiffness."""
        if self.ndim == 2:
            nely, nelx = self.shape
            nnx = self.nnx
            table = np.zeros((nely * nelx, 4), dtype=int)
            e = 0
            for iy in range(nely):
                for ix in range(nelx):
                    n0 = iy * nnx + ix
                    n1 = iy * nnx + (ix + 1)
                    n2 = (iy + 1) * nnx + (ix + 1)
                    n3 = (iy + 1) * nnx + ix
                    table[e] = (n0, n1, n2, n3)
                    e += 1
            return table

        nelz, nely, nelx = self.shape
        nny, nnx = self.nny, self.nnx
        table = np.zeros((nelz * nely * nelx, 8), dtype=int)
        e = 0
        for iz in range(nelz):
            for iy in range(nely):
                for ix in range(nelx):
                    n0 = (iz * nny + iy) * nnx + ix
                    n1 = (iz * nny + iy) * nnx + (ix + 1)
                    n2 = (iz * nny + (iy + 1)) * nnx + (ix + 1)
                    n3 = (iz * nny + (iy + 1)) * nnx + ix
                    n4 = ((iz + 1) * nny + iy) * nnx + ix
                    n5 = ((iz + 1) * nny + iy) * nnx + (ix + 1)
                    n6 = ((iz + 1) * nny + (iy + 1)) * nnx + (ix + 1)
                    n7 = ((iz + 1) * nny + (iy + 1)) * nnx + ix
                    table[e] = (n0, n1, n2, n3, n4, n5, n6, n7)
                    e += 1
        return table

    def _simp(self, rho: np.ndarray) -> np.ndarray:
        return self.k_fluid + rho ** self.p * (self.k_solid - self.k_fluid)

    def _simp_grad(self, rho: np.ndarray) -> np.ndarray:
        return self.p * rho ** (self.p - 1.0) * (self.k_solid - self.k_fluid)

    def _build_constant(self, h: float) -> None:
        """Assemble the rho-independent pieces (unit diffusion stiffness, the
        convection + SUPG operator, and the source load) once per spacing."""
        if self._cache_h == h:
            return
        if self.ndim == 2:
            Ns, Gs, det_j = _shape_and_grads(h)
            self._Ke0 = q4_diffusion_stiffness(1.0, h)  # unit-conductivity element
            npe = 4
            vol_per_node = (h * h) / 4.0
        else:
            Ns, Gs, det_j = _shape_and_grads_3d(h)
            self._Ke0 = hex8_diffusion_stiffness(1.0, h)
            npe = 8
            vol_per_node = (h ** 3) / 8.0

        nelem = self._elem_nodes.shape[0]
        vel_flat = self.velocity.reshape(self.n_nodes, self.ndim)
        Q = self.source.ravel()

        # Convection + SUPG assembled into a global non-symmetric operator, and
        # the (rho-independent) load including the SUPG source contribution.
        rows, cols, vals = [], [], []
        f = np.zeros(self.n_nodes)
        for e in range(nelem):
            nodes = self._elem_nodes[e]
            u_nodes = vel_flat[nodes]  # (npe, ndim)
            u_mean = u_nodes.mean(axis=0)
            speed = float(np.linalg.norm(u_mean))
            tau = self._supg_tau(speed, h)

            Ce = np.zeros((npe, npe))
            Se = np.zeros((npe, npe))
            fe_supg = np.zeros(npe)
            for N, G in zip(Ns, Gs):
                u_gp = N @ u_nodes  # (ndim,) velocity at the Gauss point
                uG = u_gp @ G  # (npe,) = u . grad(N_j)
                Ce += self.rho_cp * np.outer(N, uG) * det_j
                Se += tau * self.rho_cp ** 2 * np.outer(uG, uG) * det_j
                fe_supg += tau * self.rho_cp * uG * (Q[e]) * det_j
            Ae = Ce + Se
            # Galerkin source: Q_e * integral(N_i) per node (uniform for the
            # square/cube reference element: element volume / nodes_per_elem).
            fe = Q[e] * vol_per_node * np.ones(npe) + fe_supg
            for a in range(npe):
                f[nodes[a]] += fe[a]
                for b in range(npe):
                    rows.append(nodes[a])
                    cols.append(nodes[b])
                    vals.append(Ae[a, b])
        self._K_conv = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )
        self._f = f
        self._cache_h = h

    def _supg_tau(self, speed: float, h: float) -> float:
        """SUPG stabilisation parameter using a reference (solid) diffusivity so
        it is independent of the design field. tau = (h/2|u|)(coth Pe - 1/Pe),
        Pe = |u| h / (2 k_ref); nodally exact for the 1D model problem."""
        if speed < 1e-30:
            return 0.0
        k_ref = max(self.k_solid, 1e-30)
        Pe = speed * h / (2.0 * k_ref)
        if Pe < 1e-6:
            xi = Pe / 3.0
        else:
            xi = 1.0 / np.tanh(Pe) - 1.0 / Pe
        return (h / (2.0 * speed)) * xi

    def _assemble_diffusion(self, rho_elem: np.ndarray) -> sparse.csr_matrix:
        """SIMP-scaled global diffusion stiffness for the current design."""
        k_e = self._simp(rho_elem)
        nelem = rho_elem.size
        npe = self._Ke0.shape[0]
        rows, cols, vals = [], [], []
        for e in range(nelem):
            nodes = self._elem_nodes[e]
            Ke = k_e[e] * self._Ke0
            for a in range(npe):
                for b in range(npe):
                    rows.append(nodes[a])
                    cols.append(nodes[b])
                    vals.append(Ke[a, b])
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (element grid)"
            )
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("ConjugateHeatOracle assumes isotropic spacing")
        h = spacing[0]
        self._build_constant(h)

        rho = np.clip(field.values, 0.0, 1.0).ravel()
        K = (self._assemble_diffusion(rho) + self._K_conv).tocsr()
        f = self._f.copy()

        all_nodes = np.arange(self.n_nodes)
        fixed = self.fixed_nodes
        free = np.setdiff1d(all_nodes, fixed)

        T = np.zeros(self.n_nodes)
        if fixed.size:
            T[fixed] = self.fixed_values
        # Reduced free-dof system with Dirichlet lift.
        Kff = K[np.ix_(free, free)].tocsc()
        rhs = f[free]
        if fixed.size:
            rhs = rhs - K[np.ix_(free, fixed)] @ self.fixed_values
        # Non-symmetric (advection-coupled) operator: GMRES, not CG, on the
        # iterative path.
        Tf, res_fwd, it_fwd = solve_linear(
            Kff, rhs, solver=self.solver, dof_count=free.size,
            symmetric=False, cache_holder=self._amg_cache_fwd,
        )
        T[free] = Tf

        Jt = float(self._c @ T)  # mean temperature over objective nodes
        value = -Jt

        # Adjoint of the non-symmetric operator: K^T lambda = c on free dofs,
        # lambda = 0 on Dirichlet dofs. K^T has a different sparsity pattern
        # ordering than K (it is the transpose), so it gets its own AMG cache.
        lam = np.zeros(self.n_nodes)
        KffT = K[np.ix_(free, free)].T.tocsc()
        lamf, res_adj, it_adj = solve_linear(
            KffT, self._c[free], solver=self.solver, dof_count=free.size,
            symmetric=False, cache_holder=self._amg_cache_adj,
        )
        lam[free] = lamf
        residual_norm = max(res_fwd, res_adj)
        iterations = max(it_fwd, it_adj)

        # Element-local SIMP sensitivity of J = c.T : dJ/drho_e =
        #   -lambda_e^T (dK_diff/drho_e) T_e = -SIMP'(rho_e) lambda_e^T Ke0 T_e.
        simp_grad = self._simp_grad(rho)
        grad_elem = np.zeros(rho.size)
        for e in range(rho.size):
            nodes = self._elem_nodes[e]
            Te = T[nodes]
            le = lam[nodes]
            grad_elem[e] = -simp_grad[e] * (le @ (self._Ke0 @ Te))
        # value = -J, so d(value)/drho = -dJ/drho.
        gradient = (-grad_elem).reshape(self.shape)

        node_shape = (self.nny, self.nnx) if self.ndim == 2 else (
            self.nnz, self.nny, self.nnx
        )
        T_grid = T.reshape(node_shape)
        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={"temperature": T_grid, "mean_temperature": Jt},
            residual_norm=residual_norm,
            solver_iterations=iterations,
        )
