"""A real CFD physics backend: density-dependent Brinkman-Stokes flow for
fluid topology optimization -- the engine's first genuine viscous-flow oracle,
a step up from the Darcy potential-flow surrogate in :mod:`morphos.physics.darcy`
to incompressible Stokes with a true velocity field and viscous stress.

Unlike the other physics oracles, this one does not hand-roll the finite-element
assembly: an inf-sup-stable mixed (Taylor-Hood) Stokes element is exactly the
kind of specialised, error-prone solver where the project leans on a vetted
library (the same call the EM backend makes with ``ceviche``). It vendors
``scikit-fem`` (a lightweight pure-Python FEM, numpy/scipy only) for the
biquadratic-velocity / bilinear-pressure assembly on a structured quad mesh, and
keeps the SIMP density coupling, the dissipation objective, and the FD-gated
sensitivity in morphos.

Formulation (Borrvall & Petersson 2003). Incompressible Stokes with a
density-dependent Brinkman (inverse-permeability) term that drives the velocity
to zero inside solid material:

    -mu laplacian(u) + grad(p) + alpha(rho) u = 0,    div(u) = 0,

    alpha(rho) = alpha_max + (alpha_min - alpha_max) * rho (1 + q) / (rho + q)

so ``rho = 1`` is open fluid (``alpha -> alpha_min ~ 0``) and ``rho = 0`` is an
impermeable wall (``alpha -> alpha_max``). Flow is driven by a prescribed inlet
velocity profile; walls are no-slip; the outlet is a natural (do-nothing)
boundary and one pressure dof is pinned for uniqueness.

Figure of merit. The total power dissipation

    Phi = integral( mu |grad u|^2 + alpha(rho) |u|^2 ) = u^T A(rho) u,

minimised by the optimizer (so ``value = -Phi``, mirroring the Darcy oracle).
This functional is *self-adjoint*: because the divergence-free constraint holds
for every ``rho`` (``B du/drho = 0``) and there is no rho-dependent forcing, the
implicit ``du/drho`` terms cancel and the sensitivity is the explicit Brinkman
term alone,

    d(value)/drho_e = - alpha'(rho_e) * integral_e |u|^2,

no co-state solve required (exactly as compliance is self-adjoint for
elasticity). Verified against a central-FD directional gate in
``tests/test_stokes.py``. Scoped to 2D; the same construction extends to 3D
Hex/Taylor-Hood and is left as a follow-up.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

_EDGES = ("left", "right", "top", "bottom")


class StokesFlowOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, int],
        inlet: Tuple[str, Callable[[np.ndarray], np.ndarray]],
        noslip_edges: Iterable[str] = (),
        viscosity: float = 1.0,
        alpha_max: float = 1e5,
        alpha_min: float = 0.0,
        brinkman_q: float = 0.1,
    ) -> None:
        """Brinkman-Stokes fluid-TO oracle on a 2D structured quad grid.

        Parameters
        ----------
        shape:
            Number of *elements* per axis ``(ny, nx)`` (one design cell per
            quad element).
        inlet:
            ``(edge, profile)`` where ``edge`` is one of ``"left"``,
            ``"right"``, ``"top"``, ``"bottom"`` and ``profile`` maps an array
            of boundary coordinates ``x`` (shape ``(2, N)``) to a velocity field
            (shape ``(2, N)``).
        noslip_edges:
            Edges held at zero velocity (walls). Edges that are neither the
            inlet nor a wall are natural outflow boundaries.
        viscosity:
            Dynamic viscosity ``mu``.
        alpha_max, alpha_min:
            Brinkman inverse-permeability at solid (``rho = 0``) and fluid
            (``rho = 1``).
        brinkman_q:
            Convexity parameter ``q`` of the Borrvall-Petersson interpolation.
        """
        try:
            import skfem  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "StokesFlowOracle needs scikit-fem; install the 'cfd' extra "
                "(pip install morphos[cfd])"
            ) from exc

        if len(shape) != 2:
            raise ValueError("StokesFlowOracle is 2D: shape must be (ny, nx)")
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        inlet_edge, profile = inlet
        if inlet_edge not in _EDGES:
            raise ValueError(f"inlet edge must be one of {_EDGES}, got {inlet_edge!r}")
        noslip = tuple(noslip_edges)
        for e in noslip:
            if e not in _EDGES:
                raise ValueError(f"noslip edge must be one of {_EDGES}, got {e!r}")

        self.shape = (int(shape[0]), int(shape[1]))
        self.inlet_edge = inlet_edge
        self.profile = profile
        self.noslip = noslip
        self.viscosity = float(viscosity)
        self.alpha_max = float(alpha_max)
        self.alpha_min = float(alpha_min)
        self.q = float(brinkman_q)
        self._cache_h = None

    def _alpha(self, rho: np.ndarray) -> np.ndarray:
        q = self.q
        return self.alpha_max + (self.alpha_min - self.alpha_max) * rho * (1.0 + q) / (rho + q)

    def _alpha_grad(self, rho: np.ndarray) -> np.ndarray:
        q = self.q
        return (self.alpha_min - self.alpha_max) * q * (1.0 + q) / ((rho + q) ** 2)

    def _build(self, h: float) -> None:
        """Build the mesh, bases, constant operators, and the design-cell ->
        mesh-element permutation. Cached per spacing."""
        if self._cache_h == h:
            return
        import skfem
        from skfem import (
            Basis, BilinearForm, ElementQuad0, ElementQuad1, ElementQuad2,
            ElementVector, MeshQuad, asm,
        )
        from skfem.helpers import ddot, div, dot, grad

        ny, nx = self.shape
        Lx, Ly = nx * h, ny * h
        mesh = MeshQuad.init_tensor(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))
        ub = Basis(mesh, ElementVector(ElementQuad2()))
        pb = ub.with_element(ElementQuad1())
        rb = ub.with_element(ElementQuad0())  # rho per element, ub's quadrature

        @BilinearForm
        def viscous(u, v, w):
            return self.viscosity * ddot(grad(u), grad(v))

        @BilinearForm
        def brinkman(u, v, w):
            return w["a"] * dot(u, v)

        @BilinearForm
        def bdiv(u, p, w):
            return -div(u) * p

        self._mesh, self._ub, self._pb, self._rb = mesh, ub, pb, rb
        self._brinkman, self._viscous_form = brinkman, viscous
        self._asm = asm
        self._A_visc = asm(viscous, ub)
        self._B = asm(bdiv, ub, pb)
        self._tol = 1e-9 * max(Lx, Ly)

        # Dirichlet dofs and lifted inlet velocity values.
        dir_dofs, ud = self._dirichlet(Lx, Ly)
        self._dir_dofs, self._ud = dir_dofs, ud

        # Design-cell (row-major (iy, ix)) -> mesh-element permutation, robust to
        # scikit-fem's internal element ordering.
        cent = mesh.p[:, mesh.t].mean(axis=1)  # (2, nelem)
        ix = np.clip((cent[0] / h).astype(int), 0, nx - 1)
        iy = np.clip((cent[1] / h).astype(int), 0, ny - 1)
        self._cell_of_elem = iy * nx + ix  # element e draws rho from this flat cell
        self._cache_h = h

    def _edge_facets(self, edge, Lx, Ly):
        mesh, tol = self._mesh, self._tol
        pred = {
            "left": lambda x: x[0] < tol,
            "right": lambda x: x[0] > Lx - tol,
            "bottom": lambda x: x[1] < tol,
            "top": lambda x: x[1] > Ly - tol,
        }[edge]
        return mesh.facets_satisfying(pred)

    def _dirichlet(self, Lx, Ly):
        ub = self._ub
        inlet_facets = self._edge_facets(self.inlet_edge, Lx, Ly)
        inlet_dofs = ub.get_dofs(facets=inlet_facets).all()
        ud = ub.zeros()
        projected = ub.project(self.profile)
        ud[inlet_dofs] = projected[inlet_dofs]
        dir_dofs = [inlet_dofs]
        for edge in self.noslip:
            wall = ub.get_dofs(facets=self._edge_facets(edge, Lx, Ly)).all()
            dir_dofs.append(wall)
            ud[wall] = 0.0  # no-slip wins at inlet/wall corner nodes
        return np.unique(np.concatenate(dir_dofs)), ud

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle "
                f"shape {self.shape} (ny, nx elements)"
            )
        spacing = field.spacing
        if max(spacing) - min(spacing) > 1e-12:
            raise ValueError("StokesFlowOracle assumes isotropic spacing")
        h = spacing[0]
        self._build(h)

        from scipy.sparse import bmat
        from scipy.sparse.linalg import spsolve

        ub, pb, rb = self._ub, self._pb, self._rb
        rho = np.clip(field.values, 0.0, 1.0)
        rho_flat = rho.ravel()  # row-major cell index iy*nx+ix
        rho_elem = rho_flat[self._cell_of_elem]

        A = self._A_visc + self._asm(
            self._brinkman, ub, a=rb.interpolate(self._alpha(rho_elem))
        )
        B = self._B
        NU, NP = ub.N, pb.N
        K = bmat([[A, B.T], [B, None]]).tocsr()

        x = np.zeros(NU + NP)
        x[:NU] = self._ud
        pin = NU + NP - 1  # pin one pressure dof for uniqueness
        dofs = np.concatenate([self._dir_dofs, [pin]])
        free = np.setdiff1d(np.arange(NU + NP), dofs)
        rhs = -(K @ x)
        x[free] += spsolve(K[np.ix_(free, free)].tocsc(), rhs[free])

        u = x[:NU]
        p = x[NU:]
        dissipation = float(u @ (A @ u))
        value = -dissipation

        # Self-adjoint sensitivity: per element, -alpha'(rho_e) * integral_e |u|^2.
        uq = ub.interpolate(u)  # components indexable; each (nelem, nqp)
        u0, u1 = np.asarray(uq[0]), np.asarray(uq[1])
        elem_energy = np.sum((u0 ** 2 + u1 ** 2) * ub.dx, axis=1)  # (nelem,)
        grad_elem = -self._alpha_grad(rho_elem) * elem_energy
        grad_flat = np.zeros(rho_flat.size)
        np.add.at(grad_flat, self._cell_of_elem, grad_elem)
        gradient = grad_flat.reshape(self.shape)

        return PhysicsResult(
            value=value,
            gradient=gradient,
            aux={
                "dissipation": dissipation,
                "velocity": self._velocity_grid(u),
                "pressure": p,
            },
        )

    def _velocity_grid(self, u: np.ndarray) -> np.ndarray:
        """Velocity sampled on the regular (ny+1, nx+1) corner-node grid as
        ``(nvy, nvx, 2)`` -- the corners are the Quad2 vertex dofs, convenient
        for boundary checks and visualization."""
        ub, mesh = self._ub, self._mesh
        ny, nx = self.shape
        h = self._cache_h
        out = np.zeros((ny + 1, nx + 1, 2))
        # Vertex dofs: scikit-fem orders nodal (vertex) dofs first for Quad2.
        nodal = ub.nodal_dofs  # (2, n_vertices)
        pts = mesh.p  # (2, n_vertices)
        ix = np.rint(pts[0] / h).astype(int)
        iy = np.rint(pts[1] / h).astype(int)
        out[iy, ix, 0] = u[nodal[0]]
        out[iy, ix, 1] = u[nodal[1]]
        return out
