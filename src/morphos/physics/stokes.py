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
``tests/test_stokes.py``.

Both 2D (quad Taylor-Hood, ``ElementQuad2``/``ElementQuad1``) and 3D
(hexahedral Taylor-Hood, ``ElementHex2``/``ElementHex1``) are supported on a
structured grid. The element type is the only thing that changes with
dimension: the Brinkman interpolation, the saddle-point assembly, the
dissipation objective, and the self-adjoint sensitivity are all written
dimension-generally. 3D follows the project rule (MEMORY.md) of vendoring the
hard inf-sup-stable saddle-point element rather than hand-rolling an unstable
equal-order Hex8 -- scikit-fem supplies the LBB-stable Hex Taylor-Hood pair.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Tuple

import numpy as np

from morphos.field import Field
from morphos.physics._linsolve import solve_linear
from morphos.physics.oracle import PhysicsOracle, PhysicsResult

# 2D edges (x: left/right, y: bottom/top) plus the 3D z faces (back/front).
_EDGES = ("left", "right", "top", "bottom")
_FACES = ("left", "right", "top", "bottom", "back", "front")


class StokesFlowOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, ...],
        inlet: Tuple[str, Callable[[np.ndarray], np.ndarray]],
        noslip_edges: Iterable[str] = (),
        viscosity: float = 1.0,
        alpha_max: float = 1e5,
        alpha_min: float = 0.0,
        brinkman_q: float = 0.1,
        solver: Literal["direct", "iterative", "auto"] = "auto",
    ) -> None:
        """Brinkman-Stokes fluid-TO oracle on a structured 2D quad or 3D hex grid.

        Parameters
        ----------
        shape:
            Number of *elements* per axis: ``(ny, nx)`` in 2D or
            ``(nz, ny, nx)`` in 3D (one design cell per element). The number of
            axes selects the dimension.
        inlet:
            ``(edge, profile)``. In 2D ``edge`` is one of ``"left"``,
            ``"right"``, ``"top"``, ``"bottom"`` and ``profile`` maps boundary
            coordinates ``x`` (shape ``(2, N)``) to a velocity field
            (shape ``(2, N)``). In 3D the two extra faces ``"back"`` (z=0) and
            ``"front"`` (z=Lz) are also available and ``profile`` works on
            ``(3, N)`` arrays.
        noslip_edges:
            Edges/faces held at zero velocity (walls). Boundaries that are
            neither the inlet nor a wall are natural outflow boundaries.
        viscosity:
            Dynamic viscosity ``mu``.
        alpha_max, alpha_min:
            Brinkman inverse-permeability at solid (``rho = 0``) and fluid
            (``rho = 1``).
        brinkman_q:
            Convexity parameter ``q`` of the Borrvall-Petersson interpolation.
        solver:
            ``"direct"``, ``"iterative"``, or ``"auto"`` (threshold-based;
            see :mod:`morphos.physics._linsolve`). The mixed velocity-pressure
            system is an indefinite saddle point, not SPD, so the iterative
            path is ILU-preconditioned GMRES rather than SA-AMG-preconditioned
            CG/GMRES (plain smoothed-aggregation AMG was checked and does not
            converge on the raw saddle-point operator; ILU+GMRES does, and
            matches the direct solve to about 1e-13 relative error).
        """
        try:
            import skfem  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "StokesFlowOracle needs scikit-fem; install the 'cfd' extra "
                "(pip install morphos[cfd])"
            ) from exc

        if len(shape) not in (2, 3):
            raise ValueError(
                "StokesFlowOracle supports 2D (ny, nx) or 3D (nz, ny, nx) grids"
            )
        if any(int(s) < 1 for s in shape):
            raise ValueError("grid must have at least one element per axis")
        self.ndim = len(shape)
        valid = _EDGES if self.ndim == 2 else _FACES
        inlet_edge, profile = inlet
        if inlet_edge not in valid:
            raise ValueError(f"inlet edge must be one of {valid}, got {inlet_edge!r}")
        noslip = tuple(noslip_edges)
        for e in noslip:
            if e not in valid:
                raise ValueError(f"noslip edge must be one of {valid}, got {e!r}")

        self.shape = tuple(int(s) for s in shape)
        self.inlet_edge = inlet_edge
        self.profile = profile
        self.noslip = noslip
        self.viscosity = float(viscosity)
        self.alpha_max = float(alpha_max)
        self.alpha_min = float(alpha_min)
        self.q = float(brinkman_q)
        self.solver = solver
        self._cache_h = None
        self._ilu_cache: list = []

    def _alpha(self, rho: np.ndarray) -> np.ndarray:
        q = self.q
        return self.alpha_max + (self.alpha_min - self.alpha_max) * rho * (1.0 + q) / (rho + q)

    def _alpha_grad(self, rho: np.ndarray) -> np.ndarray:
        q = self.q
        return (self.alpha_min - self.alpha_max) * q * (1.0 + q) / ((rho + q) ** 2)

    def _build(self, h: float) -> None:
        """Build the mesh, bases, constant operators, and the design-cell ->
        mesh-element permutation. Cached per spacing. Element type is selected
        by dimension (quad Taylor-Hood in 2D, hex Taylor-Hood in 3D); the forms
        and assembly are identical."""
        if self._cache_h == h:
            return
        from skfem import Basis, BilinearForm, ElementVector, asm
        from skfem.helpers import ddot, div, dot, grad

        # Axis lengths in physical units, ordered fastest-varying last to match
        # the row-major design field. self.shape is (ny, nx) or (nz, ny, nx).
        n_axes = self.shape[::-1]  # (nx, ny[, nz])
        self._L = [n * h for n in n_axes]  # (Lx, Ly[, Lz])
        coords = [np.linspace(0.0, L, n + 1) for L, n in zip(self._L, n_axes)]

        if self.ndim == 2:
            from skfem import ElementQuad0, ElementQuad1, ElementQuad2, MeshQuad

            mesh = MeshQuad.init_tensor(*coords)
            e_vel, e_pre, e_rho = ElementQuad2, ElementQuad1, ElementQuad0
        else:
            from skfem import ElementHex0, ElementHex1, ElementHex2, MeshHex

            mesh = MeshHex.init_tensor(*coords)
            e_vel, e_pre, e_rho = ElementHex2, ElementHex1, ElementHex0

        ub = Basis(mesh, ElementVector(e_vel()))
        pb = ub.with_element(e_pre())
        rb = ub.with_element(e_rho())  # rho per element, ub's quadrature

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
        self._tol = 1e-9 * max(self._L)

        # Dirichlet dofs and lifted inlet velocity values.
        dir_dofs, ud = self._dirichlet()
        self._dir_dofs, self._ud = dir_dofs, ud

        # Design-cell (row-major) -> mesh-element permutation, robust to
        # scikit-fem's internal element ordering.
        cent = mesh.p[:, mesh.t].mean(axis=1)  # (ndim, nelem)
        idx = [np.clip((cent[a] / h).astype(int), 0, n - 1) for a, n in enumerate(n_axes)]
        # Flatten row-major: (nz, ny, nx) -> ((iz*ny)+iy)*nx + ix.
        flat = idx[-1].copy()  # ix
        stride = n_axes[0]  # nx
        for a in range(1, self.ndim):
            flat = flat + idx[a] * stride
            stride *= n_axes[a]
        self._cell_of_elem = flat  # element e draws rho from this flat cell
        self._cache_h = h

    def _face_predicates(self):
        """Boundary predicates keyed by face name, valid for the current dim."""
        L, tol = self._L, self._tol
        preds = {
            "left": lambda x: x[0] < tol,
            "right": lambda x: x[0] > L[0] - tol,
            "bottom": lambda x: x[1] < tol,
            "top": lambda x: x[1] > L[1] - tol,
        }
        if self.ndim == 3:
            preds["back"] = lambda x: x[2] < tol
            preds["front"] = lambda x: x[2] > L[2] - tol
        return preds

    def _edge_facets(self, edge):
        return self._mesh.facets_satisfying(self._face_predicates()[edge])

    def _dirichlet(self):
        ub = self._ub
        inlet_facets = self._edge_facets(self.inlet_edge)
        inlet_dofs = ub.get_dofs(facets=inlet_facets).all()
        ud = ub.zeros()
        projected = ub.project(self.profile)
        ud[inlet_dofs] = projected[inlet_dofs]
        dir_dofs = [inlet_dofs]
        for edge in self.noslip:
            wall = ub.get_dofs(facets=self._edge_facets(edge)).all()
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
        # Indefinite saddle-point system: ILU+GMRES on the iterative path,
        # not SA-AMG+CG/GMRES (see __init__ docstring).
        dx, residual_norm, iterations = solve_linear(
            K[np.ix_(free, free)], rhs[free], solver=self.solver,
            dof_count=free.size, indefinite=True, cache_holder=self._ilu_cache,
        )
        x[free] += dx

        u = x[:NU]
        p = x[NU:]
        dissipation = float(u @ (A @ u))
        value = -dissipation

        # Self-adjoint sensitivity: per element, -alpha'(rho_e) * integral_e |u|^2.
        uq = ub.interpolate(u)  # components indexable; each (nelem, nqp)
        usq = sum(np.asarray(uq[c]) ** 2 for c in range(self.ndim))
        elem_energy = np.sum(usq * ub.dx, axis=1)  # (nelem,)
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
            residual_norm=residual_norm,
            solver_iterations=iterations,
        )

    def _velocity_grid(self, u: np.ndarray) -> np.ndarray:
        """Velocity sampled on the regular corner-node grid -- ``(nvy, nvx, 2)``
        in 2D or ``(nvz, nvy, nvx, 3)`` in 3D. The corners are the Quad2/Hex2
        vertex dofs (scikit-fem orders nodal/vertex dofs first), convenient for
        boundary checks and visualization."""
        ub, mesh = self._ub, self._mesh
        h = self._cache_h
        nodal = ub.nodal_dofs  # (ndim, n_vertices)
        pts = mesh.p  # (ndim, n_vertices)
        # Integer grid indices per axis, ordered to match (nz, ny, nx) layout.
        gidx = [np.rint(pts[a] / h).astype(int) for a in range(self.ndim)]
        out_shape = tuple(n + 1 for n in self.shape) + (self.ndim,)
        out = np.zeros(out_shape)
        # pts rows are (x, y[, z]); design axes are (..., y, x) -> reverse for index.
        index = tuple(gidx[self.ndim - 1 - a] for a in range(self.ndim))
        for c in range(self.ndim):
            out[index + (c,)] = u[nodal[c]]
        return out
