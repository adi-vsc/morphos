"""Independent reference solutions for the validation suite.

Every function here computes a quantity **without** calling the Morphos oracle
it will be compared against. Most are closed-form (assembled from the governing
equation); one is an independent finite-element solve in scikit-fem. Each
carries, in its docstring, the exact equation/paper/solver it comes from so the
reference is auditable.

Conventions match the grid Morphos uses: a scalar field sampled on a uniform
Cartesian node grid of spacing ``h``; element grids are one cell smaller than
the node grid on every axis.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


# ==========================================================================
# Closed-form references
# ==========================================================================

def heat_mms(n_nodes: int, length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Manufactured solution for steady conduction ``-lap T = s`` on a square.

    Domain ``[0, L]^2`` with homogeneous Dirichlet ``T = 0`` on the boundary.
    Choosing ``T(x,y) = sin(pi x / L) sin(pi y / L)`` makes

        -lap T = (2 pi^2 / L^2) sin(pi x/L) sin(pi y/L) =: s,

    an exact solution of the same PDE the ``HeatConductionOracle`` discretises
    (it solves the SPD interior negative-Laplacian against a nodal source, with
    unit conductivity). This is the classic method of manufactured solutions
    (Roache, *Verification and Validation in Computational Science*, 1998).

    Returns ``(source, T_exact, h)`` on the full ``n_nodes x n_nodes`` grid.
    """
    h = length / (n_nodes - 1)
    xs = np.linspace(0.0, length, n_nodes)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    T_exact = np.sin(np.pi * X / length) * np.sin(np.pi * Y / length)
    source = (2.0 * np.pi ** 2 / length ** 2) * T_exact
    return source, T_exact, h


def membrane_fundamental_eigenvalue(lx: float, ly: float) -> float:
    """Fundamental eigenvalue of a clamped rectangular membrane.

    ``-lap phi = lambda phi`` on ``[0,Lx] x [0,Ly]`` with ``phi = 0`` on the
    boundary has eigenvalues ``lambda_mn = pi^2 (m^2/Lx^2 + n^2/Ly^2)``
    (separation of variables; e.g. Strang, *Computational Science and
    Engineering*, ch. 6). The fundamental is ``(m,n) = (1,1)``. This is the
    continuum limit the ``ModalOracle`` (which solves ``K phi = lambda M phi``
    with ``K`` the discrete negative-Laplacian and ``M = diag(rho)``)
    converges to as the mesh is refined.
    """
    return np.pi ** 2 * (1.0 / lx ** 2 + 1.0 / ly ** 2)


def euler_bernoulli_tip_deflection(
    P: float, L: float, H: float, E: float, thickness: float = 1.0
) -> float:
    """Tip deflection of an end-loaded cantilever, Euler-Bernoulli theory.

    A cantilever of length ``L`` and rectangular cross-section ``H x thickness``,
    fixed at one end, transverse point load ``P`` at the free tip:

        delta = P L^3 / (3 E I),    I = thickness * H^3 / 12

    (Euler-Bernoulli beam theory; e.g. Gere & Goodno, *Mechanics of
    Materials*). Sign follows the load: a downward ``P < 0`` gives a downward
    (negative) deflection. EB neglects transverse shear, so for a stocky beam
    the true plane-stress deflection is slightly larger in magnitude
    (Timoshenko correction below).
    """
    I = thickness * H ** 3 / 12.0
    return P * L ** 3 / (3.0 * E * I)


def timoshenko_tip_deflection(
    P: float, L: float, H: float, E: float, nu: float, thickness: float = 1.0,
    kappa: float = 5.0 / 6.0,
) -> float:
    """Tip deflection including first-order shear (Timoshenko beam).

    ``delta = P L^3/(3 E I) + P L/(kappa G A)`` with ``G = E/2(1+nu)``,
    ``A = H*thickness`` and shear coefficient ``kappa = 5/6`` for a rectangle
    (Cowper, 1966). This is the closer continuum target a 2D plane-stress FE
    cantilever actually converges to; reported alongside EB to explain the
    residual gap.
    """
    I = thickness * H ** 3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    A = H * thickness
    return P * L ** 3 / (3.0 * E * I) + P * L / (kappa * G * A)


def axial_bar_tip_displacement(P: float, L: float, H: float, E: float, thickness: float = 1.0) -> float:
    """Tip displacement of a uniform bar in pure tension: ``delta = P L /(A E)``.

    Hooke's law for a prismatic bar, ``A = H * thickness`` (e.g. Gere &
    Goodno). A bilinear-quad plane-stress element reproduces the resulting
    linear (constant-strain) displacement field *exactly*, so this is the
    machine-precision sanity anchor for the elasticity assembly.
    """
    A = H * thickness
    return P * L / (A * E)


def darcy_column_pressure_drop(Q: float, L: float, A: float, K: float) -> float:
    """Pressure drop across a uniform porous column, Darcy's law.

    For 1D flow ``q = -K dp/dx`` with total volumetric flow ``Q`` through
    cross-section ``A`` over length ``L`` and uniform permeability ``K``:

        dp = Q L / (K A)

    (Darcy 1856; the potential-flow / resistor analogue the
    ``DarcyFlowOracle`` discretises). A Q4 diffusion element reproduces the
    linear pressure field exactly, so this is also machine-precision.
    """
    return Q * L / (K * A)


def poiseuille_profile(y: np.ndarray, H: float, u_max: float = 1.0) -> np.ndarray:
    """Plane-Poiseuille velocity profile ``u(y) = u_max * 4 y (H - y)/H^2``.

    The fully-developed Stokes solution between no-slip walls at ``y = 0`` and
    ``y = H`` (parabolic, zero at the walls, peak ``u_max`` at mid-channel;
    e.g. Batchelor, *An Introduction to Fluid Dynamics*, §4.2). For a straight
    channel driven by this profile at the inlet the same parabola holds at
    every cross-section, and it lies exactly in the biquadratic Taylor-Hood
    velocity space the ``StokesFlowOracle`` uses.
    """
    return u_max * 4.0 * y * (H - y) / H ** 2


# ==========================================================================
# Topology-optimization literature values
# ==========================================================================

# Andreassen, Clausen, Schury, Lazarov, Sigmund (2011), "Efficient topology
# optimization in MATLAB using 88 lines of code", Struct. Multidisc. Optim.
# 43:1-16. The half-MBB beam top88(60, 20, 0.5, 3.0, 1.5) — density filter,
# E0=1, Emin=1e-9, nu=0.3, unit point load — converges to a compliance of
# about 205 (the value the 88-line code prints for these inputs; figures in
# the 196-211 band appear across the SIMP literature depending on the filter
# variant and convergence criterion). Used as a published reference compliance.
MBB_LITERATURE_COMPLIANCE = 205.0
MBB_LITERATURE_SOURCE = (
    "Andreassen et al. (2011), '88-line' SIMP MBB beam top88(60,20,0.5,3,1.5), "
    "converged compliance ~205 (E0=1, nu=0.3, unit load, density filter)"
)


# ==========================================================================
# Independent FEM reference (scikit-fem) — used where two FE codes should agree
# ==========================================================================

def skfem_available() -> bool:
    try:
        import skfem  # noqa: F401
        return True
    except Exception:
        return False


def skfem_version() -> str:
    import skfem
    return skfem.__version__


def skfem_cantilever_compliance(
    nx: int, ny: int, L: float, H: float, E: float = 1.0, nu: float = 0.3, P: float = 1.0
) -> Tuple[float, float]:
    """Independent plane-stress cantilever solve in scikit-fem.

    Assembles linear elasticity on a structured ``MeshQuad`` with bilinear
    vector elements (``ElementVector(ElementQuad1)``) and the *plane-stress*
    constitutive law

        sigma = 2 mu eps + lam_ps tr(eps) I,
        mu = E/2(1+nu),  lam_ps = E nu/(1 - nu^2),

    fixes the entire left edge, and distributes a total transverse load ``-P``
    over the right-edge nodes (half weight at the two corners). Returns
    ``(compliance, tip_deflection)`` where compliance is ``f . u`` and the tip
    deflection is the mean vertical displacement of the right edge.

    This is an *independent* numerical reference: same continuum problem and
    same element family as ``ElasticityOracle``, assembled by a separate
    library (scikit-fem), so the two compliances must agree to solver
    tolerance. None of Morphos's assembly is used here.
    """
    from skfem import (
        MeshQuad, Basis, ElementVector, ElementQuad1, BilinearForm, asm,
        condense, solve,
    )
    from skfem.helpers import sym_grad, ddot, trace

    mesh = MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1), np.linspace(0.0, H, ny + 1))
    basis = Basis(mesh, ElementVector(ElementQuad1()))

    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu ** 2)

    @BilinearForm
    def stiffness(u, v, w):
        eu = sym_grad(u)
        ev = sym_grad(v)
        return 2.0 * mu * ddot(eu, ev) + lam_ps * trace(eu) * trace(ev)

    K = asm(stiffness, basis)
    left = basis.get_dofs(lambda x: x[0] < 1e-9).all()

    y_dof_of_node = basis.nodal_dofs[1]          # vertical dof per node
    coords = mesh.p                               # (2, n_nodes)
    right_nodes = np.where(coords[0] > L - 1e-9)[0]
    yc = coords[1, right_nodes]
    weight = np.where((yc < 1e-9) | (yc > H - 1e-9), 0.5, 1.0)
    rdofs = y_dof_of_node[right_nodes]

    f = np.zeros(K.shape[0])
    f[rdofs] = -P * weight / ny                   # total load -P over the edge

    u = solve(*condense(K, f, D=left))
    compliance = float(f @ u)
    tip = float(u[rdofs].mean())
    return compliance, tip


# ==========================================================================
# 3D references
# ==========================================================================

def heat_mms_3d(n_nodes: int, length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """3D manufactured solution for steady conduction ``-lap T = s`` on a cube.

    Domain ``[0, L]^3`` with homogeneous Dirichlet ``T = 0`` on the boundary.
    Choosing ``T = sin(pi x/L) sin(pi y/L) sin(pi z/L)`` makes

        -lap T = (3 pi^2 / L^2) sin(pi x/L) sin(pi y/L) sin(pi z/L) =: s,

    an exact solution of the same PDE the 3D ``HeatConductionOracle``
    discretises (7-point Laplacian, unit conductivity, zero boundary). Method
    of manufactured solutions (Roache 1998), the 3D analogue of :func:`heat_mms`.

    Returns ``(source, T_exact, h)`` on the full ``n_nodes^3`` grid.
    """
    h = length / (n_nodes - 1)
    xs = np.linspace(0.0, length, n_nodes)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    T_exact = (
        np.sin(np.pi * X / length)
        * np.sin(np.pi * Y / length)
        * np.sin(np.pi * Z / length)
    )
    source = (3.0 * np.pi ** 2 / length ** 2) * T_exact
    return source, T_exact, h


def skfem_hex_cantilever_compliance(
    nx: int, ny: int, nz: int, L: float, H: float, W: float,
    E: float = 1.0, nu: float = 0.3, P: float = 1.0,
) -> Tuple[float, float]:
    """Independent 3D linear-elasticity cantilever solve in scikit-fem.

    Assembles isotropic 3D elasticity on a structured ``MeshHex`` with trilinear
    vector elements (``ElementVector(ElementHex1)``) and the full 3D Hooke law

        sigma = 2 mu eps + lam tr(eps) I,
        mu = E/2(1+nu),  lam = E nu/((1+nu)(1-2nu)),

    clamps the entire ``x = 0`` face, and distributes a total transverse load
    ``-P`` (in ``y``) over the ``x = L`` face nodes with tributary weighting
    (corner 1/4, edge 1/2, interior 1). Returns ``(compliance, tip_deflection)``
    with compliance ``f . u`` and tip the mean y-displacement of the loaded face.

    This is an *independent* numerical reference for the Hex8
    ``ElasticityOracle``: identical continuum problem, element family, mesh,
    material and nodal loads, assembled by a separate library, so the two
    compliances must agree to linear-solver tolerance. None of Morphos's
    assembly is used here.
    """
    from skfem import (
        MeshHex, Basis, ElementVector, ElementHex1, BilinearForm, asm,
        condense, solve,
    )
    from skfem.helpers import sym_grad, ddot, trace

    mesh = MeshHex.init_tensor(
        np.linspace(0.0, L, nx + 1),
        np.linspace(0.0, H, ny + 1),
        np.linspace(0.0, W, nz + 1),
    )
    basis = Basis(mesh, ElementVector(ElementHex1()))

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @BilinearForm
    def stiffness(u, v, w):
        eu = sym_grad(u)
        ev = sym_grad(v)
        return 2.0 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

    K = asm(stiffness, basis)
    left = basis.get_dofs(lambda x: x[0] < 1e-9).all()

    y_dof_of_node = basis.nodal_dofs[1]           # vertical (y) dof per node
    coords = mesh.p                                # (3, n_nodes)
    right_nodes = np.where(coords[0] > L - 1e-9)[0]
    yc = coords[1, right_nodes]
    zc = coords[2, right_nodes]
    wy = np.where((yc < 1e-9) | (yc > H - 1e-9), 0.5, 1.0)
    wz = np.where((zc < 1e-9) | (zc > W - 1e-9), 0.5, 1.0)
    weight = wy * wz
    rdofs = y_dof_of_node[right_nodes]

    f = np.zeros(K.shape[0])
    f[rdofs] = -P * weight / (ny * nz)             # total transverse load -P

    u = solve(*condense(K, f, D=left))
    compliance = float(f @ u)
    tip = float(u[rdofs].mean())
    return compliance, tip
