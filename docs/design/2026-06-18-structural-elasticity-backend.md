# Structural elasticity backend: SIMP compliance minimization

Date: 2026-06-18.

## What this is

A fourth `PhysicsOracle` backend, `ElasticityOracle`
(`src/morphos/physics/elasticity.py`), solving 2D plane-stress static
equilibrium for density-based topology optimization (SIMP: "solid isotropic
material with penalization"). It completes the classic structural / thermal /
electromagnetic triad that topology optimization is usually demonstrated on:
`HeatConductionOracle` is the thermal leg, `ceviche_em` the electromagnetic
leg, and this is the structural leg -- the one most directly tied to a
manufacturable mechanical part.

## Formulation

The design field is a per-element density `rho` in `[0, 1]` on the same
`(ny, nx)` grid convention as the existing topology-optimization backends.
Each grid cell is one bilinear-quad (Q4) plane-stress finite element. SIMP
penalizes intermediate densities so the optimizer is pushed toward "black and
white" (solid/void) designs:

```
E(rho) = E_min + rho**p * (E0 - E_min)
```

`p = 3` by default (the standard exponent), `E_min = 1e-9 * E0` (a stiffness
floor, not zero, so the global matrix never goes singular in void regions --
the fix from Andreassen et al. 2011's 88-line SIMP MATLAB code, which this
backend follows for the overall formulation).

The element stiffness matrix is derived from 2x2 Gauss quadrature
(`morphos.physics.operators.q4_plane_stress_stiffness`) rather than
transcribed from a textbook 8x8 array -- the codebase's existing
`interior_laplacian` is the finite-difference analogue of this, and now both
live in `operators.py` as the shared discrete-operator module. The function is
checked for symmetry, exact linear scaling with modulus, and the textbook
null space (three planar rigid-body modes -- two translations, one rotation
-- annihilated exactly, the other five eigenvalues strictly positive).

Global assembly, boundary conditions, and the solve:

- Per-element DOF tables map each of the `ny*nx` elements to its 8 global
  degrees of freedom (4 nodes x 2 components), built once at construction.
- `K_e = k_floor + rho_e**p * k0` where `k0` is the Q4 stiffness at modulus
  `E0 - E_min` and `k_floor` at modulus `E_min`; summing these reproduces
  `E(rho_e)` exactly while keeping the floor's contribution `rho`-independent
  (it does not enter the gradient).
- The global `K` is assembled as a sparse COO matrix from all elements'
  local 8x8 contributions (vectorized via `np.repeat`/`np.tile`, not a Python
  element loop over rows/cols, mirroring the no-Python-loop discipline of
  `interior_laplacian`).
- Supports (`fixed_dofs`) and loads (`loads`) are specified as
  `(node_x, node_y, axis)` keys, `axis` in `{"x", "y"}`, node indices 0-based
  on the `(ny+1) x (nx+1)` node grid. Dirichlet rows/columns are simply
  excluded from the linear solve (reduced system on free DOFs), the standard
  and exact way to enforce zero-displacement BCs in a direct solve.
- `K_ff u_f = F_f` is solved by sparse LU (`scipy.sparse.linalg.spsolve`);
  direct is sufficient at the grid sizes this backend targets (the design
  brief does not require an iterative path here, unlike `HeatConductionOracle`
  which offers both for scale).

Figure of merit: compliance `C = F^T u = u^T K u`, the scalar measure of
structural flexibility that SIMP topology optimization minimizes. Following
the exact convention `HeatConductionOracle` uses for its squared-error
mismatch, `PhysicsResult.value = -C`, so the engine's gradient-*ascent*
optimizer (`TopologyOptimizer`) maximizing `value` is equivalent to minimizing
compliance.

## The gradient: self-adjoint, no extra solve

Unlike `HeatConductionOracle` (whose figure of merit is a mismatch against a
target and needs a genuine co-state/adjoint solve), SIMP compliance
minimization is the textbook self-adjoint case. Because the load `F` does not
depend on `rho` and `K` is symmetric:

```
K u = F
=> K du/drho + dK/drho u = 0          (differentiate, F constant)
=> du/drho = -K^-1 (dK/drho) u
dC/drho = F^T du/drho = u^T K du/drho = -u^T (dK/drho) u   (since F = K u)
```

so per element,

```
dC/drho_e = -p * rho_e**(p-1) * (E0 - E_min) * u_e^T @ k0 @ u_e
```

(`u_e` the element's local displacement vector sliced from the global
solution, `k0` the unit-floor-removed Q4 stiffness). Since `value = -C`:

```
d(value)/drho_e = +p * rho_e**(p-1) * (E0 - E_min) * u_e^T @ k0 @ u_e
```

The same displacement solve that produces the compliance also produces every
element's sensitivity (`u_e^T k0 u_e` is the per-element strain energy at unit
modulus) -- no second linear solve, unlike the heat backend's adjoint.

This formula was not trusted on theory alone. It was implemented, then gated
against a central finite difference of the actual assembled, BC-reduced `K`
on a randomized density field (`test_simp_gradient_matches_finite_differences`,
`atol=1e-4`) and against the project's standard directional FD harness
(`tests/fd_gate.py`, `test_simp_gradient_passes_directional_fd_gate`,
`rel=1e-4`). The first implementation attempt had the textbook sign backwards
relative to this codebase's `value = -compliance` convention (caught
immediately by the FD gate reporting `rel_err ~ 2.0`, i.e. the analytic and FD
values were equal in magnitude and opposite in sign) and was corrected by
re-deriving the chain rule above rather than re-guessing the sign.

## Independent verification beyond self-consistency

Per the project's "don't trust gradients against only your own FD" discipline
extended here to the forward solve too: `test_single_element_compliance_matches_hand_calculation`
assembles the *same* Q4 element stiffness matrix from independently
re-derived first principles (a second, separately written 2x2 Gauss-quadrature
loop in the test file, not a call into `operators.py`), reduces it by hand for
a one-element, two-fixed-node, two-loaded-node case, solves the 4x4 reduced
system directly with `numpy.linalg.solve`, and checks the oracle's compliance
matches that hand-assembled answer to `rel=1e-8`. This is independent of the
oracle's own assembly and indexing code, not merely a restatement of it.

`test_q4_stiffness_null_space_is_the_three_rigid_body_modes` is a second,
purely analytic check: a single Q4 element's stiffness matrix must have rank 5
(8 DOF minus exactly 3 planar rigid-body modes) and explicitly annihilate the
two rigid translations -- a property with a known right answer independent of
any FE textbook table, derivable from the physics of an unconstrained
strain-free body.

## What this proves and does not prove

Proves: a real structural FEM (not a toy), with a self-adjoint SIMP gradient
verified against finite differences and against an independently re-derived
hand calculation (not just internal self-consistency), slots into the same
`PhysicsOracle` interface as the thermal and EM backends with no engine or
optimizer changes. The `examples/cantilever_simp.py` demo shows a uniform
0.4-volume-fraction cantilever's compliance dropping roughly 7-10x under
gradient-ascent topology optimization, settling into a strut-and-tie-like
density pattern recognizable as a cantilever topology (solid material along
the top flange near the fixed support and along the bottom flange carrying
the tip load, a sparser web between).

Does not prove: a converged, fully black-and-white, checkerboard-free,
manufacturable design. There is no minimum-feature-size filter or density
projection applied in the demo (the `Manufacturability` interface exists and
this backend is compatible with it, e.g. `MinFeatureSize`, but the demo runs
unconstrained to keep the example minimal), so the optimized field shown is a
gray-scale density map, not a crisp 0/1 part -- standard for un-filtered SIMP
and the reason real SIMP pipelines always add a filter/projection stage,
which is a manufacturability-layer concern, not a physics-oracle one. Nor does
this prove anything about 3D elasticity, large-deformation/nonlinear
behavior, multiple load cases, stress constraints (only compliance is
optimized, not local stress/yield), or buckling -- all out of scope for this
backend. Also unverified: performance/scaling behavior at large grids (the
direct sparse LU solve, like `HeatConductionOracle`'s `solver="direct"` path,
will eventually need an iterative alternative for big meshes; this backend
does not add one, matching the brief that direct-only is sufficient here).
