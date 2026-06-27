# morphos MEMORY — operational patterns for Claude Code sessions

## Codebase map (confirmed as of 2026-06-27)

### Physics oracles (src/morphos/physics/)
| File | Oracle | Notes |
|------|--------|-------|
| elasticity.py | ElasticityOracle | Q4/Hex8, SIMP, self-adjoint compliance |
| heat.py | HeatConductionOracle + FEMHeatOracle | Finite-difference + Q4/Hex8 FEM |
| darcy.py | DarcyFlowOracle | Potential flow (not viscous), Q4/Hex8 |
| stokes.py | StokesFlowOracle | Brinkman-Stokes, Taylor-Hood via scikit-fem, self-adjoint |
| thermoelastic.py | ThermoelasticOracle | Monolithic Q4/Hex8, full coupled adjoint |
| conjugate_heat.py | ConjugateHeatOracle | Convective heat transfer |
| modal.py | ModalOracle | Eigenvalue, vibration avoidance |
| fdtd3d.py | FDTD3DOracle | 3D FDTD electromagnetics |
| ceviche_em.py | CevicheEMOracle | FDFD EM (requires ceviche) |
| analytic.py | AnalyticOracle | Closed-form reference oracles |
| coupled.py | MonolithicCoupledOracle | Generic monolithic multiphysics |

### Optimizers (src/morphos/optimize/)
| File | Optimizer | Notes |
|------|-----------|-------|
| topopt.py | TopologyOptimizer | Fixed-step gradient ascent + p-cont + Heaviside + filter |
| oc.py | OCOptimizer | Optimality-Criteria + bisection |
| mma.py | MMAOptimizer | Svanberg 1987 MMA + Heaviside + PDE filter |
| parametric.py | ParametricOptimizer | Continuous parameter optimization |
| pde_filter.py | PDEFilter | Helmholtz filter, self-adjoint |

### Missing (to be implemented)
- `optimize/levelset.py` — Level-set topology optimizer (Hamilton-Jacobi)
- `optimize/robust.py` — Robust/UQ optimization (delta-p ensemble)
- `initialize/` — Bio-inspired init (reaction-diffusion, Murray's law, trabecular bone seeding)
- Double-filter min length scale — not in any optimizer yet (Guest et al. 2004)
- `physics/nonlinear_elasticity.py` — Co-rotational Q4/Hex8 geometric nonlinearity

### Optional dependency map
```
scikit-fem  → StokesFlowOracle (Taylor-Hood elements)
pyamg       → AMG preconditioner in _linsolve.py  
scikit-image→ STL export, CLI visualization, skimage-based tests
ceviche     → CevicheEMOracle (EM FDFD)
jax         → planned diff-physics backend (not yet implemented)
```
Install all dev deps: `pip install -e ".[dev]"` (includes scikit-fem, pyamg, scikit-image)

## Key invariants / bug patterns

### MMA chain_vjp shape invariant
`_chain_vjp(x, filtered, beta, grad, constraint)` requires `grad.shape == x.values.shape` (2D for 2D fields, 3D for 3D). Never pass `np.ones(N)/N` (flat 1D); always `np.ones_like(x.values)/N`.

### SIMP design chain
`raw_x → filter → Heaviside(beta) → constraint.project() → design`
Volume constraint must target `mean(design)` not `mean(raw_x)`. The Heaviside shifts volume when beta >> 1. MMA bisection RHS = `dot(df1, x_k) - (mean(design) - V*)`.

### AMG preconditioner
`_AMGCache` rebuilds every 20 calls (rebuild_every=20). This is intentional — SIMP changes density 9 decades from uniform to binary. Cache in `[_AMGCache()]` list held by the oracle.

### Self-adjoint physics (no adjoint solve)
ElasticityOracle (compliance), DarcyFlowOracle (dissipation), StokesFlowOracle (dissipation), FEMHeatOracle: all self-adjoint. Gradient = explicit density derivative of stored energy / dissipation. No co-state solve needed.

### ThermoelasticOracle adjoint structure
Two extra linear solves on already-factorized operators: mechanical co-state `mu = K⁻¹ l`, thermal co-state `psi = A⁻¹ C^T mu`. Element sensitivity has three terms: mech stiffness, coupling, conduction.

### Heaviside projection
`H(x; beta, eta=0.5) = (tanh(beta*eta) + tanh(beta*(x-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))`.
VJP = `grad * beta * (1 - tanh(beta*(x-eta))^2) / den`. Both in topopt.py.

### PDEFilter (Helmholtz)
Solves `(-r²∇² + I)ρ̃ = ρ` with Neumann BCs. Assembled once, `splu`-factorized. Self-adjoint: VJP = same solve. `radius` is physical length (not voxels). Located in `optimize/pde_filter.py`.

## Level-set implementation guidance (for new levelset.py)

Use the shape-sensitivity / velocity-extension formulation:
1. `phi` = signed distance function (negative inside solid)
2. Velocity field `V_n = -dJ/dA` (shape derivative, from adjoint state)
3. Hamilton-Jacobi advection: `dphi/dt + V_n |grad phi| = 0` (Godunov upwind)
4. Periodic reinitilization via fast marching (scikit-fmm or scipy.ndimage trick)
5. Volume correction: add constant to phi to enforce mean volume

Use `scikit-fmm` for signed distance reinitialization. `pip install scikit-fmm`.
The shape derivative for compliance is `V_n = -(σ:ε)_surface` (strain energy density on boundary).

## Double-filter minimum length scale (Guest et al. 2004)

Two sequential filter-project steps:
1. Filter raw x with radius r_min → ρ̃
2. Heaviside project: ρ̄ = H(ρ̃; β_solid) (β large → solid feature filter)
3. Filter again with same r_min → ρ̃₂  
4. Heaviside project again: ρ_design = H(ρ̃₂; β_void) (complement enforces void features)

For minimum solid length: use eroded projection (β_solid → ∞ squeezes thin solid features to zero).
For minimum void length: complement the above.
Add `min_length_scale: float = 0` and `min_length_eta: float = 0.75` params to all three optimizers.

## Reaction-diffusion / bio-inspired init

### Gray-Scott model (Turing patterns)
`du/dt = D_u ∇²u - uv² + F(1-u)`
`dv/dt = D_v ∇²v + uv² - (F+k)v`
Parameter sets: spots (F=0.035, k=0.065), stripes (F=0.060, k=0.062), labyrinthine (F=0.040, k=0.060).
Run ~2000 steps on the design domain, then use `v` field as initial density (thresholded to V*).

### Murray's law branching (channel networks)
`r_parent^3 = r_left^3 + r_right^3` (optimal branching for flow resistance).
Build recursive tree from centroid outward, rasterize to density field.
Initial field = 1 inside tubes (radius schedule), 0 outside, then blur to V*.

### Trabecular bone seeding (3D)
Generate Voronoi-tessellated strut network: place ~50 seed points uniformly, connect neighbors if `||xi - xj|| < threshold`, rasterize cylinders as solid, fill space with thin shell. Gives ~30% volume fraction naturally.

## Robust optimization (delta-p method)

For each optimization step:
1. Perturb density: `rho_eroded = H(rho - delta)`, `rho_dilated = H(rho + delta)` (threshold perturbation)
2. Evaluate FOM on nominal + eroded + dilated fields
3. Objective = `min(f_nominal, f_eroded, f_dilated)` (worst-case)
4. Gradient from the active (worst-case) configuration only

`delta` = 0.05–0.15 (fraction of rho range) for LPBF tolerances.
Wrap any existing oracle; no oracle-specific code changes needed.

## Co-rotational FEM (geometric nonlinearity)

Element co-rotational formulation (Crisfield 1991):
1. Extract rotation `R_e` from current element configuration via polar decomposition of deformation gradient `F = R U`
2. Compute linear strain in rotated frame: `ε_e = B(R_e) u_e`  
3. Element stress: `σ_e = C ε_e`
4. Internal force: `f_int = R_e B^T σ_e |J|` (rotated back to global frame)
5. Tangent stiffness: `K_T = K_mat + K_geo` (material + geometric stiffness)
6. Newton-Raphson iteration until `||f_int - f_ext|| < tol`

Key: 2D uses `arctan2(F[1,0]-F[0,1], F[0,0]+F[1,1])/2` for rotation angle. 3D uses SVD-based polar decomposition.
Sensitivity via adjoint on converged nonlinear state.

## Test infrastructure

- `fd_gate` decorator: gates FD gradient tests behind `MORPHOS_FD_TESTS=1` env var to keep CI fast
- `pytest -m benchmark` re-selects benchmark tests deselected by default
- `test_volume_constraint.py` — verifies MMA/OC/TopOpt all satisfy volume constraint
- `test_mma.py` — MISSING, needs MMA integration tests with stokes/thermoelastic
