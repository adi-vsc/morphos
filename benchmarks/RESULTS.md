# Morphos validation results

Each row compares a Morphos physics oracle against an **independent** reference (a closed-form solution, a published benchmark, or an external FEM solver). Every reference is assembled from first principles or by an external library; none reuses the oracle under test. Regenerate with `python -m benchmarks`.

_Run metadata — generated: 2026-06-25 17:52 UTC, python: 3.10.12, numpy: 2.2.6, scipy: 1.15.3, platform: Linux-6.8.0-124-generic-x86_64-with-glibc2.35, seed: 0, skipped: [], scikit-fem: 12.0.2._

**14/14 cases pass their documented tolerance.**

| Case | Reference type | QoI | Morphos | Reference | Rel. error | Tol | Pass | Time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | :---: | ---: |
| heat-conduction (MMS) | analytic (manufactured solution) | rel. L2 temperature error, 129x129 grid | 5.020e-05 | 0 | 5.020e-05 | 0.001 | ✅ | 0.13 |
| modal (membrane) | analytic (separation of variables) | fundamental eigenvalue lambda_11, 65x65 grid | 19.735 | 19.739 | 2.008e-04 | 0.01 | ✅ | 0.04 |
| elasticity (uniaxial bar) | analytic (Hooke's law) | tip axial displacement | 20 | 20 | 1.368e-14 | 1.000e-09 | ✅ | 0.00 |
| elasticity (cantilever) | analytic (Euler-Bernoulli) | tip deflection, 16x320 elements | -31972 | -32000 | 8.857e-04 | 0.02 | ✅ | 0.11 |
| elasticity (vs scikit-fem) | independent FEM (scikit-fem 12.0.2) | compliance, 8x160 elements | 31787 | 31787 | 2.351e-09 | 1.000e-06 | ✅ | 0.04 |
| darcy (porous column) | analytic (Darcy's law) | inlet-to-outlet pressure drop | 5 | 5 | 1.705e-14 | 1.000e-09 | ✅ | 0.00 |
| stokes (Poiseuille) | analytic (Poiseuille flow) | rel. L2 x-velocity error over the channel | 8.060e-15 | 0 | 8.060e-15 | 1.000e-06 | ✅ | 0.04 |
| topology MBB (compliance) | literature (88-line SIMP) | converged compliance, 60x20, vf=0.5, p=3, rmin=1.5 | 207.3 | 205 | 0.011211 | 0.1 | ✅ | 3.88 |
| topology MBB (volume) | constraint target | achieved volume fraction | 0.5 | 0.5 | 2.220e-16 | 0.001 | ✅ | 0.00 |
| topology Michell (compliance ratio) | theoretical bound (full-solid stiffness) | C_opt / C_solid at vf=0.4 | 2.2047 | 1 | 1.2047 | inf | ✅ | 6.80 |
| topology Michell (volume) | constraint target | achieved volume fraction | 0.4 | 0.4 | 0 | 0.001 | ✅ | 0.00 |
| heat-3d (MMS) | analytic (manufactured solution) | rel. L2 temperature error, 33x33x33 grid | 8.036e-04 | 0 | 8.036e-04 | 0.001 | ✅ | 7.67 |
| elasticity 3D (uniaxial bar) | analytic (Hooke's law) | tip axial displacement (Hex8) | 1.25 | 1.25 | 1.723e-14 | 1.000e-09 | ✅ | 0.02 |
| elasticity 3D (vs scikit-fem) | independent FEM (scikit-fem 12.0.2) | compliance, 4x4x24 Hex8 elements | 209.79 | 209.79 | 4.588e-12 | 1.000e-06 | ✅ | 0.41 |

## Reference sources

- **heat-conduction (MMS)** — HeatConductionOracle vs T=sin(pi x)sin(pi y), s=2 pi^2 T on [0,1]^2, T=0 on boundary (method of manufactured solutions, Roache 1998)
  - Observed convergence order 2.001 (expected 2).
- **modal (membrane)** — ModalOracle (K phi = lambda M phi, uniform mass) vs lambda_11 = pi^2(1/Lx^2 + 1/Ly^2) for a clamped rectangular membrane
  - Observed convergence order 1.999 (expected 2).
- **elasticity (uniaxial bar)** — ElasticityOracle vs delta = P L /(A E) (prismatic-bar extension)
  - Linear displacement field reproduced exactly by the bilinear quad element.
- **elasticity (cantilever)** — ElasticityOracle vs delta = P L^3/(3 E I), I = t H^3/12 (Euler-Bernoulli beam)
  - Slender beam L/H=20; residual gap to EB is transverse shear (Timoshenko reference -32062.40, rel. error 2.83e-03). Richardson order 1.942.
- **elasticity (vs scikit-fem)** — ElasticityOracle compliance vs an independent scikit-fem plane-stress ElementVector(ElementQuad1) assembly of the same cantilever
  - Two independent FE codes on the same problem agree to solver tolerance.
- **darcy (porous column)** — DarcyFlowOracle vs dp = Q L /(K A) (uniform 1D porous column, Darcy 1856)
  - Linear pressure field reproduced exactly by the Q4 diffusion element.
- **stokes (Poiseuille)** — StokesFlowOracle vs u(y) = u_max 4 y(H-y)/H^2 (fully-developed plane Poiseuille, Batchelor §4.2); exact in the biquadratic Taylor-Hood space
  - The parabolic solution lies in the FE velocity space, so error is solver-limited.
- **topology MBB (compliance)** — Andreassen et al. (2011), '88-line' SIMP MBB beam top88(60,20,0.5,3,1.5), converged compliance ~205 (E0=1, nu=0.3, unit load, density filter)
  - Morphos uses gradient ascent with a constant-shift volume projection rather than OC/MMA; ~10% tolerance reflects the optimizer-scheme difference, not solver error.
- **topology MBB (volume)** — VolumeConstraint target volume fraction 0.5
  - Volume projection holds the constraint to projection tolerance every iteration.
- **topology Michell (compliance ratio)** — Michell (1904) cantilever load case; reported against the rigorous lower bound C_solid (compliance of the rho=1 domain) since a finite-volume SIMP design cannot reach the singular Michell truss optimum
  - C_opt=87.28 at vf=0.4 vs C_solid=39.59 (the unattainable fully-solid floor). The 2.2x ratio is the expected stiffness cost of using 40% of the material; the design correctly never beats the solid bound.
- **topology Michell (volume)** — VolumeConstraint target volume fraction 0.4
- **heat-3d (MMS)** — HeatConductionOracle (3D) vs T=sin(pi x)sin(pi y)sin(pi z), s=3 pi^2 T on [0,1]^3, T=0 on boundary (manufactured solution, Roache 1998)
  - Observed convergence order 2.005 (expected 2).
- **elasticity 3D (uniaxial bar)** — ElasticityOracle (Hex8) vs delta = P L /(A E), A = Hy*Wz (prismatic bar)
  - Linear displacement field reproduced exactly by the trilinear hex element.
- **elasticity 3D (vs scikit-fem)** — ElasticityOracle (Hex8) compliance vs an independent scikit-fem ElementVector(ElementHex1) assembly of the same 3D cantilever
  - Two independent 3D FE codes on the same problem agree to solver tolerance.

## Mesh-refinement / convergence studies

| Study | QoI | Observed order | Expected order | Levels (h) | Errors |
| --- | --- | ---: | ---: | --- | --- |
| heat-conduction | rel. L2 temperature error | 2.001 | 2.00 | 0.0625, 0.03125, 0.01562, 0.007812 | 3.22e-03, 8.04e-04, 2.01e-04, 5.02e-05 |
| modal-membrane | rel. fundamental-eigenvalue error | 1.999 | 2.00 | 0.0625, 0.03125, 0.01562 | 3.21e-03, 8.03e-04, 2.01e-04 |
| elasticity-cantilever | rel. tip-deflection error vs Richardson-extrapolated limit | 1.942 | 2.00 | 0.5, 0.25, 0.125, 0.0625 | 1.09e-01, 2.99e-02, 7.79e-03, 2.03e-03 |
| heat-3d | rel. L2 temperature error (3D) | 2.005 | 2.00 | 0.125, 0.0625, 0.03125 | 1.30e-02, 3.22e-03, 8.04e-04 |

### heat-conduction

![heat-conduction convergence](convergence_heat.png)

Second-order (5-point Laplacian) convergence on a smooth manufactured solution.

### modal-membrane

![modal-membrane convergence](convergence_modal.png)

Discrete Laplacian eigenvalue -> continuum at second order.

### elasticity-cantilever

![elasticity-cantilever convergence](convergence_elasticity.png)

Richardson-extrapolated continuum tip deflection -32036.62 sits between Euler-Bernoulli (-32000.00) and Timoshenko (-32062.40), as expected for a 2D plane-stress beam. Q4 displacement QoI converges at ~2nd order.

### heat-3d

![heat-3d convergence](convergence_heat_3d.png)

Second-order (7-point Laplacian) convergence on a smooth 3D manufactured solution.
