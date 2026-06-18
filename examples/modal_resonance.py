"""Modal resonance demo: natural frequencies of a variable-mass membrane.

ModalOracle solves the generalized eigenproblem ``K phi = lambda M(rho) phi`` with
a sparse eigensolver, where the design density ``rho`` sets the mass. This script:

  1. computes the first few squared natural frequencies of a uniform membrane and
     checks them against the closed-form discrete Laplacian spectrum;
  2. confirms the eigenvalue adjoint ``dlambda/drho`` predicts the eigenvalue
     shift under a small density bump (first-order), which is what lets the engine
     tune a structure toward a target resonance.

Needs no extra dependencies beyond scipy. Run:
    python examples/modal_resonance.py
"""

import numpy as np

from morphos import Field
from morphos.physics.modal import ModalOracle


def analytic_spectrum(shape, h):
    ny, nx = shape
    p = np.arange(1, nx - 1)
    q = np.arange(1, ny - 1)
    lam_x = (2.0 / h**2) * (1.0 - np.cos(p * np.pi / (nx - 1)))
    lam_y = (2.0 / h**2) * (1.0 - np.cos(q * np.pi / (ny - 1)))
    return np.sort((lam_y[:, None] + lam_x[None, :]).ravel())


def main() -> None:
    shape = (41, 41)
    h = 1.0
    rho0 = 1.0
    analytic = analytic_spectrum(shape, h)

    print(f"uniform membrane {shape[0]}x{shape[1]}, density {rho0}")
    print(f"{'mode':>4} {'eigsh lambda':>14} {'analytic':>14} {'rel err':>10}")
    field = Field(np.full(shape, rho0), spacing=h)
    for mode in range(4):
        lam = ModalOracle(shape=shape, mode=mode).solve(field).value
        ref = analytic[mode] / rho0
        print(f"{mode:>4} {lam:>14.6f} {ref:>14.6f} {abs(lam - ref) / ref:>10.1e}")

    # Adjoint check: predicted vs actual eigenvalue shift under a density bump.
    oracle = ModalOracle(shape=shape, mode=0)
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.3 * rng.uniform(size=shape)
    base = oracle.solve(Field(rho, spacing=h))
    drho = 1e-3 * rng.uniform(-1.0, 1.0, size=shape)
    predicted = base.value + float(np.sum(base.gradient * drho))
    actual = oracle.solve(Field(rho + drho, spacing=h)).value
    print()
    print(f"fundamental lambda:       {base.value:.6f}")
    print(f"adjoint-predicted shift:  {predicted:.6f}")
    print(f"actual perturbed lambda:  {actual:.6f}")
    print(f"first-order rel error:    {abs(predicted - actual) / abs(actual):.1e}")


if __name__ == "__main__":
    main()
