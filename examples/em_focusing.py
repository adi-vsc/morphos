"""Electromagnetic focusing demo: inverse-design a dielectric lens.

A point source radiates on one side of a 2D domain. The engine designs a
material distribution (density mapped to permittivity) that focuses as much
energy as possible onto a probe point on the other side, using the FDFD physics
backend and its adjoint gradient. The figure of merit is intensity at the probe
normalized to the vacuum baseline, so the printed number is an enhancement
factor: greater than one means the designed structure beats empty space.

Needs the optional electromagnetic dependency:
    pip install -e ".[em]"
Run:
    python examples/em_focusing.py
"""

import os

import numpy as np

from morphos import Field, MaximizeValue, TopologyOptimizer, DesignSpec, Engine
from morphos.physics.ceviche_em import CevicheEMOracle


def main() -> None:
    shape = (60, 60)
    oracle = CevicheEMOracle(
        shape=shape,
        source=(12, 30),
        probe=(48, 30),
        wavelength=1550e-9,
        dl=40e-9,
        npml=10,
        eps_max=12.25,
    )

    spec = DesignSpec(
        initial=Field(np.zeros(shape), spacing=40e-9),
        oracle=oracle,
        objective=MaximizeValue(),
        optimizer=TopologyOptimizer(step_size=4.0, max_iter=30, bounds=(0.0, 1.0)),
        name="em-focusing",
    )

    result = Engine().run(spec)

    material_fraction = float(np.mean(result.field.values))
    print(f"design:               {spec.name}")
    print(f"grid:                 {shape[0]} x {shape[1]} cells at {oracle.dl*1e9:.0f} nm")
    print(f"wavelength:           {oracle.wavelength*1e9:.0f} nm")
    print(f"iterations:           {result.iterations}")
    print(f"baseline FOM (vacuum): 1.000")
    print(f"optimized enhancement: {result.figure_of_merit:.3f} x")
    print(f"material fraction:     {material_fraction:.3f}")

    os.makedirs("out", exist_ok=True)
    np.save("out/em_focusing_density.npy", result.field.values)
    print("saved design density to out/em_focusing_density.npy")


if __name__ == "__main__":
    main()
