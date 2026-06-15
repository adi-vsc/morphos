"""Electromagnetic physics backend: 2D FDFD via ceviche, with an adjoint gradient.

This is the first real physics target for Morphos. It solves the 2D frequency
domain Maxwell equations (Ez polarization) on the same regular grid the Field
uses, so the design density maps straight onto the permittivity grid with no
meshing. ceviche supplies the reverse-mode adjoint, so the gradient of the figure
of merit with respect to every voxel comes from one extra solve.

The design field is a material density in [0, 1] that maps linearly to
permittivity between ``eps_min`` and ``eps_max``. The figure of merit is the
field intensity at a probe point, normalized to the vacuum baseline, so a value
of one means no improvement over empty space and larger values mean the design
focuses more energy onto the probe. The normalization also keeps the gradient
well scaled for the optimizer.

It is an optional dependency: ``pip install morphos[em]``.
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult


class CevicheEMOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape,
        source,
        probe,
        *,
        wavelength: float = 1550e-9,
        dl: float = 40e-9,
        npml: int = 10,
        eps_min: float = 1.0,
        eps_max: float = 12.25,
    ) -> None:
        try:
            from ceviche.constants import C_0
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "CevicheEMOracle needs the optional electromagnetic dependency. "
                "Install it with: pip install morphos[em]"
            ) from exc

        self.shape = tuple(int(n) for n in shape)
        self.source_idx = tuple(int(i) for i in source)
        self.probe_idx = tuple(int(i) for i in probe)
        self.wavelength = float(wavelength)
        self.dl = float(dl)
        self.npml = int(npml)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.omega = 2.0 * np.pi * C_0 / self.wavelength

        src = np.zeros(self.shape, dtype=complex)
        src[self.source_idx] = 1.0
        self._source = src

        self._reference = self._raw_intensity(np.zeros(self.shape))
        if self._reference <= 0.0:
            raise ValueError(
                "vacuum reference intensity is zero at the probe; move the probe"
            )

    def _raw_intensity(self, density: np.ndarray) -> float:
        from ceviche import fdfd_ez

        eps_r = self.eps_min + np.asarray(density, dtype=float) * (
            self.eps_max - self.eps_min
        )
        sim = fdfd_ez(self.omega, self.dl, eps_r, [self.npml, self.npml])
        _, _, ez = sim.solve(self._source)
        return float(np.abs(ez[self.probe_idx]) ** 2)

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle grid "
                f"{self.shape}"
            )

        import autograd.numpy as npa
        from ceviche import fdfd_ez, jacobian

        omega, dl, npml = self.omega, self.dl, self.npml
        emin, emax = self.eps_min, self.eps_max
        shape, probe, source, ref = (
            self.shape,
            self.probe_idx,
            self._source,
            self._reference,
        )

        def objective(density_flat):
            density = npa.reshape(density_flat, shape)
            eps_r = emin + density * (emax - emin)
            sim = fdfd_ez(omega, dl, eps_r, [npml, npml])
            _, _, ez = sim.solve(source)
            return npa.abs(ez[probe]) ** 2 / ref

        d0 = field.values.flatten().astype(float)
        value = float(objective(d0))
        jac = jacobian(objective, mode="reverse")
        gradient = np.array(jac(d0), dtype=float).reshape(shape)

        return PhysicsResult(
            value=value, gradient=gradient, aux={"reference": ref}
        )
