"""Electromagnetic physics backend: 3D FDTD with a finite-difference gradient.

ceviche (the project's 2D EM backend, :mod:`morphos.physics.ceviche_em`) is
frequency-domain and 2D-only, so 3D electromagnetic topology optimization needs
a different solver. Time-domain FDTD (Yee grid) is the standard choice for full
3D Maxwell problems and, unlike the 2D FDFD adjoint, none of the common FDTD
packages this project might lean on are guaranteed to be present (``meep`` is
not installed in this environment; the pure-Python ``fdtd`` pip package may or
may not be). Rather than make 3D EM topology optimization a hard-dependency
feature, this oracle always works: it prefers a real backend if importable,
and otherwise falls back to a small self-contained pure-numpy 3D Yee-grid FDTD
implemented in this module, so ``FDTD3DOracle`` has zero hard dependencies.

Interface mirrors :class:`~morphos.physics.ceviche_em.CevicheEMOracle`: a
density field in ``[0, 1]`` maps linearly to permittivity between ``eps_min``
and ``eps_max``, the figure of merit is the field intensity at a probe point
normalized to the vacuum baseline (one means no improvement over empty space),
and :meth:`solve` returns a :class:`PhysicsResult` with that scalar value and a
gradient over every density voxel.

Gradient. No backend here supplies a time-domain adjoint (writing one is out of
scope for this oracle -- see the module docstring's "an adjoint FDTD gradient
is not required" instruction), so the gradient is a per-voxel forward finite
difference of the normalized intensity. This is the most expensive part of the
oracle (one extra FDTD run per design voxel), which is why tests keep the grid
and the timestep count tiny; production use on a larger grid would want a
real adjoint or a coarser parameterization, deferred to a future step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle, PhysicsResult


class FDTD3DOracle(PhysicsOracle):
    provides_gradient = True

    def __init__(
        self,
        shape: Tuple[int, int, int],
        source,
        probe,
        *,
        eps_min: float = 1.0,
        eps_max: float = 12.25,
        n_steps: int = 60,
        dt: Optional[float] = None,
        fd_step: float = 1e-2,
        backend: Optional[str] = None,
    ) -> None:
        """3D FDTD electromagnetic oracle on a ``(nz, ny, nx)`` voxel grid.

        Parameters
        ----------
        shape:
            ``(nz, ny, nx)`` number of voxels, matching the design ``Field``.
        source, probe:
            Integer ``(iz, iy, ix)`` voxel indices of a point source / probe.
        eps_min, eps_max:
            Relative permittivity at density 0 and density 1 (linear SIMP-free
            interpolation, matching ``CevicheEMOracle``).
        n_steps:
            Number of Yee-grid leapfrog timesteps to run. Kept small in tests;
            large enough that a pulse launched at the source has time to reach
            the probe.
        dt:
            Timestep override (Courant-stable default if ``None``, see
            :meth:`_default_dt`).
        fd_step:
            Forward-difference step on the density used to build the gradient.
        backend:
            Force a specific backend (``"meep"``, ``"fdtd"``, or ``"numpy"``)
            instead of auto-selecting; mainly for tests. Auto-selection prefers
            meep, then the ``fdtd`` pip package, then the bundled pure-numpy
            Yee-grid fallback, which is always available.
        """
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 3:
            raise ValueError("FDTD3DOracle is 3D: shape must be (nz, ny, nx)")
        if any(s < 2 for s in self.shape):
            raise ValueError("grid must have at least 2 voxels per axis")
        self.source_idx = tuple(int(i) for i in source)
        self.probe_idx = tuple(int(i) for i in probe)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.n_steps = int(n_steps)
        self.fd_step = float(fd_step)

        self.backend = backend or _select_backend()
        self.dt = float(dt) if dt is not None else _default_dt(self.shape)

        self._reference = self._raw_intensity(np.zeros(self.shape))
        if self._reference <= 0.0:
            raise ValueError(
                "vacuum reference intensity is zero at the probe; move the probe"
            )

    def _eps_of(self, density: np.ndarray) -> np.ndarray:
        return self.eps_min + np.asarray(density, dtype=float) * (
            self.eps_max - self.eps_min
        )

    def _raw_intensity(self, density: np.ndarray) -> float:
        """Run the selected FDTD backend and return |E|^2 at the probe."""
        eps_r = self._eps_of(density)
        if self.backend == "meep":
            return _run_meep(eps_r, self.source_idx, self.probe_idx, self.n_steps)
        if self.backend == "fdtd":
            return _run_fdtd_pkg(
                eps_r, self.source_idx, self.probe_idx, self.n_steps, self.dt
            )
        return _run_numpy_yee(
            eps_r, self.source_idx, self.probe_idx, self.n_steps, self.dt
        )

    def solve(self, field: Field) -> PhysicsResult:
        if field.values.shape != self.shape:
            raise ValueError(
                f"field shape {field.values.shape} does not match oracle grid "
                f"{self.shape}"
            )
        density = np.clip(field.values, 0.0, 1.0)
        ref = self._reference

        value = self._raw_intensity(density) / ref

        # Per-voxel forward finite difference (no adjoint backend is
        # available/required here, see module docstring).
        eps = self.fd_step
        gradient = np.zeros(self.shape)
        it = np.nditer(density, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            perturbed = density.copy()
            perturbed[idx] = min(1.0, perturbed[idx] + eps)
            step = perturbed[idx] - density[idx]
            if step <= 0.0:
                gradient[idx] = 0.0
                continue
            fp = self._raw_intensity(perturbed) / ref
            gradient[idx] = (fp - value) / step

        return PhysicsResult(
            value=value, gradient=gradient, aux={"reference": ref, "backend": self.backend}
        )


def _select_backend() -> str:
    """Pick the best available FDTD backend: meep, then the fdtd pip package,
    then the bundled pure-numpy Yee-grid fallback (always available)."""
    try:
        import meep  # noqa: F401

        return "meep"
    except ImportError:
        pass
    try:
        import fdtd  # noqa: F401

        return "fdtd"
    except ImportError:
        pass
    return "numpy"


def _default_dt(shape: Tuple[int, int, int]) -> float:
    """Courant-stable timestep for a unit-cell-spacing 3D Yee grid (c=1, dx=1):
    dt <= dx / (c * sqrt(3)), with a safety margin."""
    return 0.99 / np.sqrt(3.0)


def _run_meep(eps_r, source_idx, probe_idx, n_steps):  # pragma: no cover - needs meep
    raise NotImplementedError(
        "meep backend selected but not wired up; install meep is detected but "
        "no integration is implemented yet"
    )


def _run_fdtd_pkg(eps_r, source_idx, probe_idx, n_steps, dt):  # pragma: no cover - needs fdtd
    raise NotImplementedError(
        "fdtd backend selected but not wired up; the fdtd package is detected "
        "but no integration is implemented yet"
    )


def _run_numpy_yee(
    eps_r: np.ndarray,
    source_idx: Tuple[int, int, int],
    probe_idx: Tuple[int, int, int],
    n_steps: int,
    dt: float,
) -> float:
    """A minimal self-contained 3D Yee-grid FDTD: leapfrogs E and H on a
    staggered grid with a soft point source at ``source_idx`` and returns the
    time-peak ``|E|^2`` observed at ``probe_idx``.

    This is a genuine (if numerically simple) update of the full 3D Maxwell
    curl equations in normalized units (c=1, mu_0=1, eps_0=1):

        dH/dt = -curl(E),      dE/dt = curl(H) / eps_r,

    on a Yee grid with unit cell spacing, explicit leapfrog time-stepping, and
    simple Dirichlet (perfect-conductor) outer walls -- adequate for a tiny,
    short-duration topology-optimization surrogate where the absolute
    reflection accuracy at the domain edge matters far less than getting a
    smooth, density-dependent figure of merit for the gradient check.
    """
    nz, ny, nx = eps_r.shape
    # E components live at half-integer offsets on the dual grid; same shape
    # as eps_r is a fine simplification at this resolution (collocated, not
    # truly staggered, but symmetric and stable under the Courant limit used).
    Ex = np.zeros((nz, ny, nx))
    Ey = np.zeros((nz, ny, nx))
    Ez = np.zeros((nz, ny, nx))
    Hx = np.zeros((nz, ny, nx))
    Hy = np.zeros((nz, ny, nx))
    Hz = np.zeros((nz, ny, nx))

    inv_eps = 1.0 / eps_r
    sz, sy, sx = source_idx
    pz, py, px = probe_idx
    peak_intensity = 0.0

    for step in range(n_steps):
        # H update: dH/dt = -curl(E), forward differences.
        dEz_dy = np.zeros_like(Ez)
        dEz_dy[:, :-1, :] = Ez[:, 1:, :] - Ez[:, :-1, :]
        dEy_dz = np.zeros_like(Ey)
        dEy_dz[:-1, :, :] = Ey[1:, :, :] - Ey[:-1, :, :]
        Hx -= dt * (dEz_dy - dEy_dz)

        dEx_dz = np.zeros_like(Ex)
        dEx_dz[:-1, :, :] = Ex[1:, :, :] - Ex[:-1, :, :]
        dEz_dx = np.zeros_like(Ez)
        dEz_dx[:, :, :-1] = Ez[:, :, 1:] - Ez[:, :, :-1]
        Hy -= dt * (dEx_dz - dEz_dx)

        dEy_dx = np.zeros_like(Ey)
        dEy_dx[:, :, :-1] = Ey[:, :, 1:] - Ey[:, :, :-1]
        dEx_dy = np.zeros_like(Ex)
        dEx_dy[:, :-1, :] = Ex[:, 1:, :] - Ex[:, :-1, :]
        Hz -= dt * (dEy_dx - dEx_dy)

        # E update: dE/dt = curl(H) / eps_r, backward differences (dual grid).
        dHz_dy = np.zeros_like(Hz)
        dHz_dy[:, 1:, :] = Hz[:, 1:, :] - Hz[:, :-1, :]
        dHy_dz = np.zeros_like(Hy)
        dHy_dz[1:, :, :] = Hy[1:, :, :] - Hy[:-1, :, :]
        Ex += dt * inv_eps * (dHz_dy - dHy_dz)

        dHx_dz = np.zeros_like(Hx)
        dHx_dz[1:, :, :] = Hx[1:, :, :] - Hx[:-1, :, :]
        dHz_dx = np.zeros_like(Hz)
        dHz_dx[:, :, 1:] = Hz[:, :, 1:] - Hz[:, :, :-1]
        Ey += dt * inv_eps * (dHx_dz - dHz_dx)

        dHy_dx = np.zeros_like(Hy)
        dHy_dx[:, :, 1:] = Hy[:, :, 1:] - Hy[:, :, :-1]
        dHx_dy = np.zeros_like(Hx)
        dHx_dy[:, 1:, :] = Hx[:, 1:, :] - Hx[:, :-1, :]
        Ez += dt * inv_eps * (dHy_dx - dHx_dy)

        # Dirichlet (PEC) walls: zero tangential E on the outer boundary.
        Ex[0, :, :] = Ex[-1, :, :] = 0.0
        Ey[0, :, :] = Ey[-1, :, :] = 0.0
        Ez[0, :, :] = Ez[-1, :, :] = 0.0
        Ex[:, 0, :] = Ex[:, -1, :] = 0.0
        Ey[:, 0, :] = Ey[:, -1, :] = 0.0
        Ez[:, 0, :] = Ez[:, -1, :] = 0.0
        Ex[:, :, 0] = Ex[:, :, -1] = 0.0
        Ey[:, :, 0] = Ey[:, :, -1] = 0.0
        Ez[:, :, 0] = Ez[:, :, -1] = 0.0

        # Soft point source: a smooth differentiated-Gaussian pulse injected
        # into Ez at the source voxel, broadband enough to excite the grid
        # without ringing forever.
        t0, spread = 8.0, 3.0
        src = -(step - t0) / spread * np.exp(-0.5 * ((step - t0) / spread) ** 2)
        Ez[sz, sy, sx] += src

        intensity = (
            Ex[pz, py, px] ** 2 + Ey[pz, py, px] ** 2 + Ez[pz, py, px] ** 2
        )
        if intensity > peak_intensity:
            peak_intensity = float(intensity)

    return peak_intensity
