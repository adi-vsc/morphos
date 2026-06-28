"""Bio-inspired initial density fields via Gray-Scott reaction-diffusion.

Topology optimization is usually started from a uniform-gray density field,
which is a symmetric saddle point of most compliance objectives: gradients
near a perfectly uniform field can be weak or degenerate along certain
directions, and the optimizer needs many iterations of filtering/projection
noise before any spatial structure emerges. Seeding instead from a Turing
pattern (Gray-Scott reaction-diffusion, Pearson 1993) breaks symmetry up
front and gives the optimizer a starting point that already has the kind of
branching, spot, or stripe structure seen in biological load-bearing and
transport tissue (trabecular bone, leaf venation, coral).

The Gray-Scott model:

    du/dt = D_u * lap(u) - u*v^2 + F*(1 - u)
    dv/dt = D_v * lap(v) + u*v^2 - (F + k)*v

Depending on (F, k) this produces spots, stripes, or labyrinthine patterns
at steady state. We run the model for ``n_steps`` explicit-Euler steps on a
periodic grid and use the resulting ``v`` field (rescaled to hit the target
volume fraction) as the initial density.
"""

from __future__ import annotations

import numpy as np

from morphos.field import Field


def _laplacian(u: np.ndarray) -> np.ndarray:
    """Discrete Laplacian with periodic (wrap) boundary conditions.

    Works for any number of dimensions: sums the second difference along
    every axis using ``np.roll``, which wraps at the domain boundary.
    """
    lap = -2.0 * u.ndim * u
    for axis in range(u.ndim):
        lap += np.roll(u, 1, axis=axis)
        lap += np.roll(u, -1, axis=axis)
    return lap


class GrayScottRD:
    """Gray-Scott reaction-diffusion model for bio-inspired density seeding.

    Parameters
    ----------
    Du, Dv:
        Diffusion rates of the two species ``u`` (substrate) and ``v``
        (activator).
    F:
        Feed rate.
    k:
        Kill rate.
    dt:
        Explicit-Euler integration time step.
    n_steps:
        Number of integration steps to run.
    seed:
        RNG seed for the initial center-patch perturbation.
    """

    #: (Du, Dv, F, k) presets producing classic Turing pattern families.
    _PRESETS = {
        "spots": dict(Du=0.16, Dv=0.08, F=0.035, k=0.065),
        "stripes": dict(Du=0.16, Dv=0.08, F=0.060, k=0.062),
        "labyrinth": dict(Du=0.16, Dv=0.08, F=0.040, k=0.060),
        "holes": dict(Du=0.16, Dv=0.08, F=0.025, k=0.055),
    }

    def __init__(
        self,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.035,
        k: float = 0.065,
        dt: float = 1.0,
        n_steps: int = 2000,
        seed: int = 42,
    ) -> None:
        self.Du = float(Du)
        self.Dv = float(Dv)
        self.F = float(F)
        self.k = float(k)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.seed = int(seed)

    @classmethod
    def preset(cls, name: str, **overrides) -> "GrayScottRD":
        """Build a :class:`GrayScottRD` from a named (Du, Dv, F, k) preset.

        Valid names: "spots", "stripes", "labyrinth", "holes". Any other
        constructor keyword (dt, n_steps, seed, ...) may be passed as an
        override.
        """
        try:
            params = dict(cls._PRESETS[name])
        except KeyError as exc:
            valid = ", ".join(sorted(cls._PRESETS))
            raise ValueError(f"Unknown preset {name!r}; valid presets: {valid}") from exc
        params.update(overrides)
        return cls(**params)

    def _init_uv(self, shape) -> tuple:
        rng = np.random.default_rng(self.seed)
        u = np.ones(shape, dtype=float)
        v = np.zeros(shape, dtype=float)

        # Perturb a center patch (roughly 1/8 to 3/8 of each axis) to break
        # symmetry and seed pattern formation.
        slices = []
        for n in shape:
            lo = max(1, n // 2 - max(1, n // 8))
            hi = min(n, n // 2 + max(1, n // 8))
            if hi <= lo:
                hi = min(n, lo + 1)
            slices.append(slice(lo, hi))
        slices = tuple(slices)

        patch_shape = u[slices].shape
        noise_u = rng.uniform(-0.02, 0.02, size=patch_shape)
        noise_v = rng.uniform(-0.02, 0.02, size=patch_shape)
        u[slices] -= 0.5 + noise_u
        v[slices] += 0.25 + noise_v
        return u, v

    def _integrate(self, shape) -> np.ndarray:
        u, v = self._init_uv(shape)
        Du, Dv, F, k, dt = self.Du, self.Dv, self.F, self.k, self.dt
        for _ in range(self.n_steps):
            lap_u = _laplacian(u)
            lap_v = _laplacian(v)
            uvv = u * v * v
            u = u + dt * (Du * lap_u - uvv + F * (1.0 - u))
            v = v + dt * (Dv * lap_v + uvv - (F + k) * v)
        return v

    def generate(self, shape, volume_fraction: float, spacing=1.0) -> Field:
        """Run the reaction-diffusion model and return a density Field.

        The ``v`` species field is normalized to [0, 1] and then shifted so
        that its mean matches ``volume_fraction`` (clipping to [0, 1]).
        """
        shape = tuple(int(s) for s in shape)
        v = self._integrate(shape)

        v_min, v_max = v.min(), v.max()
        rho = (v - v_min) / (v_max - v_min + 1e-10)

        rho_scaled = np.clip(rho + (volume_fraction - rho.mean()), 0.0, 1.0)

        return Field(rho_scaled.reshape(shape), spacing=spacing)
