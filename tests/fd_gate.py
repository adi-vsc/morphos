"""Directional finite-difference gate for analytic gradients / sensitivities.

The engine's rule: never trust an analytic sensitivity until a central finite
difference confirms it. A *directional* derivative collapses a whole gradient
field to one scalar comparison, so one extra pair of forward evaluations gates
the entire gradient regardless of dimension.

Usage in a test:

    from fd_gate import fd_gate
    grad = prob.sensitivity(x0)              # analytic, shape of x0
    fd_gate(lambda x: prob.objective(x), grad, x0, rel=1e-4)

Raises AssertionError on mismatch; returns (analytic, fd, rel_err) on pass.
"""
from __future__ import annotations

import numpy as np


def directional_fd(f, x, d, h=1e-6):
    """Central finite difference of scalar-valued f along direction d at x."""
    x = np.asarray(x, float)
    d = np.asarray(d, float)
    return (f(x + h * d) - f(x - h * d)) / (2.0 * h)


def fd_gate(f, grad, x, d=None, h=1e-6, rel=1e-4, abs_=1e-7, seed=0):
    """Assert the analytic gradient `grad` matches a central FD of scalar `f`
    along a direction `d` (random if not given) at point `x`.

    f    : callable x -> float (the objective; must accept x's shape)
    grad : analytic dJ/dx, same shape as x
    d    : direction; default a reproducible random vector (seed)
    rel  : relative tolerance on |analytic - fd|
    abs_ : absolute floor (for near-zero gradients)

    Returns (analytic, fd, rel_err). Raises AssertionError on failure.
    """
    x = np.asarray(x, float)
    grad = np.asarray(grad, float)
    if grad.shape != x.shape:
        raise ValueError(f"grad shape {grad.shape} != x shape {x.shape}")
    if d is None:
        d = np.random.default_rng(seed).uniform(-1.0, 1.0, x.shape)
    d = np.asarray(d, float)

    an = float(np.sum(grad * d))
    fd = float(directional_fd(f, x, d, h))
    denom = max(abs(fd), abs(an), 1e-30)
    rel_err = abs(an - fd) / denom
    ok = abs(an - fd) <= max(rel * denom, abs_)
    if not ok:
        raise AssertionError(
            f"FD gate FAILED: analytic={an:.8e} fd={fd:.8e} "
            f"rel_err={rel_err:.2e} (rel tol={rel:.0e}, abs tol={abs_:.0e}, h={h:.0e})"
        )
    return an, fd, rel_err


if __name__ == "__main__":
    # self-test: gradient of f(x) = 0.5 x^T A x is A x.
    rng = np.random.default_rng(1)
    n = 6
    A = rng.normal(size=(n, n)); A = A + A.T
    x0 = rng.normal(size=n)
    f = lambda x: 0.5 * x @ A @ x
    grad = A @ x0
    an, fd, rel = fd_gate(f, grad, x0)
    print(f"self-test OK: analytic={an:.6e} fd={fd:.6e} rel_err={rel:.2e}")
