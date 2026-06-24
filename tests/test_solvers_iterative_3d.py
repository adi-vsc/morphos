"""Large-grid AMG-preconditioned Krylov solve path, shared across oracles.

The direct (LU) solve does not scale: fill-in blows up memory once the
interior DOF count reaches the tens of thousands, which a modest 32^3 grid
already exceeds. These tests pin the "iterative" and "auto" solver modes
(``morphos.physics._linsolve.solve_linear``) to the direct path they replace
on a grid large enough to actually exercise the AMG/CG path, and check that
"auto" switches between direct and iterative at the documented DOF threshold.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.physics.heat import HeatConductionOracle


def test_iterative_and_direct_agree_on_a_32_cubed_grid():
    pytest.importorskip("pyamg")
    shape = (32, 32, 32)
    rng = np.random.default_rng(101)
    target = np.zeros(shape)
    s = rng.normal(size=shape)

    direct = HeatConductionOracle(target=target, solver="direct")
    it = HeatConductionOracle(target=target, solver="iterative")

    Td = direct.solve(Field(s, spacing=1.0)).aux["temperature"]
    Ti = it.solve(Field(s, spacing=1.0)).aux["temperature"]

    rel_err = np.linalg.norm(Td - Ti) / np.linalg.norm(Td)
    assert rel_err < 1e-6


def test_iterative_converges_in_under_200_iterations_on_32_cubed_grid():
    pytest.importorskip("pyamg")
    shape = (32, 32, 32)
    rng = np.random.default_rng(102)
    target = np.zeros(shape)
    s = rng.normal(size=shape)

    it = HeatConductionOracle(target=target, solver="iterative")
    r = it.solve(Field(s, spacing=1.0))

    assert r.solver_iterations > 0
    assert r.solver_iterations < 200
    assert r.residual_norm < 1e-6


def test_auto_selects_direct_below_threshold_and_iterative_above():
    pytest.importorskip("pyamg")
    rng = np.random.default_rng(103)

    small_shape = (10, 10, 10)  # 8**3 = 512 interior dofs, well under 20000
    small_target = np.zeros(small_shape)
    small_o = HeatConductionOracle(target=small_target, solver="auto")
    small_r = small_o.solve(Field(rng.normal(size=small_shape), spacing=1.0))
    assert small_r.solver_iterations == 0  # direct path chosen

    big_shape = (50, 50, 50)  # 48**3 = 110592 interior dofs, over the threshold
    big_target = np.zeros(big_shape)
    big_o = HeatConductionOracle(target=big_target, solver="auto")
    big_r = big_o.solve(Field(rng.normal(size=big_shape), spacing=1.0))
    assert big_r.solver_iterations > 0  # iterative path chosen
