import numpy as np
import pytest

from fd_gate import fd_gate

from morphos.field import Field
from morphos.physics.oracle import PhysicsOracle
from morphos.physics.operators import (
    q4_plane_stress_stiffness,
    q4_thermoelastic_coupling,
    hex8_thermoelastic_coupling,
)
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.thermoelastic import ThermoElasticOracle


def thermo_bcs(shape):
    """Left edge clamped + held at zero temperature (sink); a downward point
    load at the right-mid node and a heat source injected at the far corner --
    the thermo-elastic analogue of the cantilever / channel test cases."""
    ny, nx = shape
    nny, nnx = ny + 1, nx + 1
    fixed_dofs = [(0, j, ax) for j in range(nny) for ax in ("x", "y")]
    loads = {(nnx - 1, nny // 2, "y"): -1.0}
    fixed_temps = [(0, j) for j in range(nny)]
    heat_sources = {(nnx - 1, nny - 1): 1.0}
    return fixed_dofs, loads, fixed_temps, heat_sources


def make_oracle(shape, alpha=1.0, **kw):
    fixed_dofs, loads, fixed_temps, heat_sources = thermo_bcs(shape)
    return ThermoElasticOracle(
        shape=shape,
        fixed_dofs=fixed_dofs,
        loads=loads,
        fixed_temps=fixed_temps,
        heat_sources=heat_sources,
        thermal_expansion=alpha,
        **kw,
    )


def test_is_a_physics_oracle():
    o = make_oracle((4, 8))
    assert isinstance(o, PhysicsOracle)
    assert o.provides_gradient is True


def test_coupling_matrix_shape_and_self_equilibrium():
    Le = q4_thermoelastic_coupling(1.0, 0.3, 1.0, 1.0)
    assert Le.shape == (8, 4)
    # A uniform temperature rise on a free element produces a self-equilibrated
    # thermal load (zero net force on each axis).
    f = (Le @ np.ones(4)).reshape(4, 2)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-12)

    Le3 = hex8_thermoelastic_coupling(1.0, 0.3, 1.0, 1.0)
    assert Le3.shape == (24, 8)
    f3 = (Le3 @ np.ones(8)).reshape(8, 3)
    assert np.allclose(f3.sum(axis=0), 0.0, atol=1e-12)


def test_alpha_zero_reduces_to_elasticity_oracle():
    shape = (4, 6)
    fixed_dofs, loads, fixed_temps, heat_sources = thermo_bcs(shape)
    te = ThermoElasticOracle(
        shape=shape, fixed_dofs=fixed_dofs, loads=loads,
        fixed_temps=fixed_temps, heat_sources=heat_sources, thermal_expansion=0.0,
    )
    el = ElasticityOracle(shape=shape, fixed_dofs=fixed_dofs, loads=loads)
    rng = np.random.default_rng(3)
    rho = 0.3 + 0.6 * rng.uniform(size=shape)
    field = Field(rho, spacing=1.0)
    rte, rel = te.solve(field), el.solve(field)
    assert rte.value == pytest.approx(rel.value, rel=1e-10)
    assert np.allclose(rte.gradient, rel.gradient, rtol=1e-8, atol=1e-10)


def test_thermal_expansion_changes_the_structural_response():
    shape = (4, 6)
    field = Field(np.full(shape, 0.6), spacing=1.0)
    cold = make_oracle(shape, alpha=0.0).solve(field).value
    hot = make_oracle(shape, alpha=2.0).solve(field).value
    assert not np.isclose(cold, hot)


def test_fixed_temperature_nodes_are_zero():
    shape = (3, 6)
    o = make_oracle(shape)
    T = o.solve(Field(np.ones(shape), spacing=1.0)).aux["temperature"]
    assert np.allclose(T[:, 0], 0.0, atol=1e-10)


def test_coupled_gradient_passes_directional_fd_gate_2d():
    shape = (5, 7)
    o = make_oracle(shape, alpha=1.5)
    rng = np.random.default_rng(17)
    x0 = 0.3 + 0.6 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)


def test_field_shape_mismatch_raises():
    o = make_oracle((4, 8))
    with pytest.raises(ValueError):
        o.solve(Field(np.ones((3, 8)), spacing=1.0))


def thermo_bcs_3d(shape):
    nz, ny, nx = shape
    nnz, nny, nnx = nz + 1, ny + 1, nx + 1
    fixed_dofs = [
        (0, j, k, ax)
        for j in range(nny) for k in range(nnz) for ax in ("x", "y", "z")
    ]
    loads = {(nnx - 1, nny // 2, nnz // 2, "y"): -1.0}
    fixed_temps = [(0, j, k) for j in range(nny) for k in range(nnz)]
    heat_sources = {(nnx - 1, nny - 1, nnz - 1): 1.0}
    return fixed_dofs, loads, fixed_temps, heat_sources


def test_coupled_gradient_passes_directional_fd_gate_3d():
    shape = (3, 3, 4)
    fixed_dofs, loads, fixed_temps, heat_sources = thermo_bcs_3d(shape)
    o = ThermoElasticOracle(
        shape=shape, fixed_dofs=fixed_dofs, loads=loads,
        fixed_temps=fixed_temps, heat_sources=heat_sources, thermal_expansion=1.2,
    )
    rng = np.random.default_rng(19)
    x0 = 0.3 + 0.6 * rng.uniform(size=shape)
    grad = o.solve(Field(x0, spacing=1.0)).gradient
    f = lambda x: o.solve(Field(x, spacing=1.0)).value
    fd_gate(f, grad, x0, rel=1e-4)
