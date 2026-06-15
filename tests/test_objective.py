import numpy as np
import pytest

from morphos.physics.oracle import PhysicsResult
from morphos.objective.objective import (
    Objective,
    ObjectiveValue,
    PhysicalBound,
    MaximizeValue,
)


def make_result(value=-3.0, grad=None):
    if grad is None:
        grad = np.array([[1.0, -2.0]])
    return PhysicsResult(value=value, gradient=grad)


def test_maximize_value_is_an_objective():
    assert isinstance(MaximizeValue(), Objective)


def test_evaluate_passes_value_and_gradient_through():
    obj = MaximizeValue()
    r = make_result(value=-3.0, grad=np.array([[1.0, -2.0]]))
    ov = obj.evaluate(r)
    assert isinstance(ov, ObjectiveValue)
    assert ov.fom == pytest.approx(-3.0)
    assert np.allclose(ov.gradient, np.array([[1.0, -2.0]]))


def test_scale_applies_to_fom_and_gradient():
    obj = MaximizeValue(scale=2.0)
    r = make_result(value=-3.0, grad=np.array([[1.0, -2.0]]))
    ov = obj.evaluate(r)
    assert ov.fom == pytest.approx(-6.0)
    assert np.allclose(ov.gradient, np.array([[2.0, -4.0]]))


def test_bound_returns_configured_physical_bound():
    bound = PhysicalBound(value=0.0, name="target-match")
    obj = MaximizeValue(bound=bound)
    assert obj.bound() is bound


def test_bound_default_is_none():
    assert MaximizeValue().bound() is None


def test_margin_is_gap_to_ceiling():
    bound = PhysicalBound(value=10.0, name="chu")
    assert bound.margin(7.0) == pytest.approx(3.0)


def test_attained_fraction_with_nonzero_bound():
    bound = PhysicalBound(value=10.0, name="chu")
    assert bound.attained_fraction(7.0) == pytest.approx(0.7)


def test_attained_fraction_none_when_bound_is_zero():
    bound = PhysicalBound(value=0.0, name="target-match")
    assert bound.attained_fraction(-1.0) is None
