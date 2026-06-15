"""Tests for the electromagnetic backend integration point.

These skip unless the optional electromagnetic dependency is installed, so core
CI stays fast and dependency-light. They document the backend's current state:
constructible when the dependency is present, with the solve as the next gate.
"""

import pytest

ceviche = pytest.importorskip("ceviche")

from morphos.field import Field  # noqa: E402
from morphos.physics.ceviche_em import CevicheEMOracle  # noqa: E402
from morphos.physics.oracle import PhysicsOracle  # noqa: E402
import numpy as np  # noqa: E402


def test_ceviche_oracle_constructs_when_dependency_present():
    o = CevicheEMOracle()
    assert isinstance(o, PhysicsOracle)


def test_ceviche_solve_is_the_next_gate():
    o = CevicheEMOracle()
    with pytest.raises(NotImplementedError):
        o.solve(Field(np.zeros((4, 4)), spacing=1.0))
