"""PerformanceReport: a structured, engineering-readable summary of a DesignResult.

A DesignResult carries the optimized field, the figure of merit, and the margin
to the physical bound, but not the oracle's engineering-unit quantities (those
live only in the PhysicsResult.aux dict returned at solve time, which the
optimizer discards after each step). build_report re-solves the oracle once on
the final field to recover them, then assembles a single object an engineer or
a downstream LLM copilot can read without knowing the internal Field/Optimizer
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Optional

import numpy as np

from morphos.physics.darcy import DarcyFlowOracle
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.oracle import PhysicsOracle
from morphos.spec import DesignResult


@dataclass
class PerformanceReport:
    quantities: dict
    mass_fraction: float
    figure_of_merit: float
    margin: Optional[float] = None
    attained_fraction: Optional[float] = None
    manufacturability: dict = _dc_field(default_factory=dict)


def _quantities_for(oracle: PhysicsOracle, aux: dict) -> dict:
    if isinstance(oracle, ElasticityOracle):
        return {
            "compliance": aux["compliance"],
            "max_displacement": float(np.max(np.abs(aux["displacement"]))),
        }
    if isinstance(oracle, DarcyFlowOracle):
        return {
            "dissipation": aux["dissipation"],
            "peak_pressure": float(np.max(np.abs(aux["pressure"]))),
        }
    raise TypeError(f"build_report has no quantity mapping for oracle type {type(oracle).__name__}")


def build_report(result: DesignResult, oracle: PhysicsOracle) -> PerformanceReport:
    aux = oracle.solve(result.field).aux
    return PerformanceReport(
        quantities=_quantities_for(oracle, aux),
        mass_fraction=float(np.mean(result.field.values)),
        figure_of_merit=result.figure_of_merit,
        margin=result.margin,
        attained_fraction=result.attained_fraction,
        manufacturability=result.manufacturability,
    )
