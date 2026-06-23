"""Morphos: a physics-driven engine that generates manufacturable geometry by
optimizing it toward the physical limit, and reports the margin to that limit.

The core is physics-agnostic. Every layer speaks one data type, the Field. A new
product is a new backend behind the same interfaces, not a rewrite.
"""

from morphos.field import Field
from morphos.spec import DesignSpec, ParametricSpec, DesignResult
from morphos.engine import Engine
from morphos.objective.objective import (
    Objective,
    ObjectiveValue,
    MaximizeValue,
    PhysicalBound,
)
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import TopologyOptimizer
from morphos.optimize.parametric import ParametricOptimizer
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle
from morphos.physics.heat import HeatConductionOracle
from morphos.physics.modal import ModalOracle
from morphos.physics.elasticity import ElasticityOracle
from morphos.physics.darcy import DarcyFlowOracle
from morphos.physics.thermoelastic import ThermoElasticOracle
from morphos.physics.stokes import StokesFlowOracle
from morphos.intent import DesignIntent, CantileverIntent, ChannelIntent
from morphos.report import PerformanceReport, build_report
from morphos.geometry.kernel import GeometryKernel
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.manufacturing.constraints import (
    ManufacturabilityConstraint,
    MinFeatureSize,
    Connectivity,
    Overhang,
)

__version__ = "0.0.1"

__all__ = [
    "Field",
    "DesignSpec",
    "ParametricSpec",
    "DesignResult",
    "Engine",
    "Objective",
    "ObjectiveValue",
    "MaximizeValue",
    "PhysicalBound",
    "Optimizer",
    "OptimizeResult",
    "TopologyOptimizer",
    "ParametricOptimizer",
    "PhysicsOracle",
    "PhysicsResult",
    "AnalyticOracle",
    "HeatConductionOracle",
    "ModalOracle",
    "ElasticityOracle",
    "DarcyFlowOracle",
    "ThermoElasticOracle",
    "StokesFlowOracle",
    "DesignIntent",
    "CantileverIntent",
    "ChannelIntent",
    "PerformanceReport",
    "build_report",
    "GeometryKernel",
    "VoxelKernel",
    "ManufacturabilityConstraint",
    "MinFeatureSize",
    "Connectivity",
    "Overhang",
]
