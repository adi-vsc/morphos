"""Morphos: a physics-driven engine that generates manufacturable geometry by
optimizing it toward the physical limit, and reports the margin to that limit.

The core is physics-agnostic. Every layer speaks one data type, the Field. A new
product is a new backend behind the same interfaces, not a rewrite.
"""

from morphos.field import Field
from morphos.spec import DesignSpec, DesignResult
from morphos.engine import Engine
from morphos.objective.objective import (
    Objective,
    ObjectiveValue,
    MaximizeValue,
    PhysicalBound,
)
from morphos.optimize.optimizer import Optimizer, OptimizeResult
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.oracle import PhysicsOracle, PhysicsResult
from morphos.physics.analytic import AnalyticOracle
from morphos.physics.heat import HeatConductionOracle
from morphos.geometry.kernel import GeometryKernel
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.manufacturing.constraints import (
    ManufacturabilityConstraint,
    MinFeatureSize,
    Connectivity,
)

__version__ = "0.0.1"

__all__ = [
    "Field",
    "DesignSpec",
    "DesignResult",
    "Engine",
    "Objective",
    "ObjectiveValue",
    "MaximizeValue",
    "PhysicalBound",
    "Optimizer",
    "OptimizeResult",
    "TopologyOptimizer",
    "PhysicsOracle",
    "PhysicsResult",
    "AnalyticOracle",
    "HeatConductionOracle",
    "GeometryKernel",
    "VoxelKernel",
    "ManufacturabilityConstraint",
    "MinFeatureSize",
    "Connectivity",
]
