"""Morphos: a physics-driven engine that generates manufacturable geometry by
optimizing it toward the physical limit, and reports the margin to that limit.

The core is physics-agnostic. Every layer speaks one data type, the Field. A new
product is a new backend behind the same interfaces, not a rewrite.
"""

from morphos.field import Field
from morphos.spec import DesignSpec, ParametricSpec, DesignResult
from morphos.engine import Engine, CoupledEngine
from morphos.api import run, MorphosResult
from morphos.api.spec_io import from_json, to_json
from morphos.objective.objective import (
    Objective,
    ObjectiveValue,
    MaximizeValue,
    PhysicalBound,
)
from morphos.objective.multi_objective import MultiObjective, ObjectiveVector
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
from morphos.physics.conjugate_heat import ConjugateHeatOracle
from morphos.physics.fdtd3d import FDTD3DOracle
from morphos.intent import (
    DesignIntent,
    CantileverIntent,
    ChannelIntent,
    ThermalSinkIntent,
    StokesBrinkmanChannelIntent,
    ThermoElasticIntent,
    HeatExchangerIntent,
)
from morphos.report import PerformanceReport, build_report
from morphos.geometry.kernel import GeometryKernel
from morphos.geometry.numpy_voxel import VoxelKernel
from morphos.manufacturing.constraints import (
    ManufacturabilityConstraint,
    MinFeatureSize,
    MinWallThickness,
    PowderRemoval,
    Connectivity,
    Overhang,
)
from morphos.manufacturing.export import (
    ManufacturingBundle,
    PrintParams,
    export_bundle,
)
from morphos.feedback import FeedbackRecord, OracleCalibrator
from morphos.library import ComponentLibrary, ComponentSpec, default_library

__version__ = "0.0.1"

__all__ = [
    "Field",
    "DesignSpec",
    "ParametricSpec",
    "DesignResult",
    "Engine",
    "CoupledEngine",
    "run",
    "MorphosResult",
    "from_json",
    "to_json",
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
    "ConjugateHeatOracle",
    "FDTD3DOracle",
    "DesignIntent",
    "CantileverIntent",
    "ChannelIntent",
    "ThermalSinkIntent",
    "StokesBrinkmanChannelIntent",
    "ThermoElasticIntent",
    "HeatExchangerIntent",
    "PerformanceReport",
    "build_report",
    "GeometryKernel",
    "VoxelKernel",
    "MultiObjective",
    "ObjectiveVector",
    "ManufacturabilityConstraint",
    "MinFeatureSize",
    "MinWallThickness",
    "PowderRemoval",
    "Connectivity",
    "Overhang",
    "ManufacturingBundle",
    "PrintParams",
    "export_bundle",
    "FeedbackRecord",
    "OracleCalibrator",
    "ComponentLibrary",
    "ComponentSpec",
    "default_library",
]
