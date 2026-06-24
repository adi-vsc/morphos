"""Component library: validated, parameterized sub-component templates."""

from morphos.library.component import ComponentLibrary, ComponentSpec
from morphos.library.channels import (
    register_channels,
    straight_channel,
    serpentine_channel,
)
from morphos.library.fins import corrugated_fin, pin_fin_array, plate_fin_array
from morphos.library.manifolds import tree_manifold, y_manifold
from morphos.library.structural import cross_rib, honeycomb_rib, i_beam_rib

import numpy as np


def default_library() -> ComponentLibrary:
    """A :class:`ComponentLibrary` preloaded with the built-in components."""
    lib = ComponentLibrary()
    register_channels(lib)

    lib.register(
        "pin_fin_array", pin_fin_array,
        expected_performance={"heat_transfer_coeff_W_m2K": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ConjugateHeatOracle"],
    )
    lib.register(
        "plate_fin_array", plate_fin_array,
        expected_performance={"heat_transfer_coeff_W_m2K": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ConjugateHeatOracle"],
    )
    lib.register(
        "corrugated_fin", corrugated_fin,
        expected_performance={"heat_transfer_coeff_W_m2K": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ConjugateHeatOracle"],
    )

    lib.register(
        "y_manifold", y_manifold,
        expected_performance={"pressure_drop_Pa": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize", "Connectivity"],
        oracle_types=["DarcyFlowOracle", "StokesFlowOracle"],
    )
    lib.register(
        "tree_manifold", tree_manifold,
        expected_performance={"pressure_drop_Pa": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize", "Connectivity"],
        oracle_types=["DarcyFlowOracle", "StokesFlowOracle"],
    )

    lib.register(
        "cross_rib", cross_rib,
        expected_performance={"stiffness_N_m": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ElasticityOracle", "ThermoElasticOracle"],
    )
    lib.register(
        "i_beam_rib", i_beam_rib,
        expected_performance={"stiffness_N_m": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ElasticityOracle", "ThermoElasticOracle"],
    )
    lib.register(
        "honeycomb_rib", honeycomb_rib,
        expected_performance={"stiffness_N_m": (0.0, np.inf)},
        applicable_constraints=["MinFeatureSize"],
        oracle_types=["ElasticityOracle", "ThermoElasticOracle"],
    )

    return lib


__all__ = [
    "ComponentLibrary",
    "ComponentSpec",
    "default_library",
    "register_channels",
    "straight_channel",
    "serpentine_channel",
    "pin_fin_array",
    "plate_fin_array",
    "corrugated_fin",
    "y_manifold",
    "tree_manifold",
    "cross_rib",
    "i_beam_rib",
    "honeycomb_rib",
]
