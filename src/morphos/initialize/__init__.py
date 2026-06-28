"""Bio-inspired initial density fields for topology optimization.

Uniform-gray initialization is a poor starting point for SIMP/OC/MMA
topology optimization: it is a high-symmetry saddle point that gives weak
early gradients and biases the optimizer toward simple, often non-bio-like
optima. This package supplies initializers that break symmetry up front
using the same pattern-formation and branching-network principles seen in
biological structures (Turing patterns, Murray's law vascular networks,
trabecular bone).
"""

from morphos.initialize.reaction_diffusion import GrayScottRD
from morphos.initialize.bio_seed import murray_network, uniform_noise, from_image

__all__ = ["GrayScottRD", "murray_network", "uniform_noise", "from_image"]
