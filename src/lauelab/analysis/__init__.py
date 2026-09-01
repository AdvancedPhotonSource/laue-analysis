# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Backend-independent analysis primitives for Laue data."""

from .coloring import (
    closest_pole_colors,
    cubic_ipf_colors,
    cubic_ipf_key,
    hsv_key,
    hsv_position_colors,
    rodrigues_colors,
)
from .orientation import (
    CUBIC_SYMMETRY,
    HEXAGONAL_SYMMETRY,
    crystal_direction,
    lattice_params_to_reciprocal,
    misorientation_angle,
    misorientation_from_reference,
    misorientation_matrix,
    orientation_to_rodrigues,
    pairwise_misorientation,
    reciprocal_to_orientation,
    symmetry_operations,
    symmetry_reduce_orientation,
)
from .projection import (
    SurfaceFrame,
    cubic_hkl_family,
    pole_color_radius,
    pole_figure_points,
)
from .simulation import SimulationResult, simulate_reflections

__all__ = [
    "CUBIC_SYMMETRY",
    "HEXAGONAL_SYMMETRY",
    "SurfaceFrame",
    "SimulationResult",
    "closest_pole_colors",
    "crystal_direction",
    "cubic_hkl_family",
    "cubic_ipf_colors",
    "cubic_ipf_key",
    "hsv_key",
    "hsv_position_colors",
    "lattice_params_to_reciprocal",
    "misorientation_angle",
    "misorientation_from_reference",
    "misorientation_matrix",
    "orientation_to_rodrigues",
    "pairwise_misorientation",
    "pole_color_radius",
    "pole_figure_points",
    "reciprocal_to_orientation",
    "rodrigues_colors",
    "symmetry_operations",
    "symmetry_reduce_orientation",
    "simulate_reflections",
]
