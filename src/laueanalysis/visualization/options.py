"""Discoverable built-in visualization choices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    """Stable option value and its human-readable label."""

    value: str
    label: str


AXIS_OPTIONS = (
    Choice("X", "X motor (um)"),
    Choice("Y", "Y motor (um)"),
    Choice("Z", "Z motor (um)"),
    Choice("H", "H (um)"),
    Choice("F", "F (um)"),
    Choice("depth", "Depth (um)"),
    Choice("Xlab", "X lab (um)"),
    Choice("Ylab", "Y lab (um)"),
    Choice("Zlab", "Z lab (um)"),
    Choice("Hlab", "H lab (um)"),
    Choice("Flab", "F lab (um)"),
)

COLOR_MODES = (
    Choice("cubic_ipf", "Cubic IPF"),
    Choice("rodrigues", "Rodrigues RGB"),
    Choice("misorientation", "Misorientation"),
    Choice("pole_hsv", "Pole Figure HSV"),
    Choice("n_indexed", "N Indexed"),
    Choice("goodness", "Goodness"),
    Choice("rms_error", "RMS Error"),
    Choice("n_patterns", "N Patterns"),
)

POLE_COLOR_MODES = (
    Choice("hsv_position", "Position HSV"),
    Choice("ipf", "Cubic IPF"),
    Choice("uniform", "Uniform"),
)

SURFACE_PRESETS = (
    Choice("normal", "APS 34-ID-E sample normal"),
    Choice("X", "APS 34-ID-E X"),
    Choice("H", "APS 34-ID-E H"),
    Choice("Y", "APS 34-ID-E Y"),
    Choice("Z", "APS 34-ID-E Z"),
    Choice("F", "APS 34-ID-E F"),
)

PALETTE_OPTIONS = (
    Choice("Viridis", "Viridis"),
    Choice("Plasma", "Plasma"),
    Choice("Inferno", "Inferno"),
    Choice("Magma", "Magma"),
    Choice("Jet", "Jet"),
    Choice("Rainbow", "Rainbow"),
    Choice("Earth", "Terrain"),
)
