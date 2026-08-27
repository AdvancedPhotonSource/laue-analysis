"""Discoverable built-in visualization choices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    """Stable option value and its human-readable label."""

    value: str
    label: str


AXIS_OPTIONS = (
    Choice("x", "X motor (um)"),
    Choice("y", "Y motor (um)"),
    Choice("z", "Z motor (um)"),
    Choice("h", "H (um)"),
    Choice("f", "F (um)"),
    Choice("depth", "Depth (um)"),
    Choice("x_lab", "X lab (um)"),
    Choice("y_lab", "Y lab (um)"),
    Choice("z_lab", "Z lab (um)"),
    Choice("h_lab", "H lab (um)"),
    Choice("f_lab", "F lab (um)"),
)

COLOR_MODES = (
    Choice("n_indexed", "Indexed peaks"),
    Choice("goodness", "Goodness"),
    Choice("rms_error", "RMS error"),
    Choice("n_patterns", "Patterns"),
    Choice("ipf", "Cubic IPF"),
    Choice("rodrigues", "Rodrigues RGB"),
    Choice("pole_hsv", "Pole position (HSV)"),
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
    Choice("Cividis", "Cividis"),
    Choice("Turbo", "Turbo"),
)
