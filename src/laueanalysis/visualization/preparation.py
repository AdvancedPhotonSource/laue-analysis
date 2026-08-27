"""Backend-neutral preparation of maps, pole figures, and detector views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from laueanalysis.analysis import (
    SurfaceFrame,
    closest_pole_colors,
    cubic_hkl_family,
    cubic_ipf_colors,
    orientation_to_rodrigues,
    pole_color_radius,
    pole_figure_points,
    rodrigues_colors,
    symmetry_operations,
    symmetry_reduce_orientation,
)

from .data import DataScope, FrameId, ResultSet, VisualizationDataset, _readonly

Values = np.ndarray | Callable[[VisualizationDataset], np.ndarray]
Alignment = Literal["frame", "pattern", "selected"]


@dataclass(frozen=True)
class Axis:
    """Custom map coordinate values and display metadata."""

    values: Values
    label: str
    unit: str | None = None
    alignment: Alignment = "frame"

    def __post_init__(self):
        if not self.label:
            raise ValueError("axis label cannot be empty")
        if self.alignment not in ("frame", "pattern", "selected"):
            raise ValueError("axis alignment must be 'frame', 'pattern', or 'selected'")


@dataclass(frozen=True)
class ScalarColor:
    """Scalar values and rendering metadata for a spatial map."""

    values: str | Values
    label: str | None = None
    palette: str = "Viridis"
    limits: tuple[float, float] | None = None
    alignment: Alignment = "pattern"

    def __post_init__(self):
        if self.alignment not in ("frame", "pattern", "selected"):
            raise ValueError("color alignment must be 'frame', 'pattern', or 'selected'")
        if self.limits is not None:
            if len(self.limits) != 2 or not np.isfinite(self.limits).all() or self.limits[0] >= self.limits[1]:
                raise ValueError("color limits must be two finite increasing values")


@dataclass(frozen=True)
class MapData:
    """Prepared records for a two- or three-dimensional spatial map."""

    coordinates: np.ndarray
    axis_labels: tuple[str, ...]
    frame_ids: tuple[FrameId, ...]
    pattern_indices: np.ndarray
    colors: np.ndarray
    color_kind: Literal["scalar", "rgb"]
    color_label: str
    palette: str | None = None
    color_limits: tuple[float, float] | None = None
    indexed: np.ndarray | None = None

    def __post_init__(self):
        count = len(self.frame_ids)
        if self.color_kind not in ("scalar", "rgb"):
            raise ValueError("color_kind must be 'scalar' or 'rgb'")
        coordinates = _readonly(self.coordinates, dtype=float, name="coordinates")
        if coordinates.ndim != 2 or coordinates.shape[0] != count or coordinates.shape[1] not in (2, 3):
            raise ValueError("coordinates must have shape (n, 2) or (n, 3)")
        if len(self.axis_labels) != coordinates.shape[1]:
            raise ValueError("axis_labels must align with coordinate columns")
        patterns = _readonly(self.pattern_indices, dtype=int, shape=(count,), name="pattern_indices")
        colors = _readonly(self.colors, name="colors")
        expected = (count,) if self.color_kind == "scalar" else (count, 3)
        if colors.shape != expected:
            raise ValueError(f"colors must have shape {expected}")
        indexed = np.ones(count, dtype=bool) if self.indexed is None else self.indexed
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "pattern_indices", patterns)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "indexed", _readonly(indexed, dtype=bool, shape=(count,), name="indexed"))
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        object.__setattr__(self, "axis_labels", tuple(self.axis_labels))


@dataclass(frozen=True)
class PoleFigureData:
    """Prepared stereographic pole positions and pattern identities."""

    points: np.ndarray
    frame_ids: tuple[FrameId, ...]
    pattern_indices: np.ndarray
    colors: np.ndarray
    hkl: tuple[int, int, int]
    color_kind: Literal["rgb", "uniform"]
    color_radius: float | None = None
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        count = len(self.frame_ids)
        if self.color_kind not in ("rgb", "uniform"):
            raise ValueError("color_kind must be 'rgb' or 'uniform'")
        if len(self.hkl) != 3 or tuple(self.hkl) == (0, 0, 0):
            raise ValueError("hkl must contain three integers and cannot be (0, 0, 0)")
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("center must contain two finite values")
        object.__setattr__(self, "points", _readonly(self.points, dtype=float, shape=(count, 2), name="points"))
        object.__setattr__(self, "pattern_indices", _readonly(self.pattern_indices, dtype=int, shape=(count,), name="pattern_indices"))
        colors = _readonly(self.colors, dtype=float, name="colors")
        if colors.shape not in ((3,), (count, 3)):
            raise ValueError("colors must have shape (3,) or (n, 3)")
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))


@dataclass(frozen=True)
class DetectorPatternData:
    """Prepared indexed reflections for one pattern."""

    pattern_index: int
    hkl: np.ndarray
    predicted_xy: np.ndarray
    measured_peak_indices: np.ndarray

    def __post_init__(self):
        count = len(self.measured_peak_indices)
        object.__setattr__(self, "hkl", _readonly(self.hkl, dtype=int, shape=(count, 3), name="hkl"))
        object.__setattr__(self, "predicted_xy", _readonly(self.predicted_xy, dtype=float, shape=(count, 2), name="predicted_xy"))
        object.__setattr__(self, "measured_peak_indices", _readonly(self.measured_peak_indices, dtype=int, shape=(count,), name="measured_peak_indices"))


@dataclass(frozen=True)
class DetectorViewData:
    """Prepared detector image, peaks, and indexed-reflection overlays."""

    frame_id: FrameId
    detector_id: str
    extent: tuple[float, float]
    measured_xy: np.ndarray
    measured_peak_indices: np.ndarray
    measured_intensity: np.ndarray
    measured_indexed: np.ndarray
    patterns: tuple[DetectorPatternData, ...]
    image: np.ndarray | None = None

    def __post_init__(self):
        count = len(self.measured_xy)
        object.__setattr__(self, "measured_xy", _readonly(self.measured_xy, dtype=float, shape=(count, 2), name="measured_xy"))
        object.__setattr__(self, "measured_peak_indices", _readonly(self.measured_peak_indices, dtype=int, shape=(count,), name="measured_peak_indices"))
        object.__setattr__(self, "measured_intensity", _readonly(self.measured_intensity, dtype=float, shape=(count,), name="measured_intensity"))
        object.__setattr__(self, "measured_indexed", _readonly(self.measured_indexed, dtype=bool, shape=(count,), name="measured_indexed"))
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if self.image is not None:
            image = _readonly(self.image, name="image")
            if image.ndim != 2:
                raise ValueError("detector image must be two-dimensional")
            object.__setattr__(self, "image", image)


def _dataset(source):
    if isinstance(source, ResultSet):
        return source.to_visualization()
    if isinstance(source, VisualizationDataset):
        return source
    raise TypeError("source must be a ResultSet or VisualizationDataset")


def _pattern_rows(dataset, scope):
    return np.flatnonzero((scope or DataScope()).pattern_mask(dataset))


def _frame_axis(dataset, name):
    positions = dataset.sample_positions
    depth = np.nan_to_num(dataset.depths, nan=0.0)
    h = (positions[:, 1] + positions[:, 2]) / np.sqrt(2.0)
    f = (-positions[:, 1] + positions[:, 2]) / np.sqrt(2.0)
    lab = -positions.copy()
    lab[:, 2] += depth
    h_lab = (lab[:, 1] + lab[:, 2]) / np.sqrt(2.0)
    f_lab = (-lab[:, 1] + lab[:, 2]) / np.sqrt(2.0)
    axes = {
        "x": (positions[:, 0], "X motor (um)"),
        "y": (positions[:, 1], "Y motor (um)"),
        "z": (positions[:, 2], "Z motor (um)"),
        "h": (h, "H (um)"),
        "f": (f, "F (um)"),
        "depth": (dataset.depths, "Depth (um)"),
        "x_lab": (lab[:, 0], "X lab (um)"),
        "y_lab": (lab[:, 1], "Y lab (um)"),
        "z_lab": (lab[:, 2], "Z lab (um)"),
        "h_lab": (h_lab, "H lab (um)"),
        "f_lab": (f_lab, "F lab (um)"),
    }
    if name not in axes:
        raise ValueError(f"unknown axis {name!r}; choose from {tuple(axes)}")
    return axes[name]


def _aligned_values(values, dataset, rows, alignment, name):
    values = values(dataset) if callable(values) else values
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} values must be one-dimensional")
    if alignment == "frame":
        if len(array) != dataset.n_frames:
            raise ValueError(f"{name} values must contain one value per frame")
        return array[dataset.pattern_frame_indices[rows]]
    if alignment == "pattern":
        if len(array) != dataset.n_patterns:
            raise ValueError(f"{name} values must contain one value per pattern")
        return array[rows]
    if len(array) != len(rows):
        raise ValueError(f"{name} values must contain one value per selected pattern")
    return array


def _resolve_axis(axis, dataset, rows):
    if isinstance(axis, str):
        values, label = _frame_axis(dataset, axis)
        return values[dataset.pattern_frame_indices[rows]], label
    if not isinstance(axis, Axis):
        raise TypeError("axes must contain names or Axis objects")
    values = _aligned_values(axis.values, dataset, rows, axis.alignment, "axis")
    label = f"{axis.label} ({axis.unit})" if axis.unit else axis.label
    return values, label


def _crystal_directions(rotations, normal):
    directions = np.full((len(rotations), 3), np.nan)
    for index, rotation in enumerate(rotations):
        if not np.isfinite(rotation).all():
            continue
        try:
            direction = np.linalg.solve(rotation, normal)
        except np.linalg.LinAlgError:
            continue
        norm = np.linalg.norm(direction)
        if np.isfinite(norm) and norm > 0:
            directions[index] = direction / norm
    return directions


def _map_colors(color, dataset, rows, surface):
    scalar_fields = {
        "n_indexed": (dataset.pattern_n_indexed, "Indexed peaks"),
        "goodness": (dataset.pattern_goodness, "Goodness"),
        "rms_error": (dataset.pattern_rms_error_deg, "RMS error (deg)"),
        "n_patterns": (
            np.bincount(dataset.pattern_frame_indices, minlength=dataset.n_frames),
            "Patterns",
        ),
    }
    if not isinstance(color, (str, ScalarColor)):
        color = ScalarColor(color)
    if isinstance(color, ScalarColor):
        if isinstance(color.values, str):
            if color.values not in scalar_fields:
                raise ValueError(f"unknown scalar color {color.values!r}; choose from {tuple(scalar_fields)}")
            values, default_label = scalar_fields[color.values]
            alignment = "frame" if color.values == "n_patterns" else "pattern"
            values = _aligned_values(values, dataset, rows, alignment, "color")
        else:
            values = _aligned_values(color.values, dataset, rows, color.alignment, "color")
            default_label = "Value"
        return values.astype(float), "scalar", color.label or default_label, color.palette, color.limits
    if color in scalar_fields:
        values, label = scalar_fields[color]
        alignment = "frame" if color == "n_patterns" else "pattern"
        return _aligned_values(values, dataset, rows, alignment, "color").astype(float), "scalar", label, "Viridis", None
    rotations = dataset.pattern_rotations[rows]
    if color == "ipf":
        if dataset.crystal is None:
            raise ValueError("crystal context is required for cubic IPF coloring")
        if dataset.crystal.crystal_system != "cubic":
            raise ValueError("cubic IPF coloring requires a cubic crystal")
        directions = _crystal_directions(rotations, surface.normal)
        return cubic_ipf_colors(directions), "rgb", "Cubic IPF", None, None
    if color == "rodrigues":
        operations = symmetry_operations(dataset.crystal.space_group) if dataset.crystal else None
        reduced = [
            symmetry_reduce_orientation(rotation, operations=operations)
            if np.isfinite(rotation).all() else np.full((3, 3), np.nan)
            for rotation in rotations
        ]
        vectors = np.asarray([
            orientation_to_rodrigues(rotation) if np.isfinite(rotation).all() else [np.nan] * 3
            for rotation in reduced
        ], dtype=float).reshape((-1, 3))
        return rodrigues_colors(vectors), "rgb", "Rodrigues RGB", None, None
    raise ValueError("unknown color mode; choose from 'n_indexed', 'goodness', 'rms_error', 'n_patterns', 'ipf', or 'rodrigues'")


def prepare_map(source, *, axes=("x", "y"), color="n_indexed", scope=None, surface=None):
    """Prepare a two- or three-dimensional spatial map."""
    dataset = _dataset(source)
    if len(axes) not in (2, 3):
        raise ValueError("axes must contain two or three entries")
    rows = _pattern_rows(dataset, scope)
    surface = SurfaceFrame.aps_34ide(surface or "normal") if isinstance(surface, (str, type(None))) else surface
    if not isinstance(surface, SurfaceFrame):
        raise TypeError("surface must be a SurfaceFrame, a preset name, or None")
    resolved = [_resolve_axis(axis, dataset, rows) for axis in axes]
    coordinates = np.column_stack([item[0] for item in resolved]) if rows.size else np.empty((0, len(axes)))
    if len(coordinates) and not np.isfinite(coordinates).all():
        raise ValueError("selected map coordinates contain missing or non-finite values")
    colors, kind, label, palette, limits = _map_colors(color, dataset, rows, surface)
    frame_ids = tuple(dataset.frame_ids[index] for index in dataset.pattern_frame_indices[rows])
    indexed = np.isfinite(dataset.pattern_rotations[rows]).all(axis=(1, 2))
    return MapData(
        coordinates,
        tuple(item[1] for item in resolved),
        frame_ids,
        dataset.pattern_indices[rows],
        colors,
        kind,
        label,
        palette,
        limits,
        indexed,
    )


def prepare_pole_figure(
    source,
    *,
    hkl=(1, 0, 0),
    scope=None,
    surface=None,
    color="hsv_position",
    center=(0.0, 0.0),
    color_radius_deg=22.5,
):
    """Prepare stereographic pole positions for selected patterns."""
    dataset = _dataset(source)
    if dataset.crystal is None:
        raise ValueError("crystal context is required for a cubic pole figure")
    if dataset.crystal.crystal_system != "cubic":
        raise ValueError("cubic HKL families require a cubic crystal")
    rows = _pattern_rows(dataset, scope)
    surface = SurfaceFrame.aps_34ide(surface or "normal") if isinstance(surface, (str, type(None))) else surface
    if not isinstance(surface, SurfaceFrame):
        raise TypeError("surface must be a SurfaceFrame, a preset name, or None")
    points, local_rows = pole_figure_points(
        dataset.pattern_reciprocals[rows], cubic_hkl_family(hkl), surface=surface
    )
    finite = np.isfinite(points).all(axis=1)
    points, local_rows = points[finite], local_rows[finite]
    selected_rows = rows[local_rows]
    if color == "hsv_position":
        radius = pole_color_radius(center, color_radius_deg)
        pattern_colors = closest_pole_colors(points, local_rows, len(rows), center=center, radius=radius)
        colors = pattern_colors[local_rows]
        color_kind = "rgb"
    elif color == "ipf":
        rotations = dataset.pattern_rotations[rows]
        colors = cubic_ipf_colors(_crystal_directions(rotations, surface.normal))[local_rows]
        radius = None
        color_kind = "rgb"
    elif color == "uniform":
        colors = np.array([214 / 255, 20 / 255, 0.0])
        radius = None
        color_kind = "uniform"
    else:
        raise ValueError("unknown pole color mode; choose from 'hsv_position', 'ipf', or 'uniform'")
    return PoleFigureData(
        points=points,
        frame_ids=tuple(dataset.frame_ids[index] for index in dataset.pattern_frame_indices[selected_rows]),
        pattern_indices=dataset.pattern_indices[selected_rows],
        colors=colors,
        hkl=tuple(int(value) for value in hkl),
        color_kind=color_kind,
        color_radius=radius,
        center=tuple(float(value) for value in center),
    )


def _frame_index(dataset, frame_id):
    try:
        return dataset.frame_ids.index(frame_id)
    except ValueError as error:
        raise KeyError(f"unknown frame_id {frame_id!r}") from error


def _load_image(image):
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import h5py

    with h5py.File(path, "r") as source:
        return source["entry1/data/data"][...]


def prepare_detector_view(
    source,
    *,
    frame_id,
    patterns="all",
    image=None,
    detector_index=None,
):
    """Prepare measured and indexed-reflection detector overlays."""
    dataset = _dataset(source)
    frame_index = _frame_index(dataset, frame_id)
    if dataset.geometry is None:
        raise ValueError("detector geometry is required to prepare a detector view")
    if detector_index is None:
        detector_id = dataset.detector_ids[frame_index]
        if detector_id:
            detector_index = dataset.geometry.find_detector(detector_id)
            if detector_index < 0:
                raise ValueError(f"detector {detector_id!r} is not present in the geometry")
        else:
            detector_index = 0
    detector = dataset.geometry.detector(detector_index)

    peak_rows = np.flatnonzero(dataset.peak_frame_indices == frame_index)
    measured = np.column_stack([
        dataset.peaks["fit_x"][peak_rows], dataset.peaks["fit_y"][peak_rows]
    ])
    intensity = dataset.peaks["intens"][peak_rows]
    pattern_rows = np.flatnonzero(dataset.pattern_frame_indices == frame_index)
    if patterns == "best" and len(pattern_rows):
        pattern_rows = pattern_rows[[np.argmin(dataset.pattern_indices[pattern_rows])]]
    elif patterns != "all":
        if isinstance(patterns, str):
            raise ValueError("patterns must be 'best', 'all', or a sequence of ranks")
        requested = tuple(patterns)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in requested):
            raise ValueError("pattern ranks must be nonnegative integers")
        pattern_rows = pattern_rows[np.isin(dataset.pattern_indices[pattern_rows], requested)]

    indexed_mask = np.zeros(len(peak_rows), dtype=bool)
    prepared_patterns = []
    depth = None if np.isnan(dataset.depths[frame_index]) else dataset.depths[frame_index]
    start = dataset.starts[frame_index]
    group = dataset.groups[frame_index]
    for pattern_row in pattern_rows:
        assignment_rows = np.flatnonzero(dataset.assignment_pattern_rows == pattern_row)
        peak_indices = dataset.assignment_peak_indices[assignment_rows]
        indexed_mask[peak_indices] = True
        q = dataset.assignment_hkl[assignment_rows] @ dataset.pattern_reciprocals[pattern_row]
        full_xy = detector.q_to_pixel(q, depth=depth)
        roi_xy = (full_xy - start - (group - 1) / 2.0) / group
        prepared_patterns.append(DetectorPatternData(
            int(dataset.pattern_indices[pattern_row]),
            dataset.assignment_hkl[assignment_rows],
            roi_xy,
            peak_indices,
        ))

    if image is True:
        retained = dataset.images[frame_index]
        source_path = dataset.input_images[frame_index]
        if retained is not None:
            image_data = retained
        elif source_path:
            image_data = _load_image(source_path)
        else:
            raise ValueError(f"frame {frame_id!r} has no retained image or source image path")
    elif image is None or image is False:
        image_data = None
    else:
        image_data = _load_image(image)
    if image_data is not None and np.asarray(image_data).ndim != 2:
        raise ValueError("detector image must be two-dimensional")

    extent = tuple((dataset.image_shapes[frame_index][::-1]).astype(float))
    return DetectorViewData(
        frame_id=frame_id,
        detector_id=detector.detector_id,
        extent=extent,
        measured_xy=measured,
        measured_peak_indices=dataset.peak_indices[peak_rows],
        measured_intensity=intensity,
        measured_indexed=indexed_mask,
        patterns=tuple(prepared_patterns),
        image=image_data,
    )
