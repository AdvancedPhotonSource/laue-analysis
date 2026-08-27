"""Normalized data models shared by visualization preparation and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from laueanalysis.indexing import Crystal, FrameResult, Geometry

FrameId = str | int
PatternSelection = Literal["best", "all"] | tuple[int, ...]


def _readonly(value, *, dtype=None, shape=None, name="array"):
    array = np.array(value, dtype=dtype, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}")
    array.setflags(write=False)
    return array


def _validate_frame_ids(values: Sequence[FrameId], count: int) -> tuple[FrameId, ...]:
    ids = tuple(values)
    if len(ids) != count:
        raise ValueError(f"frame_ids must contain {count} values")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in ids):
        raise TypeError("frame IDs must be strings or integers")
    try:
        unique = set(ids)
    except TypeError as error:
        raise TypeError("frame IDs must be hashable") from error
    if len(unique) != len(ids):
        raise ValueError("frame IDs must be unique")
    return ids


@dataclass(frozen=True)
class DataScope:
    """Selection shared by maps, pole figures, and tables.

    Parameters
    ----------
    patterns
        ``"best"`` selects the lowest pattern rank in each frame, ``"all"``
        selects every pattern, and a tuple selects explicit pattern ranks.
    min_indexed
        Minimum number of indexed assignments required for a selected pattern.
    min_detected
        Optional minimum number of detected peaks required for its frame.
    """

    patterns: PatternSelection = "best"
    min_indexed: int = 3
    min_detected: int | None = None

    def __post_init__(self):
        patterns = self.patterns
        if isinstance(patterns, list):
            patterns = tuple(patterns)
            object.__setattr__(self, "patterns", patterns)
        if patterns not in ("best", "all"):
            if not isinstance(patterns, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in patterns
            ):
                raise ValueError("patterns must be 'best', 'all', or a tuple of nonnegative ranks")
            if len(set(patterns)) != len(patterns):
                raise ValueError("pattern ranks must be unique")
        if isinstance(self.min_indexed, bool) or not isinstance(self.min_indexed, int) or self.min_indexed < 0:
            raise ValueError("min_indexed must be a nonnegative integer")
        if (
            self.min_detected is not None
            and (
                isinstance(self.min_detected, bool)
                or not isinstance(self.min_detected, int)
                or self.min_detected < 0
            )
        ):
            raise ValueError("min_detected must be a nonnegative integer or None")

    def pattern_mask(self, dataset: "VisualizationDataset") -> np.ndarray:
        """Return a mask selecting pattern rows from a dataset."""
        count = len(dataset.pattern_indices)
        selected = np.ones(count, dtype=bool)
        if self.patterns == "best":
            selected[:] = False
            for frame_index in np.unique(dataset.pattern_frame_indices):
                rows = np.flatnonzero(dataset.pattern_frame_indices == frame_index)
                if len(rows):
                    selected[rows[np.argmin(dataset.pattern_indices[rows])]] = True
        elif self.patterns != "all":
            selected &= np.isin(dataset.pattern_indices, self.patterns)
        selected &= dataset.pattern_n_indexed >= self.min_indexed
        if self.min_detected is not None:
            selected &= dataset.frame_n_peaks[dataset.pattern_frame_indices] >= self.min_detected
        return selected


@dataclass(frozen=True)
class ResultSet:
    """Ordered indexing results with shared crystal and geometry context."""

    results: tuple[FrameResult, ...]
    frame_ids: tuple[FrameId, ...] | None = None
    crystal: Crystal | None = None
    geometry: Geometry | None = None

    def __post_init__(self):
        results = tuple(self.results)
        if any(not isinstance(result, FrameResult) for result in results):
            raise TypeError("results must contain only FrameResult objects")
        object.__setattr__(self, "results", results)
        ids = range(len(results)) if self.frame_ids is None else self.frame_ids
        object.__setattr__(self, "frame_ids", _validate_frame_ids(ids, len(results)))
        if self.crystal is not None and not isinstance(self.crystal, Crystal):
            raise TypeError("crystal must be a Crystal or None")
        if self.geometry is not None and not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be a Geometry or None")

    @classmethod
    def from_indexer(cls, indexer, results: Iterable[FrameResult], *, frame_ids=None):
        """Construct a result set using an indexer's shared context."""
        return cls(
            tuple(results),
            frame_ids=frame_ids,
            crystal=indexer.crystal_model,
            geometry=indexer.geometry,
        )

    def to_visualization(self) -> "VisualizationDataset":
        """Normalize this collection for visualization and tabular use."""
        return VisualizationDataset.from_result_set(self)


@dataclass(frozen=True)
class VisualizationDataset:
    """Immutable columnar snapshot of indexing results for visualization."""

    frame_ids: tuple[FrameId, ...]
    sample_positions: np.ndarray
    depths: np.ndarray
    frame_n_peaks: np.ndarray
    scan_numbers: np.ndarray
    energies_kev: np.ndarray
    detector_ids: tuple[str | None, ...]
    image_shapes: np.ndarray
    starts: np.ndarray
    groups: np.ndarray
    input_images: tuple[str | None, ...]
    images: tuple[np.ndarray | None, ...] = field(repr=False)
    peak_frame_indices: np.ndarray
    peak_indices: np.ndarray
    peaks: np.ndarray
    pattern_frame_indices: np.ndarray
    pattern_indices: np.ndarray
    pattern_rotations: np.ndarray
    pattern_reciprocals: np.ndarray
    pattern_goodness: np.ndarray
    pattern_rms_error_deg: np.ndarray
    pattern_n_indexed: np.ndarray
    assignment_pattern_rows: np.ndarray
    assignment_peak_indices: np.ndarray
    assignment_hkl: np.ndarray
    assignment_error_deg: np.ndarray
    assignment_energy_kev: np.ndarray
    assignment_predicted_intensity: np.ndarray
    crystal: Crystal | None = None
    geometry: Geometry | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        frame_count = len(self.frame_ids)
        object.__setattr__(self, "frame_ids", _validate_frame_ids(self.frame_ids, frame_count))
        specifications = {
            "sample_positions": (np.float64, (frame_count, 3)),
            "depths": (np.float64, (frame_count,)),
            "frame_n_peaks": (np.int64, (frame_count,)),
            "scan_numbers": (np.float64, (frame_count,)),
            "energies_kev": (np.float64, (frame_count,)),
            "image_shapes": (np.int64, (frame_count, 2)),
            "starts": (np.int64, (frame_count, 2)),
            "groups": (np.int64, (frame_count, 2)),
        }
        for name, (dtype, shape) in specifications.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype, shape=shape, name=name))
        if (
            len(self.detector_ids) != frame_count
            or len(self.input_images) != frame_count
            or len(self.images) != frame_count
        ):
            raise ValueError("detector_ids, input_images, and images must align with frame_ids")
        object.__setattr__(self, "detector_ids", tuple(self.detector_ids))
        object.__setattr__(self, "input_images", tuple(self.input_images))
        owned_images = []
        for image in self.images:
            if image is None:
                owned_images.append(None)
            else:
                value = _readonly(image, name="image")
                owned_images.append(value)
        object.__setattr__(self, "images", tuple(owned_images))

        peak_count = len(self.peak_indices)
        pattern_count = len(self.pattern_indices)
        assignment_count = len(self.assignment_peak_indices)
        arrays = {
            "peak_frame_indices": (np.int64, (peak_count,)),
            "peak_indices": (np.int64, (peak_count,)),
            "peaks": (None, (peak_count,)),
            "pattern_frame_indices": (np.int64, (pattern_count,)),
            "pattern_indices": (np.int64, (pattern_count,)),
            "pattern_rotations": (np.float64, (pattern_count, 3, 3)),
            "pattern_reciprocals": (np.float64, (pattern_count, 3, 3)),
            "pattern_goodness": (np.float64, (pattern_count,)),
            "pattern_rms_error_deg": (np.float64, (pattern_count,)),
            "pattern_n_indexed": (np.int64, (pattern_count,)),
            "assignment_pattern_rows": (np.int64, (assignment_count,)),
            "assignment_peak_indices": (np.int64, (assignment_count,)),
            "assignment_hkl": (np.int64, (assignment_count, 3)),
            "assignment_error_deg": (np.float64, (assignment_count,)),
            "assignment_energy_kev": (np.float64, (assignment_count,)),
            "assignment_predicted_intensity": (np.float64, (assignment_count,)),
        }
        for name, (dtype, shape) in arrays.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype, shape=shape, name=name))
        if np.any(self.peak_frame_indices < 0) or np.any(self.peak_frame_indices >= frame_count):
            raise ValueError("peak frame indices are out of range")
        if np.any(self.pattern_frame_indices < 0) or np.any(self.pattern_frame_indices >= frame_count):
            raise ValueError("pattern frame indices are out of range")
        if np.any(self.assignment_pattern_rows < 0) or np.any(self.assignment_pattern_rows >= pattern_count):
            raise ValueError("assignment pattern rows are out of range")

    @classmethod
    def from_result_set(cls, result_set: ResultSet) -> "VisualizationDataset":
        """Create a normalized snapshot from complete modern results."""
        results = result_set.results
        positions = np.full((len(results), 3), np.nan)
        depths = np.full(len(results), np.nan)
        image_shapes = np.empty((len(results), 2), dtype=int)
        starts = np.empty((len(results), 2), dtype=int)
        groups = np.empty((len(results), 2), dtype=int)
        peak_frames = []
        peak_indices = []
        peak_arrays = []
        pattern_frames = []
        pattern_indices = []
        rotations = []
        reciprocals = []
        goodness = []
        rms_error = []
        n_indexed = []
        assignment_patterns = []
        assignment_peaks = []
        assignment_hkl = []
        assignment_error = []
        assignment_energy = []
        assignment_intensity = []

        for frame_index, result in enumerate(results):
            position = result.metadata.get("sample_position")
            if position is not None:
                value = np.asarray(position, dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"sample_position for frame {result_set.frame_ids[frame_index]!r} must be a finite 3-vector")
                positions[frame_index] = value
            if result.depth is not None:
                depths[frame_index] = result.depth
            image_shapes[frame_index] = result.image_shape
            starts[frame_index] = result.start
            groups[frame_index] = result.group
            peak_frames.extend([frame_index] * result.n_peaks)
            peak_indices.extend(range(result.n_peaks))
            peak_arrays.append(result.peaks)

            for pattern_index, pattern in enumerate(result.patterns):
                pattern_row = len(pattern_indices)
                pattern_frames.append(frame_index)
                pattern_indices.append(pattern_index)
                rotations.append(pattern.rotation)
                reciprocals.append(pattern.recip)
                goodness.append(pattern.goodness)
                rms_error.append(pattern.rms_error_deg)
                n_indexed.append(pattern.n_indexed)
                if np.any(pattern.pk_index < 0) or np.any(pattern.pk_index >= result.n_peaks):
                    raise ValueError(f"pattern {pattern_index} in frame {result_set.frame_ids[frame_index]!r} has invalid peak indices")
                assignment_patterns.extend([pattern_row] * pattern.n_indexed)
                assignment_peaks.extend(pattern.pk_index)
                assignment_hkl.extend(pattern.hkl)
                assignment_error.extend(pattern.err_deg)
                assignment_energy.extend(pattern.energy_kev)
                assignment_intensity.extend(pattern.pred_intens)

        peak_dtype = results[0].peaks.dtype if results else np.dtype([])
        peaks = np.concatenate(peak_arrays) if peak_arrays else np.empty(0, dtype=peak_dtype)
        return cls(
            frame_ids=result_set.frame_ids,
            sample_positions=positions,
            depths=depths,
            frame_n_peaks=np.asarray([result.n_peaks for result in results]),
            scan_numbers=np.asarray([
                result.metadata.get("scan_number", np.nan) for result in results
            ], dtype=float),
            energies_kev=np.asarray([
                result.metadata.get("energy_kev", np.nan) for result in results
            ], dtype=float),
            detector_ids=tuple(result.metadata.get("detector_id") for result in results),
            image_shapes=image_shapes,
            starts=starts,
            groups=groups,
            input_images=tuple(result.input_image for result in results),
            images=tuple(result.image for result in results),
            peak_frame_indices=np.asarray(peak_frames),
            peak_indices=np.asarray(peak_indices),
            peaks=peaks,
            pattern_frame_indices=np.asarray(pattern_frames),
            pattern_indices=np.asarray(pattern_indices),
            pattern_rotations=np.asarray(rotations, dtype=float).reshape((-1, 3, 3)),
            pattern_reciprocals=np.asarray(reciprocals, dtype=float).reshape((-1, 3, 3)),
            pattern_goodness=np.asarray(goodness),
            pattern_rms_error_deg=np.asarray(rms_error),
            pattern_n_indexed=np.asarray(n_indexed),
            assignment_pattern_rows=np.asarray(assignment_patterns),
            assignment_peak_indices=np.asarray(assignment_peaks),
            assignment_hkl=np.asarray(assignment_hkl, dtype=int).reshape((-1, 3)),
            assignment_error_deg=np.asarray(assignment_error),
            assignment_energy_kev=np.asarray(assignment_energy),
            assignment_predicted_intensity=np.asarray(assignment_intensity),
            crystal=result_set.crystal,
            geometry=result_set.geometry,
        )

    @property
    def n_frames(self) -> int:
        """Number of frames in the dataset."""
        return len(self.frame_ids)

    @property
    def n_patterns(self) -> int:
        """Number of indexed patterns in the dataset."""
        return len(self.pattern_indices)

    @property
    def n_assignments(self) -> int:
        """Number of pattern-to-peak assignments in the dataset."""
        return len(self.assignment_peak_indices)

    def pattern_ids(self, scope: DataScope | None = None) -> tuple[tuple[FrameId, int], ...]:
        """Return stable pattern identities selected by a data scope."""
        mask = (scope or DataScope()).pattern_mask(self)
        rows = np.flatnonzero(mask)
        return tuple(
            (self.frame_ids[self.pattern_frame_indices[row]], int(self.pattern_indices[row]))
            for row in rows
        )
