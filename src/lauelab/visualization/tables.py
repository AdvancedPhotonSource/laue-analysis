# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Typed tabular views of normalized indexing results."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .data import DataScope, ResultSet, VisualizationDataset


def _dataset(source):
    if isinstance(source, ResultSet):
        return source.to_visualization()
    if isinstance(source, VisualizationDataset):
        return source
    raise TypeError("source must be a ResultSet or VisualizationDataset")


@dataclass(frozen=True)
class Table:
    """Immutable named columns with direct pandas conversion."""

    columns: Mapping[str, np.ndarray]

    def __post_init__(self):
        normalized = {}
        length = None
        for name, values in self.columns.items():
            array = np.array(values, copy=True)
            if array.ndim != 1:
                raise ValueError(f"table column {name!r} must be one-dimensional")
            if length is None:
                length = len(array)
            elif len(array) != length:
                raise ValueError("table columns must have equal lengths")
            array.setflags(write=False)
            normalized[name] = array
        object.__setattr__(self, "columns", MappingProxyType(normalized))

    def __len__(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __getitem__(self, name):
        return self.columns[name]

    def to_dataframe(self):
        """Return a new pandas DataFrame containing the table columns."""
        import pandas as pd

        return pd.DataFrame({name: values.copy() for name, values in self.columns.items()})

    def _repr_html_(self):
        """Return the pandas HTML representation used by notebook displays."""
        return self.to_dataframe()._repr_html_()


def _selected_pattern_rows(dataset, scope):
    return np.flatnonzero((scope or DataScope()).pattern_mask(dataset))


def _selected_frame_mask(dataset, scope):
    scope = scope or DataScope()
    if scope.patterns == "all_frames":
        mask = np.ones(dataset.n_frames, dtype=bool)
        if scope.min_detected is not None:
            mask &= dataset.frame_n_peaks >= scope.min_detected
        return mask
    rows = _selected_pattern_rows(dataset, scope)
    mask = np.zeros(dataset.n_frames, dtype=bool)
    mask[dataset.pattern_frame_indices[rows]] = True
    return mask


def _frame_columns(dataset, frame_indices):
    return {
        "frame_id": np.asarray([dataset.frame_ids[index] for index in frame_indices], dtype=object),
        "scan_number": dataset.scan_numbers[frame_indices],
        "x_um": dataset.sample_positions[frame_indices, 0],
        "y_um": dataset.sample_positions[frame_indices, 1],
        "z_um": dataset.sample_positions[frame_indices, 2],
        "depth_um": dataset.depths[frame_indices],
        "frame_energy_kev": dataset.energies_kev[frame_indices],
    }


def peak_table(source, *, scope=None):
    """Return one row per detected peak in frames selected by ``scope``."""
    dataset = _dataset(source)
    selected = _selected_frame_mask(dataset, scope)
    rows = np.flatnonzero(selected[dataset.peak_frame_indices])
    frame_indices = dataset.peak_frame_indices[rows]
    columns = _frame_columns(dataset, frame_indices)
    columns.update({"peak_index": dataset.peak_indices[rows]})
    for name in dataset.peaks.dtype.names or ():
        values = dataset.peaks[name][rows]
        if values.ndim == 1:
            columns[name] = values
        else:
            for component in range(values.shape[1]):
                columns[f"{name}_{'xyz'[component]}"] = values[:, component]
    return Table(columns)


def pattern_table(source, *, scope=None):
    """Return one row per indexed pattern selected by ``scope``."""
    dataset = _dataset(source)
    rows = _selected_pattern_rows(dataset, scope)
    frame_indices = dataset.pattern_frame_indices[rows]
    columns = _frame_columns(dataset, frame_indices)
    columns.update({
        "pattern_index": dataset.pattern_indices[rows],
        "n_indexed": dataset.pattern_n_indexed[rows],
        "goodness": dataset.pattern_goodness[rows],
        "rms_error_deg": dataset.pattern_rms_error_deg[rows],
        "indexed_fraction": np.divide(
            dataset.pattern_n_indexed[rows],
            dataset.frame_n_peaks[frame_indices],
            out=np.zeros(len(rows), dtype=float),
            where=dataset.frame_n_peaks[frame_indices] > 0,
        ),
    })
    for prefix, values in (
        ("rotation", dataset.pattern_rotations[rows]),
        ("reciprocal", dataset.pattern_reciprocals[rows]),
    ):
        for row in range(3):
            for column in range(3):
                columns[f"{prefix}_{row}{column}"] = values[:, row, column]
    return Table(columns)


def assignment_table(source, *, scope=None):
    """Return one row per selected pattern-to-peak assignment."""
    dataset = _dataset(source)
    pattern_rows = _selected_pattern_rows(dataset, scope)
    rows = np.flatnonzero(np.isin(dataset.assignment_pattern_rows, pattern_rows))
    selected_pattern_rows = dataset.assignment_pattern_rows[rows]
    frame_indices = dataset.pattern_frame_indices[selected_pattern_rows]
    columns = _frame_columns(dataset, frame_indices)
    columns.update({
        "pattern_index": dataset.pattern_indices[selected_pattern_rows],
        "peak_index": dataset.assignment_peak_indices[rows],
        "h": dataset.assignment_hkl[rows, 0],
        "k": dataset.assignment_hkl[rows, 1],
        "l": dataset.assignment_hkl[rows, 2],
        "error_deg": dataset.assignment_error_deg[rows],
        "energy_kev": dataset.assignment_energy_kev[rows],
        "predicted_intensity": dataset.assignment_predicted_intensity[rows],
    })
    return Table(columns)


def indexed_peak_table(source, *, scope=None):
    """Return assignments joined with their detected peak and pattern values."""
    dataset = _dataset(source)
    assignments = assignment_table(dataset, scope=scope)
    peak_rows = {
        (int(frame), int(peak)): row
        for row, (frame, peak) in enumerate(zip(
            dataset.peak_frame_indices, dataset.peak_indices, strict=True
        ))
    }
    frame_lookup = {frame_id: index for index, frame_id in enumerate(dataset.frame_ids)}
    rows = np.asarray([
        peak_rows[(frame_lookup[frame_id], int(peak_index))]
        for frame_id, peak_index in zip(assignments["frame_id"], assignments["peak_index"], strict=True)
    ], dtype=int)
    columns = dict(assignments.columns)
    for name in dataset.peaks.dtype.names or ():
        values = dataset.peaks[name][rows]
        if values.ndim == 1:
            columns[name] = values
        else:
            for component in range(values.shape[1]):
                columns[f"{name}_{'xyz'[component]}"] = values[:, component]
    pattern_lookup = {
        (dataset.frame_ids[frame], int(pattern)): row
        for row, (frame, pattern) in enumerate(zip(
            dataset.pattern_frame_indices, dataset.pattern_indices, strict=True
        ))
    }
    pattern_rows = np.asarray([
        pattern_lookup[(frame_id, int(pattern_index))]
        for frame_id, pattern_index in zip(
            assignments["frame_id"], assignments["pattern_index"], strict=True
        )
    ], dtype=int)
    columns.update({
        "goodness": dataset.pattern_goodness[pattern_rows],
        "rms_error_deg": dataset.pattern_rms_error_deg[pattern_rows],
        "pattern_n_indexed": dataset.pattern_n_indexed[pattern_rows],
    })
    return Table(columns)
