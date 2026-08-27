"""Stateless conversion of Plotly events to scientific identities."""

from __future__ import annotations

from dataclasses import dataclass

from .data import FrameId


@dataclass(frozen=True)
class PlotlySelection:
    """Stable identities extracted from a Plotly click or selection event."""

    frame_ids: tuple[FrameId, ...] = ()
    pattern_ids: tuple[tuple[FrameId, int], ...] = ()
    peak_ids: tuple[tuple[FrameId, int], ...] = ()


def selection_from_plotly(event_data) -> PlotlySelection:
    """Extract stable identities from Plotly ``clickData`` or ``selectedData``.

    The first three ``customdata`` values must be ``frame_id``,
    ``pattern_index``, and ``peak_index``. A missing pattern or peak value is
    represented by `None`. Duplicate identities are removed in event order.
    """
    if event_data is None:
        return PlotlySelection()
    if not isinstance(event_data, dict):
        raise TypeError("event_data must be a mapping or None")
    points = event_data.get("points", ())
    if points is None:
        return PlotlySelection()
    if not isinstance(points, (list, tuple)):
        raise ValueError("event_data['points'] must be a sequence")

    frames = []
    patterns = []
    peaks = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("each Plotly event point must be a mapping")
        customdata = point.get("customdata")
        if customdata is None:
            continue
        if not isinstance(customdata, (list, tuple)) or len(customdata) < 3:
            raise ValueError("point customdata must contain frame, pattern, and peak identity")
        frame_id, pattern_index, peak_index = customdata[:3]
        if isinstance(frame_id, bool) or not isinstance(frame_id, (str, int)):
            raise ValueError("customdata frame IDs must be strings or integers")
        if frame_id not in frames:
            frames.append(frame_id)
        if pattern_index is not None:
            if isinstance(pattern_index, bool) or not isinstance(pattern_index, int):
                raise ValueError("customdata pattern indices must be integers or None")
            identity = (frame_id, pattern_index)
            if identity not in patterns:
                patterns.append(identity)
        if peak_index is not None:
            if isinstance(peak_index, bool) or not isinstance(peak_index, int):
                raise ValueError("customdata peak indices must be integers or None")
            identity = (frame_id, peak_index)
            if identity not in peaks:
                peaks.append(identity)
    return PlotlySelection(tuple(frames), tuple(patterns), tuple(peaks))
