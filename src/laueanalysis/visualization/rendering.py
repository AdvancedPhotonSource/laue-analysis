"""Plotly renderers for prepared Laue visualization data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from .preparation import (
    DetectorViewData,
    MapData,
    PoleFigureData,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
)

_BACKGROUND = "rgb(245, 245, 245)"


def _rgb(values):
    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        array = array.reshape(1, 3)
    colors = []
    for value in array:
        if np.isfinite(value).all():
            red, green, blue = np.clip(np.rint(value * 255), 0, 255).astype(int)
            colors.append(f"rgb({red},{green},{blue})")
        else:
            colors.append("rgb(150,150,150)")
    return colors


def _customdata(frame_ids, pattern_indices=None, peak_indices=None):
    count = len(frame_ids)
    patterns = [None] * count if pattern_indices is None else pattern_indices
    peaks = [None] * count if peak_indices is None else peak_indices
    return [
        [frame_id, None if pattern is None else int(pattern), None if peak is None else int(peak)]
        for frame_id, pattern, peak in zip(frame_ids, patterns, peaks, strict=True)
    ]


def _apply_updates(figure, roles, allowed_roles, trace_update, layout_update):
    if trace_update is not None:
        if not isinstance(trace_update, Mapping):
            raise TypeError("trace_update must be a mapping or None")
        unknown = set(trace_update) - set(allowed_roles)
        if unknown:
            raise ValueError(
                f"unknown trace roles {tuple(sorted(unknown))}; choose from {tuple(allowed_roles)}"
            )
        for role, update in trace_update.items():
            if not isinstance(update, Mapping):
                raise TypeError(f"trace update for role {role!r} must be a mapping")
            for index in roles.get(role, ()):
                figure.data[index].update(update)
    if layout_update is not None:
        if not isinstance(layout_update, Mapping):
            raise TypeError("layout_update must be a mapping or None")
        figure.update_layout(layout_update)
    return figure


def _add_empty_annotation(figure, text):
    figure.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "rgb(100,100,100)", "size": 14},
    )


def plot_map(
    source,
    *,
    axes=("x", "y"),
    color="n_indexed",
    scope=None,
    surface=None,
    marker_size=10,
    layout_update=None,
    trace_update=None,
):
    """Render a two- or three-dimensional spatial map with Plotly.

    ``source`` can be a :class:`MapData`, :class:`ResultSet`, or
    :class:`VisualizationDataset`. Semantic trace roles are ``"data"`` and
    ``"unindexed"``.
    """
    data = source if isinstance(source, MapData) else prepare_map(
        source, axes=axes, color=color, scope=scope, surface=surface
    )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")

    figure = go.Figure()
    roles = {"data": [], "unindexed": []}
    dimensions = data.coordinates.shape[1]
    masks = (("data", data.indexed), ("unindexed", ~data.indexed))
    for role, mask in masks:
        if not np.any(mask):
            continue
        marker = {"size": marker_size, "line": {"width": 0}}
        if role == "unindexed":
            marker["color"] = "rgb(150,150,150)"
        elif data.color_kind == "rgb":
            marker["color"] = _rgb(data.colors[mask])
        else:
            marker.update({
                "color": data.colors[mask],
                "colorscale": data.palette,
                "colorbar": {"title": {"text": data.color_label}},
                "showscale": True,
            })
            if data.color_limits is not None:
                marker["cmin"], marker["cmax"] = data.color_limits
        customdata = _customdata(
            [data.frame_ids[index] for index in np.flatnonzero(mask)],
            data.pattern_indices[mask],
        )
        common = {
            "mode": "markers",
            "name": "Data" if role == "data" else "Unindexed",
            "marker": marker,
            "customdata": customdata,
            "meta": {"role": role},
            "uid": f"map-{role}",
            "hovertemplate": (
                "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                + (f"{data.color_label}: %{{marker.color:.4g}}<br>" if data.color_kind == "scalar" and role == "data" else "")
                + "<extra></extra>"
            ),
        }
        coordinates = data.coordinates[mask]
        trace = (
            go.Scattergl(x=coordinates[:, 0], y=coordinates[:, 1], **common)
            if dimensions == 2
            else go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                **common,
            )
        )
        figure.add_trace(trace)
        roles[role].append(len(figure.data) - 1)

    if dimensions == 2:
        figure.update_layout(
            xaxis={"title": {"text": data.axis_labels[0]}},
            yaxis={"title": {"text": data.axis_labels[1]}, "scaleanchor": "x", "scaleratio": 1},
            plot_bgcolor=_BACKGROUND,
            margin={"l": 60, "r": 30, "t": 35, "b": 55},
            dragmode="lasso",
            uirevision="map-2d",
        )
    else:
        figure.update_layout(
            scene={
                "xaxis": {"title": {"text": data.axis_labels[0]}},
                "yaxis": {"title": {"text": data.axis_labels[1]}},
                "zaxis": {"title": {"text": data.axis_labels[2]}},
                "aspectmode": "data",
                "bgcolor": _BACKGROUND,
            },
            margin={"l": 0, "r": 0, "t": 35, "b": 0},
            uirevision="map-3d",
        )
    if not len(data.coordinates):
        _add_empty_annotation(figure, "No patterns match the selected scope.")
    return _apply_updates(
        figure, roles, ("data", "unindexed"), trace_update, layout_update
    )


def plot_pole_figure(
    source,
    *,
    hkl=(1, 0, 0),
    scope=None,
    surface=None,
    color="hsv_position",
    center=(0.0, 0.0),
    color_radius_deg=22.5,
    marker_size=7,
    hover_point_limit=100_000,
    layout_update=None,
    trace_update=None,
):
    """Render an upper-hemisphere stereographic pole figure with Plotly.

    Semantic trace roles are ``"data"``, ``"boundary"``, and ``"reference"``.
    """
    data = source if isinstance(source, PoleFigureData) else prepare_pole_figure(
        source,
        hkl=hkl,
        scope=scope,
        surface=surface,
        color=color,
        center=center,
        color_radius_deg=color_radius_deg,
    )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")
    if hover_point_limit is not None and (
        isinstance(hover_point_limit, bool)
        or not isinstance(hover_point_limit, int)
        or hover_point_limit < 0
    ):
        raise ValueError("hover_point_limit must be a nonnegative integer or None")

    figure = go.Figure()
    roles = {"data": [], "boundary": [], "reference": []}
    hover_disabled = hover_point_limit is not None and len(data.points) > hover_point_limit
    if len(data.points):
        marker_color = _rgb(data.colors)
        if data.color_kind == "uniform":
            marker_color = marker_color[0]
        figure.add_trace(go.Scattergl(
            x=data.points[:, 0],
            y=data.points[:, 1],
            mode="markers",
            marker={"size": marker_size, "color": marker_color, "line": {"width": 0}},
            customdata=_customdata(data.frame_ids, data.pattern_indices),
            name=f"{{{''.join(str(value) for value in data.hkl)}}}",
            hoverinfo="skip" if hover_disabled else None,
            hovertemplate=None if hover_disabled else (
                "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                "x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>"
            ),
            meta={"role": "data"},
            uid="pole-data",
        ))
        roles["data"].append(len(figure.data) - 1)

    theta = np.linspace(0, 2 * np.pi, 241)
    figure.add_trace(go.Scattergl(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line={"color": "black", "width": 1},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "boundary"},
        uid="pole-boundary",
    ))
    roles["boundary"].append(len(figure.data) - 1)

    if data.color_radius is not None:
        figure.add_trace(go.Scattergl(
            x=data.center[0] + data.color_radius * np.cos(theta),
            y=data.center[1] + data.color_radius * np.sin(theta),
            mode="lines",
            line={"color": "black", "width": 1, "dash": "dot"},
            showlegend=False,
            hoverinfo="skip",
            meta={"role": "reference"},
            uid="pole-reference-radius",
        ))
        roles["reference"].append(len(figure.data) - 1)
    figure.add_trace(go.Scattergl(
        x=[data.center[0]],
        y=[data.center[1]],
        mode="markers",
        marker={"symbol": "cross", "size": 8, "color": "black"},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "reference"},
        uid="pole-reference-center",
    ))
    roles["reference"].append(len(figure.data) - 1)

    figure.update_layout(
        xaxis={"range": [-1.1, 1.1], "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"range": [-1.1, 1.1], "showgrid": False, "zeroline": False},
        plot_bgcolor=_BACKGROUND,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        dragmode="lasso",
        uirevision="pole-figure",
    )
    if not len(data.points):
        _add_empty_annotation(figure, "No poles match the selected scope.")
    elif hover_disabled:
        figure.add_annotation(
            text=f"Point hover disabled above {hover_point_limit:,} poles.",
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
        )
    return _apply_updates(
        figure, roles, ("data", "boundary", "reference"), trace_update, layout_update
    )


def plot_detector_view(
    source,
    *,
    frame_id=None,
    patterns="all",
    image=None,
    detector_index=None,
    show_detected=True,
    show_indexed=True,
    show_hkl_labels=False,
    marker_size=8,
    image_colorscale="Gray",
    image_limits=None,
    image_opacity=1.0,
    layout_update=None,
    trace_update=None,
):
    """Render a detector image and reflection overlays with Plotly.

    Semantic trace roles are ``"image"``, ``"boundary"``, ``"detected"``, and
    ``"indexed"``.
    """
    if isinstance(source, DetectorViewData):
        data = source
    else:
        if frame_id is None:
            raise TypeError("frame_id is required when source is not DetectorViewData")
        data = prepare_detector_view(
            source,
            frame_id=frame_id,
            patterns=patterns,
            image=image,
            detector_index=detector_index,
        )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")
    if not np.isfinite(image_opacity) or not 0 <= image_opacity <= 1:
        raise ValueError("image_opacity must be between 0 and 1")
    if image_limits is not None and (
        len(image_limits) != 2
        or not np.isfinite(image_limits).all()
        or image_limits[0] >= image_limits[1]
    ):
        raise ValueError("image_limits must contain two finite increasing values")

    figure = go.Figure()
    roles = {"image": [], "boundary": [], "detected": [], "indexed": []}
    if data.image is not None:
        heatmap = {
            "z": data.image,
            "colorscale": image_colorscale,
            "opacity": image_opacity,
            "colorbar": {"title": {"text": "Intensity"}},
            "name": "Detector image",
            "hovertemplate": "x: %{x}<br>y: %{y}<br>I: %{z:.4g}<extra></extra>",
            "meta": {"role": "image"},
            "uid": "detector-image",
        }
        if image_limits is not None:
            heatmap["zmin"], heatmap["zmax"] = image_limits
        figure.add_trace(go.Heatmap(**heatmap))
        roles["image"].append(len(figure.data) - 1)

    width, height = data.extent
    figure.add_trace(go.Scatter(
        x=[0, width, width, 0, 0],
        y=[0, 0, height, height, 0],
        mode="lines",
        line={"color": "rgb(120,120,120)", "width": 1},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "boundary"},
        uid="detector-boundary",
    ))
    roles["boundary"].append(len(figure.data) - 1)

    if show_detected and len(data.measured_xy):
        figure.add_trace(go.Scattergl(
            x=data.measured_xy[:, 0],
            y=data.measured_xy[:, 1],
            mode="markers",
            name=f"Detected ({len(data.measured_xy)})",
            marker={
                "size": marker_size,
                "symbol": "square-open",
                "color": "rgb(0,180,210)",
                "line": {"width": 1.5, "color": "rgb(0,180,210)"},
            },
            customdata=_customdata(
                [data.frame_id] * len(data.measured_xy),
                peak_indices=data.measured_peak_indices,
            ),
            hovertemplate=(
                "frame: %{customdata[0]}<br>peak: %{customdata[2]}<br>"
                "x: %{x:.2f} px<br>y: %{y:.2f} px<extra></extra>"
            ),
            meta={"role": "detected"},
            uid="detector-detected",
        ))
        roles["detected"].append(len(figure.data) - 1)

    if show_indexed:
        for pattern in data.patterns:
            finite = np.isfinite(pattern.predicted_xy).all(axis=1)
            if not np.any(finite):
                continue
            positions = pattern.predicted_xy[finite]
            hkl = pattern.hkl[finite]
            peaks = pattern.measured_peak_indices[finite]
            on_detector = (
                (positions[:, 0] >= 0)
                & (positions[:, 0] <= width)
                & (positions[:, 1] >= 0)
                & (positions[:, 1] <= height)
            )
            for mask, suffix, visible in (
                (on_detector, "on-detector", True),
                (~on_detector, "off-detector", "legendonly"),
            ):
                if not np.any(mask):
                    continue
                selected = positions[mask]
                selected_hkl = hkl[mask]
                stable = _customdata(
                    [data.frame_id] * len(selected),
                    [pattern.pattern_index] * len(selected),
                    peaks[mask],
                )
                customdata = [
                    identity + [int(value) for value in reflection]
                    for identity, reflection in zip(stable, selected_hkl, strict=True)
                ]
                labels = [
                    f"({value[0]} {value[1]} {value[2]})" for value in selected_hkl
                ]
                figure.add_trace(go.Scattergl(
                    x=selected[:, 0],
                    y=selected[:, 1],
                    mode="markers+text" if show_hkl_labels else "markers",
                    text=labels if show_hkl_labels else None,
                    textposition="top right",
                    visible=visible,
                    name=f"Pattern {pattern.pattern_index} ({suffix})",
                    marker={
                        "size": marker_size + 4,
                        "symbol": "x-thin-open",
                        "color": "rgb(220,50,47)",
                        "line": {"width": 1.5, "color": "rgb(220,50,47)"},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                        "peak: %{customdata[2]}<br>hkl: (%{customdata[3]} "
                        "%{customdata[4]} %{customdata[5]})<br>"
                        "x: %{x:.2f} px<br>y: %{y:.2f} px<extra></extra>"
                    ),
                    meta={"role": "indexed"},
                    uid=f"detector-indexed-{pattern.pattern_index}-{suffix}",
                ))
                roles["indexed"].append(len(figure.data) - 1)

    figure.update_layout(
        xaxis={
            "title": {"text": "X pixel"},
            "range": [0, width],
            "scaleanchor": "y",
            "scaleratio": 1,
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Y pixel"},
            "range": [height, 0],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        plot_bgcolor="white",
        margin={"l": 55, "r": 30, "t": 40, "b": 50},
        uirevision=f"detector-{width:g}x{height:g}",
    )
    if not len(data.measured_xy) and not data.patterns:
        _add_empty_annotation(figure, "This frame has no detected peaks or indexed patterns.")
    return _apply_updates(
        figure,
        roles,
        ("image", "boundary", "detected", "indexed"),
        trace_update,
        layout_update,
    )
