from dataclasses import replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest


from laueanalysis.indexing import FrameResult, Pattern, load_crystal, load_geometry
from laueanalysis.indexing.indexer import PEAK_DTYPE
from laueanalysis.visualization import (
    DataScope,
    DetectorSimulationData,
    PlotlySelection,
    ResultSet,
    Axis,
    plot_detector_view,
    plot_map,
    plot_pole_figure,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
    selection_from_plotly,
)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_FILE = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL_FILE = ROOT / "tests/config/Ni.xml"


def _pattern(count):
    return Pattern(
        euler_deg=np.zeros(3),
        rotation=np.eye(3),
        reciprocal=np.eye(3),
        goodness=10 + count,
        rms_error_deg=0.1,
        hkl=np.tile([1, 0, -1], (count, 1)),
        pk_index=np.arange(count, dtype=np.int32),
        err_deg=np.arange(count, dtype=float) / 10,
        energy_kev=np.arange(count, dtype=float) + 10,
        pred_intens=np.arange(count, dtype=float) + 100,
    )


def _result(patterns, position):
    peaks = np.zeros(4, dtype=PEAK_DTYPE)
    peaks["fit_x"] = [10, 20, 30, 40]
    peaks["fit_y"] = [11, 21, 31, 41]
    peaks["intens"] = [100, 200, 300, 400]
    return FrameResult(
        peaks=peaks,
        patterns=patterns,
        threshold_used=10,
        total_sum=100,
        sum_above_threshold=50,
        num_above_threshold=4,
        peaksearch_seconds=0.1,
        indexing_seconds=0.2,
        metadata={"sample_position": position, "detector_id": "PE1621 723-3335"},
        image_shape=(8, 10),
        image=np.arange(80, dtype=np.uint16).reshape(8, 10),
    )


def _result_set():
    return ResultSet(
        (
            _result((_pattern(3), _pattern(2)), (0, 1, 2)),
            _result((_pattern(4),), (3, 4, 5)),
        ),
        frame_ids=("a", "b"),
        crystal=load_crystal(CRYSTAL_FILE),
        geometry=load_geometry(GEOMETRY_FILE),
    )


def _roles(figure):
    return [trace.meta["role"] for trace in figure.data]


def test_plot_map_renders_prepared_2d_data_with_stable_identity():
    figure = plot_map(
        prepare_map(_result_set(), axes=("X", "H"), color="goodness"),
        marker_size=12,
        trace_update={"data": {"marker": {"symbol": "circle"}}},
        layout_update={"template": "plotly_white"},
    )

    assert isinstance(figure, go.Figure)
    assert figure.data[0].type == "scattergl"
    assert _roles(figure) == ["data"]
    assert list(figure.data[0].customdata[0]) == ["a", 0, None]
    assert figure.data[0].marker.size == 12
    assert figure.data[0].marker.symbol == "circle"
    assert figure.layout.xaxis.title.text == "X motor (um)"


def test_plot_map_does_not_lock_custom_axes_to_equal_scale():
    figure = plot_map(
        _result_set(),
        axes=(Axis([1, 2], "Temperature", "K"), Axis([10, 20], "Load", "N")),
    )

    assert figure.layout.yaxis.scaleanchor is None
    assert figure.layout.yaxis.scaleratio is None


def test_plot_map_renders_scalar_colors_without_crystal_context():
    dataset = replace(_result_set().to_visualization(), crystal=None)

    figure = plot_map(dataset, color="goodness")

    assert _roles(figure) == ["data"]
    np.testing.assert_array_equal(figure.data[0].marker.color, [13, 14])
    assert figure.data[0].marker.showscale is True


def test_prepared_sources_reject_preparation_keywords():
    map_data = prepare_map(_result_set())
    pole_data = prepare_pole_figure(_result_set())
    detector_data = prepare_detector_view(_result_set(), frame_id="a")

    with pytest.raises(TypeError, match="color"):
        plot_map(map_data, color="n_indexed")
    with pytest.raises(TypeError, match="hkl"):
        plot_pole_figure(pole_data, hkl=(1, 0, 0))
    with pytest.raises(TypeError, match="patterns"):
        plot_detector_view(detector_data, patterns="all")


def test_plot_map_3d_uses_opaque_markers_and_validates_updates():
    figure = plot_map(_result_set(), axes=("X", "Y", "Z"), color="cubic_ipf")

    assert figure.data[0].type == "scatter3d"
    assert figure.data[0].marker.opacity is None
    assert figure.layout.scene.aspectmode == "data"
    with pytest.raises(ValueError, match="unknown trace roles"):
        plot_map(_result_set(), trace_update={"boundary": {"visible": False}})
    with pytest.raises(TypeError, match="mapping"):
        plot_map(_result_set(), layout_update=[])


def test_plot_map_empty_scope_returns_annotated_figure():
    figure = plot_map(_result_set(), scope=DataScope(min_indexed=100))

    assert len(figure.data) == 0
    assert "No patterns" in figure.layout.annotations[0].text


def test_plot_pole_figure_has_semantic_roles_and_hover_limit():
    figure = plot_pole_figure(
        _result_set(),
        pole_center=(0.1, -0.2),
        pole_color_radius_deg=15.0,
        hover_point_limit=0,
        trace_update={"boundary": {"line": {"color": "white"}}},
    )

    assert _roles(figure) == ["data", "boundary", "reference", "reference"]
    assert list(figure.data[0].customdata[0]) == ["a", 0, None]
    assert figure.data[0].hoverinfo == "skip"
    assert figure.data[1].line.color == "white"
    assert figure.layout.dragmode == "lasso"
    assert "hover disabled" in figure.layout.annotations[0].text


def test_plot_pole_figure_empty_scope_retains_boundary_and_message():
    figure = plot_pole_figure(_result_set(), scope=DataScope(min_indexed=100))

    assert "data" not in _roles(figure)
    assert "boundary" in _roles(figure)
    assert "No poles" in figure.layout.annotations[0].text


def test_plot_detector_view_preserves_detector_conventions_and_ids():
    data = prepare_detector_view(_result_set(), frame_id="a", image=True)
    figure = plot_detector_view(
        data,
        show_hkl_labels=True,
        image_limits=(0, 100),
        trace_update={"indexed": {"marker": {"size": 20}}},
    )

    assert _roles(figure)[:3] == ["image", "boundary", "detected"]
    detected = figure.data[2]
    assert list(detected.customdata[0]) == ["a", None, 0]
    indexed = next(trace for trace in figure.data if trace.meta["role"] == "indexed")
    assert list(indexed.customdata[0][:3]) == ["a", 0, 0]
    assert indexed.marker.size == 20
    assert figure.layout.xaxis.range == (-0.5, 9.5)
    assert figure.layout.yaxis.range == (7.5, -0.5)
    assert list(figure.data[1].x) == [-0.5, 9.5, 9.5, -0.5, -0.5]
    assert figure.layout.xaxis.scaleanchor == "y"
    assert figure.data[0].zmin == 0
    assert figure.data[0].zmax == 100


def test_plot_detector_view_renders_simulated_reflections_and_updates_by_role():
    data = prepare_detector_view(_result_set(), frame_id="a")
    simulation = DetectorSimulationData(
        pattern_index=0,
        hkl=np.array([[0, 1, -1], [1, 1, -2]]),
        predicted_xy=np.array([[1.5, 2.5], [3.5, 4.5]]),
        energy_kev=np.array([11.25, 14.5]),
        relative_intensity=np.array([0.75, 0.25]),
    )
    prepared = replace(data, simulations=(simulation,))
    figure = plot_detector_view(
        prepared,
        show_hkl_labels=True,
        trace_update={"simulated": {"marker": {"size": 23}}},
    )

    simulated = next(trace for trace in figure.data if trace.meta["role"] == "simulated")
    assert simulated.type == "scattergl"
    assert simulated.marker.symbol == "triangle-up-open"
    assert simulated.marker.color == "rgb(128,0,128)"
    assert simulated.marker.size == 23
    assert list(simulated.customdata[0]) == ["a", 0, None, 0, 1, -1, 11.25, 0.75]
    assert simulated.text[0] == "(0 1 -1)"
    assert "relative intensity" in simulated.hovertemplate

    hidden = plot_detector_view(prepared, show_simulated=False)
    assert "simulated" not in _roles(hidden)


def test_plot_detector_view_raw_simulation_argument_is_prepared_once(monkeypatch):
    import laueanalysis.visualization.rendering as rendering

    prepared = prepare_detector_view(_result_set(), frame_id="a")
    calls = []

    def fake_prepare(source, **kwargs):
        calls.append(kwargs)
        return prepared

    monkeypatch.setattr(rendering, "prepare_detector_view", fake_prepare)
    plot_detector_view(
        object(),
        frame_id="a",
        simulation_energy_range_kev=(8.0, 20.0),
        show_simulated=False,
    )
    assert calls == [{
        "frame_id": "a",
        "patterns": "all",
        "image": None,
        "detector_index": None,
        "simulation_energy_range_kev": (8.0, 20.0),
    }]

    monkeypatch.setattr(
        rendering,
        "prepare_detector_view",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reran")),
    )
    with pytest.raises(TypeError, match="simulation_energy_range_kev"):
        plot_detector_view(
            prepared,
            simulation_energy_range_kev=(9.0, 19.0),
            show_simulated=False,
        )


def test_plot_detector_view_propagates_preparation_errors(monkeypatch):
    import laueanalysis.visualization.rendering as rendering

    monkeypatch.setattr(
        rendering,
        "prepare_detector_view",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulation failed")),
    )
    with pytest.raises(RuntimeError, match="simulation failed"):
        plot_detector_view(
            object(),
            frame_id="a",
            simulation_energy_range_kev=(6.0, 30.0),
        )


def test_plot_detector_view_requires_frame_for_unprepared_source():
    with pytest.raises(TypeError, match="frame_id"):
        plot_detector_view(_result_set())


def test_plot_detector_view_annotates_empty_frame():
    empty = replace(
        _result_set().results[0],
        peaks=np.zeros(0, dtype=PEAK_DTYPE),
        patterns=(),
    )
    figure = plot_detector_view(
        ResultSet((empty,), frame_ids=("empty",), geometry=_result_set().geometry),
        frame_id="empty",
    )

    assert _roles(figure) == ["boundary"]
    assert "no detected peaks" in figure.layout.annotations[0].text


def test_plot_detector_view_distinguishes_empty_simulation_result():
    source = _result_set()
    empty = replace(source.results[0], peaks=np.zeros(0, dtype=PEAK_DTYPE), patterns=())
    data = prepare_detector_view(
        ResultSet((empty,), frame_ids=("empty",), geometry=source.geometry),
        frame_id="empty",
    )
    attempted = replace(data, simulations=(DetectorSimulationData(
        pattern_index=0,
        hkl=np.empty((0, 3), dtype=int),
        predicted_xy=np.empty((0, 2)),
        energy_kev=np.empty(0),
        relative_intensity=np.empty(0),
    ),))

    figure = plot_detector_view(attempted)
    assert _roles(figure) == ["boundary"]
    assert "no missing reflections" in figure.layout.annotations[0].text


def test_selection_from_plotly_deduplicates_stable_identities():
    event = {
        "points": [
            {"customdata": ["a", 0, None]},
            {"customdata": ["a", 0, 2]},
            {"customdata": ["a", 0, 2, 1, 0, -1]},
            {"customdata": ["a", 0, None, 0, 1, -1, 11.0, 0.5]},
            {"customdata": ["a", 0, None, 0, 1, -1, 11.0, 0.5]},
            {"customdata": ["a", 1, None, 1, 1, -2, 12.0, 0.25]},
            {"customdata": ["b", None, 1]},
            {},
        ]
    }

    assert selection_from_plotly(event) == PlotlySelection(
        frame_ids=("a", "b"),
        pattern_ids=(("a", 0), ("a", 1)),
        peak_ids=(("a", 2), ("b", 1)),
        reflection_ids=(("a", 0, 0, 1, -1), ("a", 1, 1, 1, -2)),
    )
    assert selection_from_plotly(None) == PlotlySelection()


def test_selection_from_plotly_normalizes_numpy_integer_identities():
    selection = selection_from_plotly({
        "points": [{
            "customdata": [
                np.int64(5),
                np.int32(2),
                None,
                np.int16(1),
                np.int16(0),
                np.int16(-1),
            ]
        }]
    })
    assert selection == PlotlySelection(
        frame_ids=(5,),
        pattern_ids=((5, 2),),
        reflection_ids=((5, 2, 1, 0, -1),),
    )
    assert all(type(value) is int for value in selection.reflection_ids[0])


def test_selection_from_plotly_rejects_invalid_payloads():
    with pytest.raises(TypeError, match="mapping"):
        selection_from_plotly([])
    with pytest.raises(ValueError, match="frame, pattern, and peak"):
        selection_from_plotly({"points": [{"customdata": ["a"]}]})
    with pytest.raises(ValueError, match="Miller indices"):
        selection_from_plotly({
            "points": [{"customdata": ["a", 0, None, 1.0, 0, -1]}]
        })
    with pytest.raises(ValueError, match="frame IDs"):
        selection_from_plotly({"points": [{"customdata": [np.bool_(True), 0, None]}]})
