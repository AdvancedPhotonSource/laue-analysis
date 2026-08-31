from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


from laueanalysis.analysis import (
    SimulationResult,
    SurfaceFrame,
    orientation_to_rodrigues,
    rodrigues_colors,
)
from laueanalysis.indexing import Cell, Crystal, FrameResult, Pattern, load_crystal, load_geometry
from laueanalysis.indexing.indexer import PEAK_DTYPE
from laueanalysis.visualization import (
    Axis,
    DataScope,
    DetectorSimulationData,
    AXIS_OPTIONS,
    COLOR_MODES,
    PALETTE_OPTIONS,
    POLE_COLOR_MODES,
    DetectorViewData,
    MapData,
    PoleFigureData,
    ResultSet,
    ScalarColor,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_FILE = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL_FILE = ROOT / "tests/config/Ni.xml"


def _pattern(count, rotation=None):
    return Pattern(
        euler_deg=np.zeros(3),
        rotation=np.eye(3) if rotation is None else rotation,
        recip=np.eye(3),
        goodness=10 + count,
        rms_error_deg=0.1,
        hkl=np.tile([1, 0, -1], (count, 1)),
        pk_index=np.arange(count, dtype=np.int32),
        err_deg=np.arange(count, dtype=float) / 10,
        energy_kev=np.arange(count, dtype=float) + 10,
        pred_intens=np.arange(count, dtype=float) + 100,
    )


def _result(patterns, position, detector_id="PE1621 723-3335"):
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
        metadata={"sample_position": position, "detector_id": detector_id},
        image_shape=(8, 10),
        image=np.arange(80, dtype=np.uint16).reshape(8, 10),
    )


def _result_set(with_context=False):
    context = {}
    if with_context:
        context = {
            "crystal": load_crystal(CRYSTAL_FILE),
            "geometry": load_geometry(GEOMETRY_FILE),
        }
    return ResultSet(
        (
            _result((_pattern(3), _pattern(2)), (0, 1, 2)),
            _result((_pattern(4),), (3, 4, 5)),
        ),
        frame_ids=("a", "b"),
        **context,
    )


def test_prepare_map_builtin_axes_scope_and_scalar_color():
    prepared = prepare_map(_result_set(), axes=("X", "H"), color="goodness")

    assert isinstance(prepared, MapData)
    assert prepared.frame_ids == ("a", "b")
    assert prepared.pattern_indices.tolist() == [0, 0]
    np.testing.assert_allclose(prepared.coordinates[:, 0], [0, 3])
    np.testing.assert_allclose(prepared.coordinates[:, 1], [3 / np.sqrt(2), 9 / np.sqrt(2)])
    np.testing.assert_allclose(prepared.colors, [13, 14])
    assert not prepared.coordinates.flags.writeable
    assert not prepared.colors.flags.writeable


def test_prepare_map_supports_custom_axis_and_color_arrays_or_callables():
    dataset = _result_set().to_visualization()
    prepared = prepare_map(
        dataset,
        axes=(
            Axis(np.array([10, 20]), "Load", "N", alignment="frame"),
            Axis(lambda data: np.arange(data.n_patterns), "Pattern order", alignment="pattern"),
        ),
        color=ScalarColor(lambda data: data.pattern_goodness * 2, label="Twice goodness"),
        scope=DataScope(patterns="all", min_indexed=0),
    )

    assert prepared.axis_labels == ("Load (N)", "Pattern order")
    np.testing.assert_array_equal(prepared.coordinates[:, 0], [10, 10, 20])
    np.testing.assert_array_equal(prepared.coordinates[:, 1], [0, 1, 2])
    np.testing.assert_array_equal(prepared.colors, [26, 24, 28])
    direct = prepare_map(dataset, color=np.array([1.0, 2.0, 3.0]), scope=DataScope(patterns="all", min_indexed=0))
    np.testing.assert_array_equal(direct.colors, [1, 2, 3])
    named_surface = prepare_map(_result_set(with_context=True), color="cubic_ipf", surface="Z")
    assert named_surface.colors.shape == (2, 3)
    with pytest.raises(ValueError, match="one value per frame"):
        prepare_map(dataset, axes=(Axis([1], "Bad"), "Y"))


def test_prepare_map_empty_scope_has_stable_shapes():
    prepared = prepare_map(_result_set(), scope=DataScope(min_indexed=100))
    assert prepared.coordinates.shape == (0, 2)
    assert prepared.colors.shape == (0,)
    assert prepared.frame_ids == ()


def test_prepare_map_requires_real_coordinates():
    result_set = _result_set()
    missing = replace(result_set.results[0], metadata={})
    with pytest.raises(ValueError, match="coordinates"):
        prepare_map(ResultSet((missing,), frame_ids=("missing",)))


def test_prepare_map_rodrigues_uses_unreduced_unsupported_symmetry():
    angle = np.radians(30)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    result_set = ResultSet(
        (_result((_pattern(3, rotation),), (0, 1, 2)),),
        crystal=Crystal("tetragonal", 139, Cell(0.4, 0.4, 0.6)),
    )

    prepared = prepare_map(result_set, color="rodrigues")
    expected = rodrigues_colors([orientation_to_rodrigues(rotation)])
    np.testing.assert_allclose(prepared.colors, expected)


def test_advertised_axes_and_color_modes_are_accepted():
    source = _result_set(with_context=True)
    dataset = replace(source.to_visualization(), depths=np.array([1.0, 2.0]))
    scope = DataScope(patterns="all", min_indexed=0)
    axis_names = tuple(choice.value for choice in AXIS_OPTIONS)
    for axis in axis_names:
        assert prepare_map(dataset, axes=(axis, "Y"), scope=scope).coordinates.shape == (3, 2)

    for choice in COLOR_MODES:
        kwargs = {}
        if choice.value == "misorientation":
            kwargs["misorientation_reference"] = ("a", 0)
        prepared = prepare_map(dataset, color=choice.value, scope=scope, **kwargs)
        expected_shape = (3, 3) if prepared.color_kind == "rgb" else (3,)
        assert prepared.colors.shape == expected_shape

    for choice in POLE_COLOR_MODES:
        prepared = prepare_pole_figure(dataset, color=choice.value, scope=scope)
        assert len(prepared.points) > 0

    for choice in PALETTE_OPTIONS:
        prepared = prepare_map(
            dataset,
            color=ScalarColor("goodness", palette=choice.value),
            scope=scope,
        )
        assert prepared.palette == choice.value


def test_prepare_map_misorientation_uses_stable_pattern_identity():
    source = _result_set(with_context=True)
    prepared = prepare_map(
        source,
        color="misorientation",
        scope=DataScope(patterns="all", min_indexed=0),
        misorientation_reference=("a", 0),
    )
    np.testing.assert_allclose(prepared.colors[0], 0)
    with pytest.raises(ValueError, match="misorientation_reference"):
        prepare_map(source, color="misorientation")


def test_prepare_map_ipf_requires_crystal_context():
    with pytest.raises(ValueError, match="crystal context"):
        prepare_map(_result_set(), color="cubic_ipf")


def test_prepare_map_ipf_uses_surface_and_crystal_rotation():
    prepared = prepare_map(
        _result_set(with_context=True),
        color="cubic_ipf",
        surface=SurfaceFrame.aps_34ide("Z"),
    )
    assert prepared.color_kind == "rgb"
    assert prepared.colors.shape == (2, 3)
    assert np.all((prepared.colors >= 0) & (prepared.colors <= 1))


def test_prepare_pole_figure_requires_crystal_context():
    with pytest.raises(ValueError, match="crystal context"):
        prepare_pole_figure(_result_set())


def test_prepare_pole_figure_preserves_pattern_identity_and_is_read_only():
    prepared = prepare_pole_figure(
        _result_set(with_context=True),
        hkl=(1, 0, 0),
        scope=DataScope(patterns="all", min_indexed=0),
    )

    assert isinstance(prepared, PoleFigureData)
    assert len(prepared.points) > 0
    assert set(zip(prepared.frame_ids, prepared.pattern_indices, strict=True)) == {
        ("a", 0),
        ("a", 1),
        ("b", 0),
    }
    assert not prepared.points.flags.writeable
    assert not prepared.pattern_indices.flags.writeable


def test_prepare_pole_figure_empty_scope():
    result_set = _result_set(with_context=True)
    prepared = prepare_pole_figure(result_set, scope=DataScope(min_indexed=100))
    map_ipf = prepare_map(result_set, color="cubic_ipf", scope=DataScope(min_indexed=100))
    map_rodrigues = prepare_map(result_set, color="rodrigues", scope=DataScope(min_indexed=100))
    assert prepared.points.shape == (0, 2)
    assert prepared.colors.shape == (0, 3)
    assert map_ipf.colors.shape == (0, 3)
    assert map_rodrigues.colors.shape == (0, 3)


def test_prepare_detector_view_requires_geometry_and_preserves_layers():
    without_geometry = _result_set()
    with pytest.raises(ValueError, match="geometry"):
        prepare_detector_view(without_geometry, frame_id="a")

    prepared = prepare_detector_view(_result_set(with_context=True), frame_id="a", image=True)
    assert isinstance(prepared, DetectorViewData)
    assert prepared.frame_id == "a"
    assert prepared.measured_xy.shape == (4, 2)
    assert prepared.measured_indexed.tolist() == [True, True, True, False]
    assert len(prepared.patterns) == 2
    assert prepared.patterns[0].predicted_xy.shape == (3, 2)
    assert prepared.image.shape == (8, 10)
    assert prepared.measured_peak_indices.tolist() == [0, 1, 2, 3]
    assert not prepared.measured_xy.flags.writeable
    assert not prepared.measured_peak_indices.flags.writeable
    assert not prepared.image.flags.writeable


def test_prepare_detector_view_validates_frame_and_image():
    result_set = _result_set(with_context=True)
    with pytest.raises(KeyError, match="unknown frame_id"):
        prepare_detector_view(result_set, frame_id="missing")
    with pytest.raises(ValueError, match="two-dimensional"):
        prepare_detector_view(result_set, frame_id="a", image=np.zeros(3))
    with pytest.raises(ValueError, match="patterns"):
        prepare_detector_view(result_set, frame_id="a", patterns="first")


def test_detector_simulation_data_normalizes_validates_and_owns_arrays():
    hkl = np.array([[1, 2, 3]], dtype=np.int32)
    prepared = DetectorSimulationData(
        pattern_index=np.int64(2),
        hkl=hkl,
        predicted_xy=[[4, 5]],
        energy_kev=[12.5],
        relative_intensity=[0.75],
    )
    hkl[0, 0] = 99

    assert prepared.pattern_index == 2
    assert prepared.hkl.tolist() == [[1, 2, 3]]
    assert prepared.hkl.dtype.kind == "i"
    assert prepared.predicted_xy.dtype == np.float64
    assert all(
        not value.flags.writeable
        for value in (
            prepared.hkl,
            prepared.predicted_xy,
            prepared.energy_kev,
            prepared.relative_intensity,
        )
    )
    with pytest.raises(ValueError, match="shape"):
        replace(prepared, predicted_xy=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        replace(prepared, energy_kev=[np.nan])
    with pytest.raises(TypeError, match="integer dtype"):
        replace(prepared, hkl=np.array([[1.0, 2.0, 3.0]]))


def test_prepare_detector_view_simulates_selected_patterns_in_roi_coordinates(monkeypatch):
    import laueanalysis.visualization.preparation as preparation

    context = _result_set(with_context=True)
    frame = replace(
        context.results[0], start=(100, 200), group=(2, 4), depth=12.5
    )
    source = ResultSet(
        (frame,),
        frame_ids=("a",),
        crystal=context.crystal,
        geometry=context.geometry,
    )
    simulated = SimulationResult(
        hkl=np.array([[2, 0, -2], [0, 1, -1], [0, 2, -1]]),
        q=np.zeros((3, 3)),
        detector_xy=np.array([
            [100.5, 201.5],
            [118.5, 229.5],
            [120.5, 201.5],
        ]),
        energy_kev=np.array([10.0, 11.0, 12.0]),
        relative_intensity=np.array([1.0, 0.5, 0.25]),
    )
    calls = []

    def fake_simulate(crystal, reciprocal, detector, **kwargs):
        calls.append((crystal, reciprocal.copy(), detector, kwargs))
        return simulated

    monkeypatch.setattr(preparation, "simulate_reflections", fake_simulate)
    prepared = prepare_detector_view(
        source,
        frame_id="a",
        simulation_energy_range_kev=(7.0, 25.0),
    )

    assert len(calls) == 2
    assert all(call[0] is context.crystal for call in calls)
    assert all(call[3] == {"energy_range_kev": (7.0, 25.0), "depth": 12.5} for call in calls)
    assert [value.pattern_index for value in prepared.simulations] == [0, 1]
    for value in prepared.simulations:
        assert value.hkl.tolist() == [[0, 1, -1]]
        np.testing.assert_allclose(value.predicted_xy, [[9.0, 7.0]])
        np.testing.assert_allclose(value.energy_kev, [11.0])
        np.testing.assert_allclose(value.relative_intensity, [0.5])


def test_prepare_detector_view_loads_hdf5_image_with_shared_reader(
    monkeypatch, tmp_path
):
    import laueanalysis.visualization.preparation as preparation

    path = tmp_path / "frame.h5"
    expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
    calls = []

    def fake_read_h5_frame(received):
        calls.append(received)
        return expected, {"sample_name": "ignored"}, {"start": (10, 20)}

    monkeypatch.setattr(preparation, "read_h5_frame", fake_read_h5_frame)
    prepared = prepare_detector_view(
        _result_set(with_context=True), frame_id="a", image=path
    )

    assert calls == [path]
    np.testing.assert_array_equal(prepared.image, expected)
    assert not prepared.image.flags.writeable


def test_prepare_detector_view_simulation_is_opt_in_and_errors_propagate(monkeypatch):
    import laueanalysis.visualization.preparation as preparation

    source = _result_set(with_context=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("simulation should not run")

    monkeypatch.setattr(preparation, "simulate_reflections", unexpected)
    assert prepare_detector_view(source, frame_id="a").simulations == ()

    missing_crystal = ResultSet(
        source.results,
        frame_ids=source.frame_ids,
        geometry=source.geometry,
    )
    assert prepare_detector_view(missing_crystal, frame_id="a").simulations == ()
    with pytest.raises(ValueError, match="crystal context"):
        prepare_detector_view(
            missing_crystal,
            frame_id="a",
            simulation_energy_range_kev=(6.0, 30.0),
        )

    monkeypatch.setattr(
        preparation,
        "simulate_reflections",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )
    with pytest.raises(RuntimeError, match="backend failed"):
        prepare_detector_view(
            source,
            frame_id="a",
            simulation_energy_range_kev=(6.0, 30.0),
        )


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [("best", [0]), ("all", [0, 1]), ((1,), [1])],
)
def test_prepare_detector_view_simulates_only_selected_patterns(
    monkeypatch, patterns, expected
):
    import laueanalysis.visualization.preparation as preparation

    empty = SimulationResult(
        hkl=np.empty((0, 3), dtype=int),
        q=np.empty((0, 3)),
        detector_xy=np.empty((0, 2)),
        energy_kev=np.empty(0),
        relative_intensity=np.empty(0),
    )
    depths = []

    def fake_simulate(*args, **kwargs):
        depths.append(kwargs["depth"])
        return empty

    monkeypatch.setattr(preparation, "simulate_reflections", fake_simulate)
    prepared = prepare_detector_view(
        _result_set(with_context=True),
        frame_id="a",
        patterns=patterns,
        simulation_energy_range_kev=(6.0, 30.0),
    )

    assert [value.pattern_index for value in prepared.simulations] == expected
    assert depths == [0.0] * len(expected)
