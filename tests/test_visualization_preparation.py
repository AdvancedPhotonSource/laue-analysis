from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from laueanalysis.analysis import SurfaceFrame
from laueanalysis.indexing import FrameResult, Pattern, load_crystal, load_geometry
from laueanalysis.indexing.indexer import PEAK_DTYPE
from laueanalysis.visualization import (
    Axis,
    DataScope,
    DetectorViewData,
    MapData,
    PoleFigureData,
    ResultSet,
    ScalarColor,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
)


DATA = Path(__file__).resolve().parents[1] / "sandbox/data/i71"


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
            "crystal": load_crystal(DATA / "Ni.xml"),
            "geometry": load_geometry(DATA / "geoN_2026-07-07_16-30-21.xml"),
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
    prepared = prepare_map(_result_set(), axes=("x", "h"), color="goodness")

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
    named_surface = prepare_map(_result_set(with_context=True), color="ipf", surface="Z")
    assert named_surface.colors.shape == (2, 3)
    with pytest.raises(ValueError, match="one value per frame"):
        prepare_map(dataset, axes=(Axis([1], "Bad"), "y"))


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


def test_prepare_map_ipf_requires_crystal_context():
    with pytest.raises(ValueError, match="crystal context"):
        prepare_map(_result_set(), color="ipf")


def test_prepare_map_ipf_uses_surface_and_crystal_rotation():
    prepared = prepare_map(
        _result_set(with_context=True),
        color="ipf",
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
    map_ipf = prepare_map(result_set, color="ipf", scope=DataScope(min_indexed=100))
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
