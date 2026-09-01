import numpy as np
import pandas as pd

from lauelab.indexing import FrameResult, Pattern
from lauelab.indexing.indexer import PEAK_DTYPE
from lauelab.visualization import (
    AXIS_OPTIONS,
    COLOR_MODES,
    PALETTE_OPTIONS,
    POLE_COLOR_MODES,
    SURFACE_PRESETS,

    DataScope,
    ResultSet,
    assignment_table,
    indexed_peak_table,
    pattern_table,
    peak_table,
)


def _dataset():
    peaks = np.zeros(4, dtype=PEAK_DTYPE)
    peaks["fit_x"] = [10, 20, 30, 40]
    peaks["fit_y"] = [11, 21, 31, 41]
    peaks["intens"] = [100, 200, 300, 400]
    pattern = Pattern(
        euler_deg=np.zeros(3),
        rotation=np.eye(3),
        reciprocal=np.eye(3),
        goodness=12,
        rms_error_deg=0.2,
        hkl=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        pk_index=np.array([0, 2, 3]),
        err_deg=np.array([0.1, 0.2, 0.3]),
        energy_kev=np.array([10, 11, 12]),
        pred_intens=np.array([20, 30, 40]),
    )
    result = FrameResult(
        peaks=peaks,
        patterns=(pattern,),
        threshold_used=10,
        total_sum=100,
        sum_above_threshold=50,
        num_above_threshold=4,
        peaksearch_seconds=0.1,
        indexing_seconds=0.2,
        metadata={"sample_position": (1, 2, 3), "scan_number": 42, "energy_kev": 18},
        image_shape=(8, 8),
    )
    return ResultSet((result,), frame_ids=("frame-a",)).to_visualization()


def test_normalized_tables_have_expected_cardinality_and_ids():
    dataset = _dataset()
    peaks = peak_table(dataset)
    patterns = pattern_table(dataset)
    assignments = assignment_table(dataset)
    indexed = indexed_peak_table(dataset)

    assert len(peaks) == 4
    assert len(patterns) == 1
    assert len(assignments) == len(indexed) == 3
    assert assignments["frame_id"].tolist() == ["frame-a"] * 3
    assert assignments["peak_index"].tolist() == [0, 2, 3]
    assert indexed["fit_x"].tolist() == [10, 30, 40]
    assert indexed["goodness"].tolist() == [12, 12, 12]


def test_peak_table_all_frames_includes_unindexed_frame_peaks():
    indexed = _dataset()
    empty_pattern_result = FrameResult(
        peaks=np.zeros(2, dtype=PEAK_DTYPE),
        patterns=(),
        threshold_used=10,
        total_sum=20,
        sum_above_threshold=10,
        num_above_threshold=2,
        peaksearch_seconds=0.1,
        indexing_seconds=0.0,
        metadata={"sample_position": (4, 5, 6)},
        image_shape=(8, 8),
    )
    dataset = ResultSet(
        (
            FrameResult(
                peaks=indexed.peaks,
                patterns=(),
                threshold_used=10,
                total_sum=100,
                sum_above_threshold=50,
                num_above_threshold=4,
                peaksearch_seconds=0.1,
                indexing_seconds=0.2,
                metadata={"sample_position": (1, 2, 3)},
                image_shape=(8, 8),
            ),
            empty_pattern_result,
        ),
        frame_ids=("frame-a", "frame-b"),
    ).to_visualization()

    table = peak_table(dataset, scope=DataScope(patterns="all_frames"))

    assert len(table) == 6
    assert table["frame_id"].tolist() == ["frame-a"] * 4 + ["frame-b"] * 2


def test_tables_convert_to_independent_dataframes():
    table = indexed_peak_table(_dataset())
    dataframe = table.to_dataframe()

    assert isinstance(dataframe, pd.DataFrame)
    assert "<table" in table._repr_html_()
    assert list(dataframe.query("energy_kev >= 11")["peak_index"]) == [2, 3]
    dataframe.loc[0, "fit_x"] = -1
    assert table["fit_x"][0] == 10
    assert not table["fit_x"].flags.writeable


def test_empty_scoped_tables_keep_schema():
    dataset = _dataset()
    scope = DataScope(min_indexed=10)

    for factory in (peak_table, pattern_table, assignment_table, indexed_peak_table):
        table = factory(dataset, scope=scope)
        assert len(table) == 0
        assert len(table.columns) > 0


def test_zero_frame_result_set_peak_table_keeps_peak_schema():
    table = peak_table(ResultSet((), frame_ids=()))

    assert len(table) == 0
    assert {"fit_x", "fit_y", "background", "qhat_x", "qhat_y", "qhat_z"} <= set(table.columns)


def test_choice_descriptors_match_portal_values_and_are_unique():
    assert tuple(choice.value for choice in AXIS_OPTIONS) == (
        "X", "Y", "Z", "H", "F", "depth", "Xlab", "Ylab", "Zlab", "Hlab", "Flab"
    )
    assert tuple(choice.value for choice in COLOR_MODES) == (
        "cubic_ipf", "rodrigues", "misorientation", "pole_hsv",
        "n_indexed", "goodness", "rms_error", "n_patterns",
    )
    assert tuple(choice.value for choice in POLE_COLOR_MODES) == (
        "hsv_position", "ipf", "uniform"
    )
    assert tuple(choice.value for choice in PALETTE_OPTIONS) == (
        "Viridis", "Plasma", "Inferno", "Magma", "Jet", "Rainbow", "Earth"
    )
    for choices in (
        AXIS_OPTIONS,
        COLOR_MODES,
        PALETTE_OPTIONS,
        POLE_COLOR_MODES,
        SURFACE_PRESETS,
    ):
        values = [choice.value for choice in choices]
        assert len(values) == len(set(values))
        assert all(choice.label for choice in choices)
