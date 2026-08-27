from dataclasses import replace

import numpy as np
import pytest

from laueanalysis.indexing import FrameResult, Pattern
from laueanalysis.indexing.indexer import PEAK_DTYPE
from laueanalysis.visualization import DataScope, ResultSet, VisualizationDataset


def _pattern(count, goodness=10.0):
    return Pattern(
        euler_deg=np.zeros(3),
        rotation=np.eye(3),
        recip=np.eye(3),
        goodness=goodness,
        rms_error_deg=0.1,
        hkl=np.tile([1, 0, 0], (count, 1)),
        pk_index=np.arange(count, dtype=np.int32),
        err_deg=np.arange(count, dtype=float) / 10,
        energy_kev=np.arange(count, dtype=float) + 10,
        pred_intens=np.arange(count, dtype=float) + 100,
    )


def _result(patterns=(), *, position=(1.0, 2.0, 3.0), n_peaks=4):
    peaks = np.zeros(n_peaks, dtype=PEAK_DTYPE)
    peaks["fit_x"] = np.arange(n_peaks)
    return FrameResult(
        peaks=peaks,
        patterns=tuple(patterns),
        threshold_used=100,
        total_sum=1000,
        sum_above_threshold=500,
        num_above_threshold=4,
        peaksearch_seconds=0.1,
        indexing_seconds=0.2,
        metadata={"sample_position": position},
        image_shape=(10, 20),
        image=np.zeros((10, 20), dtype=np.uint16),
    )


def test_result_set_validates_ids_and_preserves_results():
    results = (_result(), _result(position=(4, 5, 6)))
    result_set = ResultSet(results, frame_ids=("a", "b"))
    assert result_set.results == results
    assert result_set.frame_ids == ("a", "b")
    with pytest.raises(ValueError, match="unique"):
        ResultSet(results, frame_ids=("same", "same"))
    with pytest.raises(ValueError, match="contain 2"):
        ResultSet(results, frame_ids=("one",))


def test_visualization_dataset_normalizes_all_record_levels():
    result_set = ResultSet(
        (_result((_pattern(3), _pattern(2))), _result((_pattern(4),), position=(4, 5, 6))),
        frame_ids=(10, 20),
    )
    dataset = result_set.to_visualization()

    assert isinstance(dataset, VisualizationDataset)
    assert dataset.n_frames == 2
    assert dataset.n_patterns == 3
    assert dataset.n_assignments == 9
    assert dataset.pattern_ids(DataScope()) == ((10, 0), (20, 0))
    assert dataset.pattern_ids(DataScope(patterns="all")) == ((10, 0), (20, 0))
    assert dataset.pattern_ids(DataScope(patterns="all", min_indexed=2)) == (
        (10, 0),
        (10, 1),
        (20, 0),
    )
    assert dataset.pattern_ids(DataScope(patterns=(1,), min_indexed=0)) == ((10, 1),)
    assert not dataset.sample_positions.flags.writeable
    assert not dataset.pattern_rotations.flags.writeable
    assert not dataset.images[0].flags.writeable


def test_result_set_rejects_invalid_context():
    with pytest.raises(TypeError, match="crystal"):
        ResultSet((_result(),), crystal="Ni")
    with pytest.raises(TypeError, match="geometry"):
        ResultSet((_result(),), geometry="geo.xml")


def test_visualization_dataset_preserves_missing_positions():
    result = replace(_result(), metadata={})
    dataset = ResultSet((result,)).to_visualization()
    assert np.isnan(dataset.sample_positions).all()


def test_data_scope_validates_values():
    with pytest.raises(ValueError, match="patterns"):
        DataScope(patterns="first")
    with pytest.raises(ValueError, match="min_indexed"):
        DataScope(min_indexed=-1)


def test_invalid_assignment_peak_index_is_rejected():
    result = _result((_pattern(3),), n_peaks=2)
    with pytest.raises(ValueError, match="invalid peak indices"):
        ResultSet((result,)).to_visualization()
