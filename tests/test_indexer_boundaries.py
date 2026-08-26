from pathlib import Path

import h5py
import numpy as np
import pytest

from laueanalysis.indexing import (
    Indexer,
    IndexingError,
    IndexParams,
    InputError,
    PeakParams,
)
from laueanalysis.indexing import indexer as indexer_module
from laueanalysis.indexing._liblaue import get_library


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "src/laueanalysis/indexing/bin/liblaue.so"
GEOMETRY = ROOT / "sandbox/data/i71/geoN_2026-07-07_16-30-21.xml"
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="liblaue.so is not built")


@pytest.mark.parametrize(
    "peak_params, message",
    [
        (PeakParams(boxsize=0), "positive"),
        (PeakParams(min_size=0), "positive"),
        (PeakParams(min_separation=0), "positive"),
        (PeakParams(max_peaks=0), "positive"),
        (PeakParams(max_rfactor=0), "max_rfactor"),
        (PeakParams(peak_shape="Voigt"), "peak_shape"),
        (PeakParams(detect_binning=2), "detect_binning"),
    ],
)
def test_indexer_rejects_invalid_peak_parameters(peak_params, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY, peak_params=peak_params)


@pytest.mark.parametrize(
    "index_params, message",
    [
        (IndexParams(kev_max_calc=0), "positive"),
        (IndexParams(kev_max_test=0), "positive"),
        (IndexParams(angle_tolerance_deg=0), "positive"),
        (IndexParams(cone_deg=0), "positive"),
        (IndexParams(hkl_prefer=(0, 1)), "exactly three"),
        (IndexParams(max_data=1), "at least 2"),
    ],
)
def test_indexer_rejects_invalid_index_parameters(index_params, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY, index_params=index_params)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 3, 4), dtype=np.uint16),
        np.zeros((3, 4), dtype=np.float32),
    ],
)
def test_process_rejects_invalid_frame_arrays(frame):
    with pytest.raises(InputError, match="2D uint16"):
        Indexer(GEOMETRY).process(frame)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start": (-1, 0)}, "nonnegative"),
        ({"start": (0.5, 0)}, "nonnegative"),
        ({"group": (0, 1)}, "positive"),
        ({"group": (1, 1.5)}, "positive"),
        ({"depth": np.inf}, "finite"),
        ({"mask": np.zeros((3, 3))}, "mask shape"),
    ],
)
def test_process_rejects_invalid_frame_options(kwargs, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY).process(np.zeros((4, 5), dtype=np.uint16), **kwargs)


def test_hdf5_metadata_override_and_processing_values(tmp_path):
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 5), dtype=np.uint16))
        output.create_dataset("entry1/sample/name", data=np.asarray([b"from file"]))
        output.create_dataset("entry1/detector/startx", data=10)
        output.create_dataset("entry1/detector/starty", data=20)
        output.create_dataset("entry1/detector/binx", data=2)
        output.create_dataset("entry1/detector/biny", data=3)

    result = Indexer(GEOMETRY).process(
        path, start=(100, 100), group=(1, 1), metadata={"sample_name": "supplied"}
    )

    assert result.metadata["sample_name"] == "supplied"
    assert result.start == (10, 20)
    assert result.group == (2, 3)
    assert result.to_step().sampleName == "supplied"


def test_hdf5_requires_image_dataset(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(KeyError):
        Indexer(GEOMETRY).process(path)


def test_batch_order_retention_and_compatibility_alias():
    frames = [np.zeros((3, width), dtype=np.uint16) for width in (4, 5)]
    indexer = Indexer(GEOMETRY)

    retained = indexer.index_many(frames, keep_images=True)
    discarded = indexer.process_many(frames)

    assert [result.image_shape for result in retained] == [(3, 4), (3, 5)]
    assert all(result.image is frame for result, frame in zip(retained, frames))
    assert all(result.image is None for result in discarded)


class _FailingLibrary:
    def __init__(self, stage, status):
        self.real = get_library()
        self.stage = stage
        self.status = status
        self.free_calls = 0

    def __getattr__(self, name):
        return getattr(self.real, name)

    def laue_find_peaks(self, pixels, nx, ny, params, result):
        if self.stage == "peak search":
            result.status = self.status
            result.message = b"injected failure"
            return self.status
        return self.real.laue_find_peaks(pixels, nx, ny, params, result)

    def laue_pixels_to_q(self, geometry, detector_index, result):
        if self.stage == "pixel-to-q conversion":
            result.status = self.status
            result.message = b"injected failure"
            return self.status
        return self.real.laue_pixels_to_q(geometry, detector_index, result)

    def laue_frame_result_free(self, result):
        self.free_calls += 1
        self.real.laue_frame_result_free(result)


@pytest.mark.parametrize(
    ("status", "error"),
    [(1, InputError), (2, MemoryError), (3, IndexingError), (4, IndexingError)],
)
@pytest.mark.parametrize("stage", ["peak search", "pixel-to-q conversion"])
def test_native_failures_map_to_exceptions_and_release_results(
    monkeypatch, stage, status, error
):
    indexer = Indexer(GEOMETRY)
    library = _FailingLibrary(stage, status)
    monkeypatch.setattr(indexer_module, "get_library", lambda: library)

    with pytest.raises(error, match=f"{stage} failed: injected failure"):
        indexer.process(np.zeros((4, 5), dtype=np.uint16))

    assert library.free_calls == 1
