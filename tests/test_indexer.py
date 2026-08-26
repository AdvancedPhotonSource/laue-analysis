from dataclasses import replace
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from laueanalysis.indexing import (
    FrameMetadata, Indexer, IndexParams, PeakParams, index_frame, load_crystal,
    load_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "src/laueanalysis/indexing/bin/liblaue.so"
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="liblaue.so is not built")
DATA = ROOT / "sandbox/data/i71"
BASELINE = ROOT / "sandbox/results/i71_baseline_20"


def _table_after(path: Path, marker: str, delimiter=None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    return np.loadtxt(lines[start:], delimiter=delimiter, ndmin=2)


def test_lauego_has_explicit_compatibility_interface():
    from laueanalysis.indexing import index, lauego

    assert lauego is index


def test_geometry_and_crystal_can_be_preloaded():
    geometry = load_geometry(DATA / "geoN_2026-07-07_16-30-21.xml")
    crystal = load_crystal(DATA / "Ni.xml")
    indexer = Indexer(geometry, crystal)

    assert indexer.geometry is geometry
    assert indexer.crystal_model is crystal


def test_public_crystal_is_editable_by_replacement():
    crystal = load_crystal(DATA / "Ni.xml")
    modified = replace(crystal, space_group=229)

    assert crystal.space_group == 225
    assert crystal.crystal_system == "cubic"
    assert modified.space_group == 229
    assert modified is not crystal
    Indexer(DATA / "geoN_2026-07-07_16-30-21.xml", modified)


def test_indexer_process_matches_i71_peak_and_q_reference():
    stem = "Ni_FelixFoil_practice_Probe2_Probe8_fromlefthole_10679"
    with h5py.File(DATA / "frames" / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]

    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            threshold_ratio=4.0,
            peak_shape="Lorentzian",
            max_peaks=200,
        ),
    )
    result = indexer.process(image)
    expected_peaks = _table_after(BASELINE / "peaks" / f"peaks_{stem}.txt", "$peakList")
    expected_q = _table_after(BASELINE / "p2q" / f"p2q_{stem}.txt", "$N_Ghat+Intens", delimiter=",")

    assert result.indexed is False
    assert result.patterns == ()
    assert result.n_peaks == len(expected_peaks) == 23
    np.testing.assert_allclose(result.peaks["fit_x"], expected_peaks[:, 0], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["fit_y"], expected_peaks[:, 1], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["intens"], expected_peaks[:, 2], atol=5e-4, rtol=1e-8)
    np.testing.assert_allclose(result.peaks["integral"], expected_peaks[:, 3], atol=5e-6, rtol=1e-8)
    np.testing.assert_allclose(result.peaks["hwhm_x"], expected_peaks[:, 4], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["hwhm_y"], expected_peaks[:, 5], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["tilt"], expected_peaks[:, 6], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["chisq"], expected_peaks[:, 7], atol=5e-6, rtol=1e-5)
    # The reference q vectors were computed from peak positions rounded to 0.001 px.
    np.testing.assert_allclose(result.peaks["qhat"], expected_q[:, :3], atol=2e-7, rtol=0)


def test_indexer_process_matches_i71_index_reference():
    stem = "Ni_FelixFoil_practice_Probe2_Probe8_fromlefthole_10679"
    with h5py.File(DATA / "frames" / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]
    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        DATA / "Ni.xml",
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )

    result = indexer.process(image)
    expected = [
        (np.array([-101.11369418, 123.62147144, 60.31287179]), 10),
        (np.array([-148.04441816, 93.25320264, 132.68896985]), 5),
        (np.array([-114.96303786, 96.34447861, 84.93955000]), 5),
    ]

    assert result.indexed is True
    assert len(result.patterns) == 3
    assert sum(pattern.n_indexed for pattern in result.patterns) == 20
    for pattern, (euler, count) in zip(result.patterns, expected):
        assert pattern.n_indexed == count
        np.testing.assert_allclose(pattern.euler_deg, euler, atol=5e-5, rtol=0)


@pytest.mark.parametrize(
    "index_file",
    sorted((BASELINE / "index").glob("index_*.txt")),
    ids=lambda path: path.stem.removeprefix("index_"),
)
def test_indexer_matches_all_i71_index_references(index_file):
    stem = index_file.stem.removeprefix("index_")
    with h5py.File(DATA / "frames" / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]
    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        DATA / "Ni.xml",
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )

    result = indexer.process(image)
    text = index_file.read_text()
    expected_count = int(re.search(r"\$NpatternsFound\s+(\d+)", text).group(1))
    assert len(result.patterns) == expected_count

    lines = text.splitlines()
    for index, pattern in enumerate(result.patterns):
        euler_text = re.search(rf"\$EulerAngles{index}\s+\{{([^}}]+)", text).group(1)
        expected_euler = np.asarray([float(value) for value in euler_text.split(",")])
        rotation_text = re.search(rf"\$rotation_matrix{index}\s+(.+)", text).group(1)
        expected_rotation = np.asarray([
            float(value) for value in re.findall(r"[-+]?\d+\.\d+", rotation_text)
        ]).reshape((3, 3)).T
        marker = next(i for i, line in enumerate(lines) if line.startswith(f"$array{index}"))
        count = int(lines[marker].split()[2])
        rows = [re.findall(r"[-+]?\d+(?:\.\d+)?", line) for line in lines[marker + 1:marker + 1 + count]]
        expected_hkl = np.asarray([[int(value) for value in row[4:7]] for row in rows])
        expected_indices = np.asarray([int(row[-1]) for row in rows])

        assert pattern.n_indexed == count
        np.testing.assert_array_equal(pattern.hkl, expected_hkl)
        np.testing.assert_array_equal(pattern.pk_index, expected_indices)
        np.testing.assert_allclose(pattern.euler_deg, expected_euler, atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.rotation, expected_rotation, atol=2e-6, rtol=0)


def test_indexer_matches_peak_positions_for_all_i71_frames():
    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            threshold_ratio=4.0,
            max_peaks=200,
        ),
    )

    for frame_file in sorted((DATA / "frames").glob("*.h5")):
        with h5py.File(frame_file) as source:
            result = indexer.process(source["entry1/data/data"][...])
        peaks_file = BASELINE / "peaks" / f"peaks_{frame_file.stem}.txt"
        lines = peaks_file.read_text().splitlines()
        start = next(index for index, line in enumerate(lines) if "$peakList" in line) + 1
        expected = np.loadtxt(lines[start:], ndmin=2) if lines[start:] else np.empty((0, 8))

        assert result.n_peaks == len(expected), frame_file.name
        if len(expected):
            actual_xy = np.column_stack((result.peaks["fit_x"], result.peaks["fit_y"]))
            np.testing.assert_allclose(actual_xy, expected[:, :2], atol=5e-4, rtol=0)


def test_index_frame_and_image_retention_defaults():
    image = np.zeros((8, 12), dtype=np.uint16)
    result = index_frame(
        image,
        geometry=DATA / "geoN_2026-07-07_16-30-21.xml",
    )
    assert result.image is image
    assert result.n_patterns == 0
    assert result.indexed_peak_indices.size == 0
    np.testing.assert_array_equal(result.unindexed_peak_indices, np.arange(result.n_peaks))

    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    batch = indexer.index_many([image, image])
    assert all(item.image is None for item in batch)


def test_result_is_independent_of_indexer_configuration():
    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    original = result.to_step().detector.peaksXY.boxsize
    first_step = result.to_step()
    first_step.detector.peaksXY.boxsize = 123
    indexer.peak_params = replace(indexer.peak_params, boxsize=99)

    assert result.to_step().detector.peaksXY.boxsize == original


def test_indexer_requires_no_metadata_for_in_memory_frame():
    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    result = indexer.process(np.zeros((8, 12), dtype=np.uint16))
    step = result.to_step()

    assert result.metadata == {}
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8
    assert step.detector.detectorID is None


def test_indexer_accepts_optional_manual_metadata():
    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    image = np.zeros((8, 12), dtype=np.uint16)
    image[2:6, 3:7] = 100
    result = indexer.process(
        image,
        start=(100, 200),
        group=(2, 3),
        metadata=FrameMetadata(
            sample_name="Ni foil",
            scan_number=42,
            detector_id="IGOR",
            exposure_seconds=0.25,
        ),
    )
    step = result.to_step()

    assert step.sampleName == "Ni foil"
    assert step.scanNum == 42
    assert step.detector.detectorID == "IGOR"
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8
    assert step.detector.roi.startx == 100
    assert step.detector.roi.endx == 122
    assert step.detector.roi.starty == 200
    assert step.detector.roi.endy == 221


def test_indexer_accepts_partial_hdf5_metadata(tmp_path):
    path = tmp_path / "minimal.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/sample/name", data=np.asarray([b"partial"]))

    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    result = indexer.process(path)
    step = result.to_step()

    assert result.metadata == {"sample_name": "partial"}
    assert step.sampleName == "partial"
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8


def test_indexer_processes_file_and_builds_step(tmp_path):
    stem = "Ni_FelixFoil_practice_Probe2_Probe8_fromlefthole_10679"
    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        DATA / "Ni.xml",
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.process(DATA / "frames" / f"{stem}.h5")
    step = result.to_step()
    output = tmp_path / "result.xml"
    indexer.write_xml(result, output)

    assert result.input_image.endswith(f"{stem}.h5")
    assert step.sampleName == "HZO"
    assert step.detector.detectorID == "PE1621 723-3335"
    assert step.detector.peaksXY.Npeaks == 23
    assert step.indexing.NpatternsFound == 3
    assert step.indexing.Nindexed == 20
    assert output.read_text().startswith('<?xml version="1.0" ?>')


def test_indexer_selects_detector_by_id():
    indexer = Indexer(
        DATA / "geoN_2026-07-07_16-30-21.xml",
        detector_id="PE1621 723-3335",
    )
    assert indexer.detector_index == 0

    with pytest.raises(ValueError, match="not present"):
        Indexer(DATA / "geoN_2026-07-07_16-30-21.xml", detector_id="missing")


def test_indexer_validates_index_parameters():
    with pytest.raises(ValueError, match="hkl_prefer"):
        Indexer(
            DATA / "geoN_2026-07-07_16-30-21.xml",
            index_params=IndexParams(hkl_prefer=(0, 1)),
        )


def test_indexer_requires_uint16_2d_frame():
    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    with pytest.raises(ValueError, match="uint16"):
        indexer.process(np.zeros((4, 4), dtype=np.float64))


def test_native_invalid_input_maps_to_input_error():
    from laueanalysis.indexing import InputError
    from laueanalysis.indexing.indexer import _raise_native_error

    with pytest.raises(InputError, match="peak search failed"):
        _raise_native_error(1, "peak search", "invalid input")


def test_native_allocation_failure_maps_to_memory_error():
    from laueanalysis.indexing.indexer import _raise_native_error

    with pytest.raises(MemoryError, match="orientation indexing failed"):
        _raise_native_error(2, "orientation indexing", "unable to allocate storage")


def test_indexer_validates_frame_roi_against_detector():
    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    image = np.zeros((8, 12), dtype=np.uint16)

    indexer.process(image, start=(2024, 2032), group=(2, 2))
    with pytest.raises(ValueError, match="exceeds detector bounds 2048x2048"):
        indexer.process(image, start=(2025, 2032), group=(2, 2))
    with pytest.raises(ValueError, match="two nonnegative integers"):
        indexer.process(image, start=(0.5, 0))


def test_indexer_validates_hdf5_detector_id(tmp_path):
    path = tmp_path / "wrong-detector.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/detector/ID", data=np.asarray([b"wrong detector"]))

    indexer = Indexer(DATA / "geoN_2026-07-07_16-30-21.xml")
    with pytest.raises(ValueError, match="does not match selected detector"):
        indexer.process(path)
