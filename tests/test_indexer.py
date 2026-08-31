from dataclasses import replace
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from laueanalysis.analysis import simulate_reflections
from laueanalysis.indexing import (
    FrameMetadata, Indexer, IndexParams, PeakParams, index_frame, load_crystal,
    load_geometry,
)
from laueanalysis.visualization import (
    DataScope, ResultSet, load_visualization_xml, prepare_detector_view,
    prepare_pole_figure,
)


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"
BASELINE = ROOT / "tests/data/synthetic/baseline"


def _table_after(path: Path, marker: str, delimiter=None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    return np.loadtxt(lines[start:], delimiter=delimiter, ndmin=2)


def test_lauego_has_explicit_compatibility_interface():
    from laueanalysis.indexing import index, lauego

    assert lauego is index


def test_geometry_and_crystal_can_be_preloaded():
    geometry = load_geometry(GEOMETRY)
    crystal = load_crystal(CRYSTAL)
    indexer = Indexer(geometry, crystal)

    assert indexer.geometry is geometry
    assert indexer.crystal_model is crystal


def test_public_crystal_is_editable_by_replacement():
    crystal = load_crystal(CRYSTAL)
    modified = replace(crystal, space_group=229)

    assert crystal.space_group == 225
    assert crystal.crystal_system == "cubic"
    assert modified.space_group == 229
    assert modified is not crystal
    Indexer(GEOMETRY, modified)


def test_indexer_process_matches_lauego_peak_and_q_reference():
    stem = "synthetic_ni_two_grains"
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]

    indexer = Indexer(
        GEOMETRY,
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
    assert result.n_peaks == len(expected_peaks) == 41
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


def test_indexer_process_matches_lauego_index_reference():
    stem = "synthetic_ni_two_grains"
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
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
        (np.array([-169.14354065, 143.65364921, 137.25271596]), 23),
        (np.array([-110.30109407, 143.68406785, 76.90290940]), 17),
    ]

    assert result.indexed is True
    assert len(result.patterns) == 2
    assert sum(pattern.n_indexed for pattern in result.patterns) == 40
    for pattern, (euler, count) in zip(result.patterns, expected):
        assert pattern.n_indexed == count
        # The LaueGo reference refines orientation from peak positions rounded
        # to 0.001 px; the in-process path uses unrounded positions.
        np.testing.assert_allclose(pattern.euler_deg, euler, atol=5e-4, rtol=0)


@pytest.mark.parametrize(
    "index_file",
    sorted((BASELINE / "index").glob("index_*.txt")),
    ids=lambda path: path.stem.removeprefix("index_"),
)
def test_indexer_matches_all_lauego_index_references(index_file):
    stem = index_file.stem.removeprefix("index_")
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
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
        expected_energy = np.asarray([float(row[8]) for row in rows])
        expected_indices = np.asarray([int(row[-1]) for row in rows])

        assert pattern.n_indexed == count
        np.testing.assert_array_equal(pattern.hkl, expected_hkl)
        np.testing.assert_array_equal(pattern.pk_index, expected_indices)
        np.testing.assert_allclose(pattern.energy_kev, expected_energy, atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.euler_deg, expected_euler, atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.rotation, expected_rotation, atol=2e-6, rtol=0)


def test_reported_hkl_and_energy_describe_the_same_reflection():
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.process(FRAMES / "synthetic_ni_two_grains.h5")

    reported_hkls = {
        tuple(hkl) for pattern in result.patterns for hkl in pattern.hkl
    }
    # In FCC Ni, (0, 2, 2) is the lowest allowed harmonic along (0, 1, 1).
    assert (0, 2, 2) in reported_hkls

    for pattern in result.patterns:
        reciprocal_per_angstrom = pattern.hkl @ pattern.recip / 10.0
        sin_theta = -result.peaks["qhat"][pattern.pk_index, 2]
        expected_energy = (
            12.3984187
            * np.linalg.norm(reciprocal_per_angstrom, axis=1)
            / (4.0 * np.pi * sin_theta)
        )
        np.testing.assert_allclose(
            pattern.energy_kev, expected_energy, atol=0, rtol=1e-5
        )


def test_indexer_matches_peak_positions_for_all_synthetic_frames():
    indexer = Indexer(
        GEOMETRY,
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

    for frame_file in sorted((FRAMES).glob("*.h5")):
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
        geometry=GEOMETRY,
    )
    assert result.image is image
    assert result.n_patterns == 0
    assert result.indexed_peak_indices.size == 0
    np.testing.assert_array_equal(result.unindexed_peak_indices, np.arange(result.n_peaks))

    indexer = Indexer(GEOMETRY)
    batch = indexer.index_many([image, image])
    assert all(item.image is None for item in batch)


def test_result_is_independent_of_indexer_configuration():
    indexer = Indexer(GEOMETRY)
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    original = result.to_step().detector.peaksXY.boxsize
    first_step = result.to_step()
    first_step.detector.peaksXY.boxsize = 123
    indexer.peak_params = replace(indexer.peak_params, boxsize=99)

    assert result.to_step().detector.peaksXY.boxsize == original


def test_indexer_requires_no_metadata_for_in_memory_frame():
    indexer = Indexer(GEOMETRY)
    result = indexer.process(np.zeros((8, 12), dtype=np.uint16))
    step = result.to_step()

    assert result.metadata == {}
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8
    assert step.detector.detectorID is None


def test_indexer_accepts_optional_manual_metadata():
    indexer = Indexer(GEOMETRY)
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
    assert step.detector.roi.endx == 123
    assert step.detector.roi.starty == 200
    assert step.detector.roi.endy == 223


def test_indexer_accepts_partial_hdf5_metadata(tmp_path):
    path = tmp_path / "minimal.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/sample/name", data=np.asarray([b"partial"]))

    indexer = Indexer(GEOMETRY)
    result = indexer.process(path)
    step = result.to_step()

    assert result.metadata == {"sample_name": "partial"}
    assert step.sampleName == "partial"
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8


def test_indexer_processes_file_and_builds_step(tmp_path):
    stem = "synthetic_ni_two_grains"
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.process(FRAMES / f"{stem}.h5")
    step = result.to_step()
    output = tmp_path / "result.xml"
    indexer.write_xml(result, output)

    assert result.input_image.endswith(f"{stem}.h5")
    assert step.sampleName == "synthetic Ni"
    assert step.detector.detectorID == "PE1621 723-3335"
    assert step.detector.peaksXY.Npeaks == 41
    assert step.indexing.NpatternsFound == 2
    assert step.indexing.Nindexed == 40
    assert output.read_text().startswith('<?xml version="1.0" ?>')


def test_reciprocal_convention_is_consistent_across_live_xml_and_visualization(
    tmp_path,
):
    stem = "synthetic_ni_two_grains"
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.process(FRAMES / f"{stem}.h5")
    live = ResultSet.from_indexer(indexer, (result,))
    live_data = live.to_visualization()

    output = tmp_path / "result.xml"
    indexer.write_xml(result, output)
    xml_data = load_visualization_xml(output, geometry=indexer.geometry)

    np.testing.assert_allclose(
        live_data.pattern_reciprocals,
        xml_data.pattern_reciprocals,
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        live_data.pattern_rotations,
        xml_data.pattern_rotations,
        atol=1e-12,
        rtol=0,
    )
    for pattern in result.patterns:
        calculated = pattern.hkl @ pattern.recip
        calculated /= np.linalg.norm(calculated, axis=1, keepdims=True)
        measured = result.peaks["qhat"][pattern.pk_index]
        cosine = np.clip(np.sum(calculated * measured, axis=1), -1, 1)
        angles = np.degrees(np.arccos(cosine))
        assert np.max(angles) < 5e-4

    pattern = result.patterns[0]
    simulated = simulate_reflections(
        indexer.crystal_model,
        pattern.recip,
        indexer.detector,
        energy_range_kev=(6.0, 35.0),
        depth=0.0 if result.depth is None else result.depth,
    )
    np.testing.assert_allclose(simulated.q, simulated.hkl @ pattern.recip)
    indexed_divisors = np.gcd.reduce(np.abs(pattern.hkl), axis=1)
    simulated_divisors = np.gcd.reduce(np.abs(simulated.hkl), axis=1)
    indexed_directions = {
        tuple(row) for row in pattern.hkl // indexed_divisors[:, None]
    }
    simulated_directions = {
        tuple(row) for row in simulated.hkl // simulated_divisors[:, None]
    }
    assert indexed_directions <= simulated_directions

    live_view = prepare_detector_view(live, frame_id=0, patterns="all")
    xml_view = prepare_detector_view(xml_data, frame_id=0, patterns="all")
    for live_pattern, xml_pattern in zip(
        live_view.patterns, xml_view.patterns, strict=True
    ):
        np.testing.assert_allclose(
            live_pattern.predicted_xy, xml_pattern.predicted_xy, atol=1e-12, rtol=0
        )
        measured = live_view.measured_xy[live_pattern.measured_peak_indices]
        error = np.linalg.norm(live_pattern.predicted_xy - measured, axis=1)
        assert np.max(error) < 0.02

    scope = DataScope(patterns="all", min_indexed=0)
    live_poles = prepare_pole_figure(live, scope=scope)
    xml_poles = prepare_pole_figure(xml_data, scope=scope)
    np.testing.assert_allclose(
        live_poles.points, xml_poles.points, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(
        live_poles.pattern_indices, xml_poles.pattern_indices
    )


def test_indexer_selects_detector_by_id():
    indexer = Indexer(
        GEOMETRY,
        detector_id="PE1621 723-3335",
    )
    assert indexer.detector_index == 0

    with pytest.raises(ValueError, match="not present"):
        Indexer(GEOMETRY, detector_id="missing")


def test_indexer_replace_preserves_detector_identity_across_geometry_slots(tmp_path):
    original_id = "PE1621 723-3335"
    other_id = "PE0822 883-4841"
    reordered_text = GEOMETRY.read_text().replace(original_id, "temporary detector")
    reordered_text = reordered_text.replace(other_id, original_id)
    reordered_text = reordered_text.replace("temporary detector", other_id)
    reordered = tmp_path / "reordered.xml"
    reordered.write_text(reordered_text)

    indexer = Indexer(GEOMETRY, detector_id=original_id)
    replaced = indexer.replace(geo_file=reordered)

    assert replaced.detector_id == original_id
    assert replaced.detector.detector_id == original_id
    assert replaced.detector_index == 1

    by_slot = indexer.replace(geo_file=reordered, detector_index=0)
    assert by_slot.detector_id == other_id
    assert by_slot.detector_index == 0


def test_indexer_validates_index_parameters():
    with pytest.raises(ValueError, match="hkl_prefer"):
        Indexer(
            GEOMETRY,
            index_params=IndexParams(hkl_prefer=(0, 1)),
        )


def test_indexer_requires_uint16_2d_frame():
    indexer = Indexer(GEOMETRY)
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
    indexer = Indexer(GEOMETRY)
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

    indexer = Indexer(GEOMETRY)
    with pytest.raises(ValueError, match="does not match selected detector"):
        indexer.process(path)
