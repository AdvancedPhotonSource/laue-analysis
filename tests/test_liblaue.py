# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from importlib import resources


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
LIBRARY = Path(str(resources.files("lauelab.indexing.bin") / "liblaue.so"))  # installed copy


def _table_after(path: Path, marker: str, delimiter=None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    return np.loadtxt(lines[start:], delimiter=delimiter, ndmin=2)


def test_shared_library_imports_only_safe_native_dependencies():
    symbols = subprocess.run(
        ["nm", "-D", str(LIBRARY)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    imports = {line.split()[-1].split("@")[0] for line in symbols if " U " in line}

    assert "exit" not in imports
    assert "abort" not in imports
    assert "GOMP_parallel" in imports
    assert not {symbol for symbol in imports if symbol.startswith(("H5", "gsl_"))}


@pytest.mark.parametrize(
    "p2q_file",
    sorted((ROOT / "tests/data/synthetic/baseline/p2q").glob("p2q_*.txt")),
    ids=lambda path: path.stem.removeprefix("p2q_"),
)
def test_pixels_to_q_matches_lauego_reference(p2q_file):
    from lauelab.indexing._liblaue import Geometry, version

    geometry_file = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    stem = p2q_file.stem.removeprefix("p2q_")
    peaks_file = ROOT / "tests/data/synthetic/baseline/peaks" / f"peaks_{stem}.txt"
    frame_file = ROOT / "tests/data/synthetic/frames" / f"{stem}.h5"

    peak_rows = _table_after(peaks_file, "$peakList")
    expected = _table_after(p2q_file, "$N_Ghat+Intens", delimiter=",")[:, :3]
    with h5py.File(frame_file) as frame:
        detector = frame["entry1/detector"]
        start = (int(detector["startx"][0]), int(detector["starty"][0]))
        group = (int(detector["binx"][0]), int(detector["biny"][0]))
    geometry = Geometry(geometry_file)
    actual = geometry.pixels_to_q(peak_rows[:, :2], start=start, group=group)

    assert version() == "0.2.0"
    assert geometry.detector_count == 3
    assert geometry.find_detector("PE1621 723-3335") == 0
    np.testing.assert_allclose(actual, expected, atol=5e-8, rtol=0)


def test_pixels_to_q_validates_input_shape():
    from lauelab.indexing._liblaue import Geometry

    geometry = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml")
    with pytest.raises(ValueError, match="shape"):
        geometry.pixels_to_q(np.zeros(3))


def test_pixels_to_q_accepts_empty_input():
    from lauelab.indexing._liblaue import Geometry

    geometry = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml")
    result = geometry.pixels_to_q(np.empty((0, 2)))
    assert result.shape == (0, 3)


def test_geometry_exposes_wire_metadata():
    from lauelab.indexing import Geometry, WireGeometry

    wire = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml").wire

    assert isinstance(wire, WireGeometry)
    assert wire.diameter_um == pytest.approx(101.5)
    assert wire.F_um == pytest.approx(0.0)
    np.testing.assert_allclose(wire.origin_um, [0, 0, 0])
    np.testing.assert_allclose(wire.axis, [0.99999608, 0, 0.0028000002])
    np.testing.assert_allclose(wire.rotation @ wire.rotation.T, np.eye(3), atol=1e-12)
    assert wire.rotation_magnitude_deg == pytest.approx(0.4691147329582128)
    assert not wire.axis_rotated.flags.writeable


def test_geometry_exposes_detector_metadata():
    from lauelab.indexing import DetectorGeometry, Geometry

    geometry = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml")
    detector = geometry.detector(0)

    assert isinstance(detector, DetectorGeometry)
    assert (detector.nx, detector.ny) == (2048, 2048)
    assert detector.detector_id == "PE1621 723-3335"
    assert detector.size_x == pytest.approx(409600)
    assert detector.size_y == pytest.approx(409600)
    np.testing.assert_allclose(detector.translation, [28720, 3010, 513097])
    assert detector.rotation_vector.shape == (3,)
    np.testing.assert_allclose(detector.rotation @ detector.rotation.T, np.eye(3), atol=1e-12)
    assert not detector.translation.flags.writeable
    assert not detector.rotation.flags.writeable


def test_detector_projection_round_trip():
    from lauelab.indexing._liblaue import Geometry

    geometry = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml")
    detector = geometry.detector(0)
    pixels = np.asarray([[100.25, 300.5], [1024.0, 1024.0], [1900.0, 1700.0]])
    lab = detector.pixel_to_lab(pixels)
    outgoing = lab / np.linalg.norm(lab, axis=1)[:, None]
    q = outgoing - np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(detector.q_to_pixel(q), pixels, atol=1e-10)
    assert detector.pixel_to_lab(np.asarray([100.25, 300.5])).shape == (3,)
    assert detector.q_to_pixel(q[0]).shape == (2,)


def test_detector_projection_handles_invalid_and_off_detector_rays():
    from lauelab.indexing._liblaue import Geometry

    detector = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml").detector(0)

    assert np.isnan(detector.q_to_pixel([0.0, 0.0, 0.0])).all()
    off_detector = detector.q_to_pixel([1.0, 0.0, -1.0], on_detector=True)
    assert np.isnan(off_detector).all()
    with pytest.raises(ValueError, match="shape"):
        detector.pixel_to_lab(np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        detector.q_to_pixel(np.zeros(2))


def test_geometry_rejects_invalid_detector_parameters(tmp_path):
    from lauelab.indexing._liblaue import Geometry

    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "invalid-geometry.xml"
    path.write_text(source.read_text().replace("<Npixels>2048 2048</Npixels>", "<Npixels>0 2048</Npixels>"))

    with pytest.raises(ValueError, match="unable to read detector geometry"):
        Geometry(path)


def test_pixels_to_q_validates_detector_bounds_and_coordinates():
    from lauelab.indexing._liblaue import Geometry

    geometry = Geometry(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml")

    with pytest.raises(ValueError, match="detector bounds"):
        geometry.pixels_to_q(np.asarray([[2048.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        geometry.pixels_to_q(np.asarray([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="integers"):
        geometry.pixels_to_q(np.asarray([[0.0, 0.0]]), start=(0.5, 0))
    with pytest.raises(ValueError, match="invalid detector index"):
        geometry.pixels_to_q(np.asarray([[0.0, 0.0]]), detector_index=3)


def _geometry_with(source, *, count="3", detector_1=None, detector_2=None, extras=""):
    text = source.read_text()
    start_1 = text.index('\t\t<Detector N="1">')
    start_2 = text.index('\t\t<Detector N="2">')
    end_2 = text.index("\t</Detectors>")
    block_1 = text[start_1:start_2]
    block_2 = text[start_2:end_2]
    transformed = text[:start_1] + (block_1 if detector_1 is None else detector_1)
    transformed += (block_2 if detector_2 is None else detector_2) + extras + text[end_2:]
    return transformed.replace('Ndetectors="3"', f'Ndetectors="{count}"')


def test_sparse_detector_slots_are_addressable(tmp_path):
    from lauelab.indexing import Indexer
    from lauelab.indexing._liblaue import Geometry

    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "sparse.xml"
    path.write_text(_geometry_with(source, count="2", detector_1=""))
    geometry = Geometry(path)

    assert geometry.detector_count == 2
    assert geometry.find_detector("PE0822 883-4843") == 2
    assert geometry.detector(2).detector_id == "PE0822 883-4843"
    with pytest.raises(ValueError, match="invalid detector index"):
        geometry.detector(1)

    indexer = Indexer(geometry, detector_index=2)
    by_id = Indexer(geometry, detector_id="PE0822 883-4843")
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    assert by_id.detector_index == 2
    assert result.image_shape == (8, 12)


@pytest.mark.parametrize(
    "transform",
    [
        lambda source: source.replace('<Detector N="2">', '<Detector N="3">'),
        lambda source: source.replace('<Detector N="2">', '<Detector N="1">'),
        lambda source: source.replace('Ndetectors="3"', 'Ndetectors="2"'),
        lambda source: source.replace('<Detector N="2">', '<Detector>'),
        lambda source: source.replace('<Detector N="2">', '<Detector N="two">'),
        lambda source: source.replace('Ndetectors="3"', 'Ndetectors="three"'),
        lambda source: source.replace(
            '<P unit="mm">28.720 3.010 513.097</P>', ""
        ),
    ],
    ids=[
        "slot-too-large",
        "duplicate-slot",
        "count-mismatch",
        "missing-slot",
        "nonnumeric-slot",
        "nonnumeric-count",
        "incomplete",
    ],
)
def test_geometry_rejects_malformed_detector_declarations(tmp_path, transform):
    from lauelab.indexing._liblaue import Geometry

    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "malformed.xml"
    path.write_text(transform(source.read_text()))

    with pytest.raises(ValueError, match="unable to read detector geometry"):
        Geometry(path)


def test_geometry_rejects_duplicate_detector_ids(tmp_path):
    from lauelab.indexing._liblaue import Geometry

    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "duplicate-id.xml"
    path.write_text(source.read_text().replace("PE0822 883-4841", "PE1621 723-3335"))

    with pytest.raises(ValueError, match="duplicate detector ID"):
        Geometry(path)


def test_geometry_accepts_old_style_tag_value_format(tmp_path):
    # The native reader supports the pre-XML "$tag value" geometry format;
    # the XML-based duplicate-ID pre-scan must not reject it.
    from lauelab.indexing._liblaue import Geometry

    path = tmp_path / "geo_old_style.txt"
    path.write_text(
        "$filetype\tgeometryFileN\n"
        "$Ndetectors\t1\n"
        "$d0_Nx\t2048\n"
        "$d0_Ny\t2048\n"
        "$d0_sizeX\t409.6\n"
        "$d0_sizeY\t409.6\n"
        "$d0_R\t{-1.20310066,-1.21179927,-1.21933886}\n"
        "$d0_P\t{25.826,-0.728,510.812}\n"
        "$d0_detectorID\tPE1621 723-3335\n"
    )

    geometry = Geometry(path)
    assert geometry.detector_count == 1
    assert geometry.wire is None


def test_detector_geometry_ignores_unrelated_invalid_sample_and_wire(tmp_path):
    from lauelab.indexing._liblaue import Geometry

    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "detector-only.xml"
    text = source.read_text()
    text = text.replace(
        '<Origin unit="micron">0 0 0</Origin>',
        '<Origin unit="micron">nan nan nan</Origin>',
        1,
    )
    text = text.replace('<dia unit="micron">101.5</dia>', '<dia unit="micron">0</dia>')
    path.write_text(text)

    geometry = Geometry(path)
    assert geometry.detector_count == 3
    assert geometry.detector(0).detector_id == "PE1621 723-3335"
