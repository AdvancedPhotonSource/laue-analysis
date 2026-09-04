# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the in-process reconstruction driver."""

import json
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import InputError
from lauelab.reconstruct import (
    ImageGeometry, ReconstructionResult, Reconstructor, find_executable, reconstruct,
    reconstruct_points,
)
from lauelab.reconstruct._reader import positioner_from_file_time, read_scan_info
from lauelab.reconstruct.reconstruct import _find_executable
from tests.data.reconstruction.generate_reference import (
    BINNING, DEPTH_RANGE_UM, FULL_PIXELS, GEOMETRY_FILE, VARIANTS,
    write_input_file,
)

pytestmark = requires_liblaue
REFERENCE_DIR = Path(__file__).parent / "data/reconstruction"


def _reconstructor(variant=None, **changes):
    options = VARIANTS.get(variant, {})
    values = dict(
        geometry=GEOMETRY_FILE,
        detector=0,
        depth_range=DEPTH_RANGE_UM,
        wire_edge=options.get("wire_edge", "leading"),
        normalization=options.get("normalization"),
        norm_exponent=options.get("norm_exponent"),
        cosmic_filter=options.get("cosmic_filter", False),
        output_pixel_type=options.get("output_pixel_type", 5),
        num_threads=1,
        rows_per_stripe=31,
    )
    values.update(changes)
    return Reconstructor(**values)


@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_reconstructor_matches_golden(tmp_path, variant):
    options = VARIANTS.get(variant, {})
    source = tmp_path / "synthetic.h5"
    write_input_file(source, write_mA=options.get("write_mA", False),
                     write_microdiffraction=options.get("write_microdiffraction", False))
    result = _reconstructor(variant).reconstruct(source, return_images=True)
    suffix = f"_{variant}" if variant else ""
    expected = np.load(REFERENCE_DIR / f"cpu_reference{suffix}.npz")
    metadata = json.loads((REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text())

    assert result.success, result.error
    np.testing.assert_array_equal(result.depth_um, expected["depth_um"])
    actual = result.images
    if np.issubdtype(expected["images"].dtype, np.integer):
        info = np.iinfo(expected["images"].dtype)
        actual = np.clip(np.trunc(actual), info.min, info.max).astype(expected["images"].dtype)
    tolerance = metadata["comparison"]
    np.testing.assert_allclose(actual, expected["images"], rtol=tolerance["rtol"], atol=tolerance["atol"])


def test_array_and_file_paths_are_identical(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    reconstructor = _reconstructor()
    file_result = reconstructor.reconstruct(source, return_images=True)
    with h5py.File(source) as handle:
        images = np.asarray(handle["entry1/data/data"])[1:-1]
        intensity = np.asarray(handle["entry1/data/data"])[1]
        wire = np.column_stack([
            np.asarray(handle[f"entry1/wire/wire{name}"])[2:len(images) + 3]
            for name in "XYZ"
        ])
    image_geometry = ImageGeometry(FULL_PIXELS, FULL_PIXELS, group=(BINNING, BINNING),
                                   n_rows=images.shape[1], n_cols=images.shape[2])
    array_result = reconstructor.reconstruct_array(
        images, wire, intensity_map=intensity, positioner="alio",
        image_geometry=image_geometry,
    )
    assert array_result.success
    np.testing.assert_array_equal(array_result.images, file_result.images)


@pytest.mark.parametrize("threads", [1, 4])
def test_driver_is_thread_invariant(tmp_path, threads):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    expected = _reconstructor(num_threads=1).reconstruct(source, return_images=True)
    actual = _reconstructor(num_threads=threads).reconstruct(source, return_images=True)
    np.testing.assert_array_equal(actual.images, expected.images)
    np.testing.assert_array_equal(actual.depth_intensity, expected.depth_intensity)


def test_output_files_use_hdf5_conversion_and_drop_wire_group(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    output_base = tmp_path / "out" / "recon_"
    result = _reconstructor("out_uint16").reconstruct(source, output_base)

    assert result.success, result.error
    assert result.images is None
    assert result.output_files[-1] == f"{output_base}summary.txt"
    with h5py.File(result.output_files[0]) as output:
        assert output["entry1/data/data"].dtype == np.dtype("uint16")
        assert "entry1/wire" not in output
        assert output["entry1/data/data"].attrs["signal"] == 1
        assert "Facility/facility_name" in output


def _objects(handle):
    objects = {}
    handle.visititems(lambda name, obj: objects.setdefault(name, obj))
    return objects


def _summary(path):
    lines = Path(path).read_text().splitlines()
    tags = {}
    array_start = None
    for index, line in enumerate(lines):
        match = re.match(r"^\$(\w+)\s+(.*?)(\s*//.*)?$", line)
        if match:
            tags[match.group(1)] = match.group(2).rstrip()
            if match.group(1) == "array0":
                array_start = index + 1
    return tags, "\n".join(lines[array_start:])


def test_native_output_is_structurally_equivalent_to_subprocess(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    native_base = tmp_path / "native_"
    subprocess_base = tmp_path / "subprocess_"
    native = _reconstructor().reconstruct(source, native_base)
    executable = find_executable()
    try:
        external = reconstruct(
            source, subprocess_base, GEOMETRY_FILE, DEPTH_RANGE_UM,
            output_pixel_type=5, num_threads=1, executable=executable,
        )
    finally:
        _find_executable.cache_clear()
    assert native.success and external.success
    for index in range(len(native.depth_um)):
        with h5py.File(f"{native_base}{index}.h5") as left, h5py.File(f"{subprocess_base}{index}.h5") as right:
            left_objects = _objects(left)
            right_objects = _objects(right)
            assert left_objects.keys() == right_objects.keys()
            for name in left_objects:
                a, b = left_objects[name], right_objects[name]
                assert type(a) is type(b)
                assert a.attrs.keys() == b.attrs.keys()
                for attribute in a.attrs:
                    np.testing.assert_array_equal(a.attrs[attribute], b.attrs[attribute])
                if isinstance(a, h5py.Dataset):
                    assert a.dtype == b.dtype
                    assert a.shape == b.shape
                    np.testing.assert_array_equal(a, b)

    native_tags, native_array = _summary(f"{native_base}summary.txt")
    external_tags, external_array = _summary(f"{subprocess_base}summary.txt")
    # Paths/program names differ by design, timings are nondeterministic, and
    # each driver selects its stripe size independently.
    excluded = {"ws_outfile", "program_name", "executionTime", "rows_at_one_time"}
    for tag, value in native_tags.items():
        if tag not in excluded:
            assert external_tags[tag] == value
    assert native_array == external_array


def test_runtime_write_failure_returns_partial_result(tmp_path, monkeypatch):
    from lauelab.reconstruct.reconstructor import write_stripe as real_write_stripe

    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    calls = 0

    def fail_second_write(handles, row0, values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("output became unavailable")
        real_write_stripe(handles, row0, values)

    monkeypatch.setattr("lauelab.reconstruct.reconstructor.write_stripe", fail_second_write)
    output_base = tmp_path / "failed_"
    result = _reconstructor(rows_per_stripe=32).reconstruct(source, output_base)
    assert not result.success
    assert "output became unavailable" in result.error
    assert result.last_completed_stripe == 0
    with h5py.File(f"{output_base}0.h5") as output:
        image = np.asarray(output["entry1/data/data"])
    assert np.any(image[:32])
    assert not np.any(image[32:])


def test_missing_wire_vector_normalization_raises_input_error(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with pytest.raises(InputError, match="normalization vector 'mA' is missing"):
        _reconstructor(normalization="mA").reconstruct(source)


def test_invalid_geometry_detector_raises_before_work(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with h5py.File(source, "r+") as handle:
        handle["entry1/detector/Nx"][0] = FULL_PIXELS // 2
    output_base = tmp_path / "missing" / "failed_"
    with pytest.raises(InputError, match="image geometry is outside detector bounds"):
        _reconstructor().reconstruct(source, output_base)
    assert not output_base.parent.exists()


def test_reader_requires_five_stored_slices(tmp_path):
    source = tmp_path / "short.h5"
    with h5py.File(source, "w") as output:
        output.create_dataset("entry1/data/data", shape=(4, 2, 2), dtype=np.uint16)
    with h5py.File(source) as handle:
        with pytest.raises(InputError, match="needs at least 5 stored slices"):
            read_scan_info(handle)


@pytest.mark.parametrize(
    ("file_time", "expected"),
    [
        ("2008-01-01 12:00:00", "pm500"),
        ("2008-01-01T12:00:00", "none"),
        ("2023-04-13T03:39:25-06:00", "none"),
    ],
)
def test_positioner_from_file_time_matches_executable(file_time, expected):
    assert positioner_from_file_time(file_time) == expected


def test_reconstruct_points_uses_one_reconstructor_per_spawn_worker(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    results = reconstruct_points(
        [source], tmp_path / "batch", geometry=GEOMETRY_FILE, detector=0,
        workers=1, threads_per_worker=1, depth_range=DEPTH_RANGE_UM,
        output_pixel_type=5, rows_per_stripe=32,
    )
    assert len(results) == 1
    assert results[0].success, results[0].error
    assert Path(results[0].output_files[0]).name == "synthetic_0.h5"


def test_result_preserves_six_positional_fields():
    result = ReconstructionResult(True, ["a"], "log", None, "command", 0)
    assert result.success
    assert result.images is None
    assert result._asdict()["command"] == "command"
