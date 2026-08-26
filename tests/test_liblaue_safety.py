import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "src/laueanalysis/indexing/bin/liblaue.so"
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="liblaue.so is not built")


def _run_python(script, *arguments):
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_invalid_native_calls_return_errors_without_terminating_python():
    script = r'''
from laueanalysis.indexing._liblaue import ffi, get_library

lib = get_library()
error = ffi.new("char[256]")
result = ffi.new("laue_frame_result *")
params = ffi.new("laue_peak_params *")
info = ffi.new("laue_detector_info *")

assert lib.laue_geometry_from_file(ffi.NULL, error, 256) == ffi.NULL
assert ffi.string(error) == b"geometry path is NULL"
assert lib.laue_crystal_from_file(ffi.NULL, error, 256) == ffi.NULL
assert ffi.string(error) == b"crystal path is NULL"
assert lib.laue_crystal_create(b"bad", 0, 1, 1, 1, 90, 90, 90,
                               ffi.NULL, 0, error, 256) == ffi.NULL
assert ffi.string(error) == b"invalid crystal parameters"

assert lib.laue_geometry_detector_count(ffi.NULL) == 0
assert lib.laue_geometry_find_detector(ffi.NULL, b"detector") == -1
assert lib.laue_geometry_detector_info(ffi.NULL, 0, info, error, 256) == 1
assert ffi.string(error) == b"invalid detector index"

assert lib.laue_find_peaks(ffi.NULL, 1, 1, params, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"invalid peak-search input"
assert lib.laue_pixels_to_q(ffi.NULL, 0, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"geometry is NULL"
assert lib.laue_index(ffi.NULL, ffi.NULL, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"invalid indexing input"

assert lib.laue_find_peaks(ffi.NULL, 1, 1, params, ffi.NULL) == 1
assert lib.laue_pixels_to_q(ffi.NULL, 0, ffi.NULL) == 1
assert lib.laue_index(ffi.NULL, ffi.NULL, ffi.NULL) == 1
lib.laue_frame_result_free(result)
lib.laue_frame_result_free(result)
lib.laue_frame_result_free(ffi.NULL)
lib.laue_geometry_free(ffi.NULL)
lib.laue_crystal_free(ffi.NULL)
'''
    completed = _run_python(script)

    assert completed.returncode == 0, completed.stderr


def test_out_of_range_detector_slot_is_rejected_without_crashing(tmp_path):
    source = ROOT / "sandbox/data/i71/geoN_2026-07-07_16-30-21.xml"
    path = tmp_path / "slot-3.xml"
    path.write_text(source.read_text().replace('<Detector N="2">', '<Detector N="3">'))
    script = r'''
import sys
from laueanalysis.indexing._liblaue import Geometry

try:
    Geometry(sys.argv[1])
except ValueError as error:
    assert "unable to read detector geometry" in str(error)
else:
    raise AssertionError("invalid detector slot was accepted")
'''

    completed = _run_python(script, path)

    assert completed.returncode == 0, completed.stderr
