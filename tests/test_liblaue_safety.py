import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from conftest import requires_liblaue


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue


def _run_python(script, *arguments):
    environment = dict(os.environ)  # child imports the same installed laueanalysis as the test runner
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
    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
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


def test_repeated_peak_search_releases_native_allocations(tmp_path):
    valgrind = shutil.which("valgrind")
    compiler = shutil.which("cc")
    if valgrind is None:
        pytest.skip("Valgrind is required for the native memory regression test")
    if compiler is None:
        pytest.skip("a C compiler is required for the native memory regression test")

    from importlib import resources

    library = Path(str(resources.files("laueanalysis.indexing.bin") / "liblaue.so"))
    source = ROOT / "tests/native/peaksearch_memory.c"
    executable = tmp_path / "peaksearch-memory"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O0",
            "-g",
            f"-I{ROOT / 'src/laueanalysis/indexing/src/liblaue'}",
            str(source),
            str(library),
            "-lm",
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr

    checked = subprocess.run(
        [
            valgrind,
            "--quiet",
            "--leak-check=full",
            "--show-leak-kinds=definite",
            "--errors-for-leak-kinds=definite",
            "--error-exitcode=99",
            str(executable),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
