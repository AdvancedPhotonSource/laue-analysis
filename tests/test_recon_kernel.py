# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab._native import ffi, get_library
from lauelab.indexing import Geometry
from tests.data.reconstruction.generate_reference import (
    BINNING,
    DEPTH_RANGE_UM,
    DEPTH_RESOLUTION_UM,
    FULL_PIXELS,
    GEOMETRY_FILE,
    write_input_file,
)

pytestmark = requires_liblaue
HERE = Path(__file__).parent


def _run_kernel(input_file, *, threads=1, pixel_type=None, scale=None, norm_plane=None,
                cosmic=False, wire_edge=1):
    library = get_library()
    geometry = Geometry(GEOMETRY_FILE)
    with h5py.File(input_file) as source:
        stored = np.asarray(source["entry1/data/data"])
        intensity_map = stored[1].astype(np.float64)
        scan_images = stored[1:-1]
        images = np.ascontiguousarray(scan_images if pixel_type is None else scan_images.astype(pixel_type))
        wire = np.column_stack([
            np.asarray(source[f"entry1/wire/wire{name}"])[2:len(images)+3]
            for name in "XYZ"
        ]).astype(np.float64)
    params = ffi.new("laue_recon_params *")
    params.depth_start_um, params.depth_end_um = DEPTH_RANGE_UM
    params.resolution_um = DEPTH_RESOLUTION_UM
    params.wire_edge = wire_edge
    params.cosmic_filter = cosmic
    params.nx_full = params.ny_full = FULL_PIXELS
    params.bin_i = params.bin_j = BINNING
    params.n_rows_total, params.n_cols = images.shape[1:]
    error = ffi.new("char[256]")
    recon = library.laue_recon_create(geometry._handle, 0, params, error, 256)
    assert recon != ffi.NULL, ffi.string(error).decode()
    try:
        assert library.laue_recon_set_wire_positions(
            recon, ffi.from_buffer("double[]", wire), len(wire), library.LAUE_POSITIONER_NONE
        ) == library.LAUE_OK
        output = np.zeros((library.laue_recon_n_depths(recon), *images.shape[1:]), dtype=np.float64)
        mask = (intensity_map >= 1).astype(np.uint8)
        elapsed = ffi.new("double *")
        scale_pointer = ffi.NULL if scale is None else ffi.from_buffer("double[]", scale)
        norm_pointer = ffi.NULL if norm_plane is None else ffi.from_buffer("double[]", norm_plane)
        kind = library.LAUE_PIXEL_U16 if images.dtype == np.uint16 else library.LAUE_PIXEL_F64
        status = library.laue_recon_stripe(
            recon, ffi.from_buffer(images), kind, len(images), 0, images.shape[1],
            scale_pointer, norm_pointer, ffi.from_buffer("unsigned char[]", mask),
            ffi.from_buffer("double[]", output), threads, elapsed,
        )
        assert status == library.LAUE_OK, ffi.string(library.laue_recon_last_error(recon)).decode()
        return output
    finally:
        library.laue_recon_free(recon)


def test_reconstruction_kernel_matches_default_golden(tmp_path):
    input_file = tmp_path / "synthetic.h5"
    write_input_file(input_file)
    expected = np.load(HERE / "data/reconstruction/cpu_reference.npz")["images"]

    actual = _run_kernel(input_file)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("variant", "write_options", "kernel_options"),
    [
        ("trailing", {}, {"wire_edge": 0}),
        ("both", {}, {"wire_edge": -1}),
        ("cosmic", {"write_microdiffraction": True}, {"cosmic": True}),
        ("norm_vector", {"write_mA": True}, {"scale": "mA"}),
        ("norm_exponent", {"write_microdiffraction": True}, {"norm_plane": "exponent"}),
        ("out_uint16", {}, {}),
        ("out_int16", {}, {"wire_edge": -1}),
    ],
)
def test_reconstruction_kernel_matches_variant_goldens(
    tmp_path, variant, write_options, kernel_options
):
    input_file = tmp_path / "synthetic.h5"
    write_input_file(input_file, **write_options)
    options = dict(kernel_options)
    with h5py.File(input_file) as source:
        intensity_map = np.asarray(source["entry1/data/data"][1], dtype=np.float64)
        if options.pop("scale", None):
            options["scale"] = np.ascontiguousarray(
                    np.asarray(source["entry1/mA"][1:len(source["entry1/data/data"])-1]) / 102.0,
                dtype=np.float64,
            )
    if options.pop("norm_plane", None):
        ordered = np.sort(intensity_map.ravel())
        count = len(ordered) // 2
        total = 0.0
        total_squared = 0.0
        for value in ordered[:count]:
            total += value
            total_squared += value * value
        mean = total / count
        sigma = np.sqrt((total_squared - count * mean * mean) / (count - 1))
        threshold = float(np.float32(mean + 5 * sigma))
        exponent = float(np.float32(0.5))
        options["norm_plane"] = np.ascontiguousarray(
            np.where(intensity_map < threshold, threshold**-exponent, intensity_map**-exponent)
        )
    actual = _run_kernel(input_file, **options)
    expected = np.load(HERE / f"data/reconstruction/cpu_reference_{variant}.npz")["images"]
    metadata = json.loads((HERE / f"data/reconstruction/cpu_reference_{variant}.json").read_text())
    if np.issubdtype(expected.dtype, np.integer):
        actual = np.clip(np.trunc(actual), np.iinfo(expected.dtype).min,
                         np.iinfo(expected.dtype).max).astype(expected.dtype)
    np.testing.assert_allclose(actual, expected, **{
        key: metadata["comparison"][key] for key in ("rtol", "atol")
    })


@pytest.mark.parametrize("threads", [4, 16])
def test_reconstruction_kernel_is_thread_invariant(tmp_path, threads):
    input_file = tmp_path / "synthetic.h5"
    write_input_file(input_file)
    expected = _run_kernel(input_file, threads=1)

    np.testing.assert_array_equal(_run_kernel(input_file, threads=threads), expected)
