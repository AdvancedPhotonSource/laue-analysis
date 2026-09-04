# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Local acceptance test for the Twin2 production wire scan."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import TWIN2_SKIP_REASON
from lauelab.reconstruct import Reconstructor, reconstruct, reconstruct_points


GEOMETRY_NAME = "geoN_2023-04-06_03-07-11_cor6.xml"
REFERENCE_NAME = "reference_rec8_point1"


def _twin2_fixture(points):
    """Return the directory named by ``LAUELAB_TWIN2_FIXTURE`` or skip.

    The directory must hold ``Twin2_wire_<n>.h5`` for each requested point,
    the geometry file, and the portal reference output for point 1.
    """
    root = os.environ.get("LAUELAB_TWIN2_FIXTURE")
    if not root:
        pytest.skip(TWIN2_SKIP_REASON)
    fixture = Path(root)
    required = [
        fixture / GEOMETRY_NAME,
        fixture / REFERENCE_NAME,
        *(fixture / f"Twin2_wire_{index}.h5" for index in points),
    ]
    if not all(path.exists() for path in required):
        pytest.skip(TWIN2_SKIP_REASON)
    return fixture


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("num_threads", [1, 16])
def test_twin2_point1_matches_portal_reference(tmp_path, num_threads):
    fixture = _twin2_fixture([1])
    output_base = tmp_path / f"threads_{num_threads}" / "Twin2_wire_1_"
    result = reconstruct(
        fixture / "Twin2_wire_1.h5",
        output_base,
        fixture / GEOMETRY_NAME,
        (-30.0, 90.0),
        resolution=1.0,
        verbose=1,
        percent_brightest=100.0,
        wire_edge="leading",
        memory_limit_mb=50000,
        detector_number=0,
        num_threads=num_threads,
    )
    assert result.success, f"{result.command}\n{result.log}\n{result.error}"

    for depth_index in range(121):
        actual_path = Path(f"{output_base}{depth_index}.h5")
        expected_path = fixture / REFERENCE_NAME / f"Twin2_wire_1_{depth_index}.h5"
        assert actual_path.is_file()
        assert expected_path.is_file()
        with h5py.File(actual_path, "r") as actual, h5py.File(expected_path, "r") as expected:
            np.testing.assert_array_equal(
                actual["entry1/data/data"][...], expected["entry1/data/data"][...]
            )


@pytest.mark.integration
@pytest.mark.slow
def test_native_twin2_point1_matches_portal_reference(tmp_path):
    fixture = _twin2_fixture([1])
    output_base = tmp_path / "native" / "Twin2_wire_1_"
    result = Reconstructor(
        fixture / GEOMETRY_NAME, 0, depth_range=(-30.0, 90.0), resolution=1.0,
        num_threads=16, rows_per_stripe=256,
    ).reconstruct(fixture / "Twin2_wire_1.h5", output_base)
    assert result.success, result.error
    for depth_index in range(121):
        with h5py.File(f"{output_base}{depth_index}.h5") as actual, h5py.File(
            fixture / REFERENCE_NAME / f"Twin2_wire_1_{depth_index}.h5"
        ) as expected:
            np.testing.assert_array_equal(
                actual["entry1/data/data"], expected["entry1/data/data"]
            )


@pytest.mark.integration
@pytest.mark.slow
def test_reconstruct_points_three_twin2_points(tmp_path):
    fixture = _twin2_fixture([1, 2, 3])
    inputs = [fixture / f"Twin2_wire_{index}.h5" for index in range(1, 4)]
    executable = tmp_path / "exe"
    for path in inputs[1:]:
        result = reconstruct(
            path,
            executable / f"{path.stem}_",
            fixture / GEOMETRY_NAME,
            (-30.0, 90.0),
            resolution=1.0,
            verbose=1,
            percent_brightest=100.0,
            wire_edge="leading",
            memory_limit_mb=50000,
            detector_number=0,
            num_threads=16,
        )
        assert result.success, f"{result.command}\n{result.log}\n{result.error}"

    output = tmp_path / "pool"
    results = reconstruct_points(
        inputs,
        output,
        geometry=fixture / GEOMETRY_NAME,
        detector=0,
        workers=3,
        threads_per_worker=16,
        depth_range=(-30.0, 90.0),
        rows_per_stripe=256,
    )
    assert all(result.success for result in results)

    for point in range(1, 4):
        expected_base = fixture / REFERENCE_NAME if point == 1 else executable
        for depth_index in range(121):
            actual_path = output / f"Twin2_wire_{point}_{depth_index}.h5"
            expected_path = expected_base / f"Twin2_wire_{point}_{depth_index}.h5"
            assert actual_path.is_file()
            assert expected_path.is_file()
            with h5py.File(actual_path, "r") as actual, h5py.File(
                expected_path, "r"
            ) as expected:
                np.testing.assert_array_equal(
                    actual["entry1/data/data"], expected["entry1/data/data"]
                )
