# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Local acceptance test for the Twin2 production wire scan."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from lauelab.reconstruct import Reconstructor, reconstruct, reconstruct_points


FIXTURE = Path(__file__).resolve().parents[1] / "sandbox" / "data" / "twin2_wire"
INPUT = FIXTURE / "Twin2_wire_1.h5"
GEOMETRY = FIXTURE / "geoN_2023-04-06_03-07-11_cor6.xml"
REFERENCE = FIXTURE / "reference_rec8_point1"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("num_threads", [1, 16])
def test_twin2_point1_matches_portal_reference(tmp_path, num_threads):
    if not INPUT.is_file() or not GEOMETRY.is_file() or not REFERENCE.is_dir():
        pytest.skip("Twin2 wire-scan fixture not available")

    output_base = tmp_path / f"threads_{num_threads}" / "Twin2_wire_1_"
    result = reconstruct(
        INPUT,
        output_base,
        GEOMETRY,
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
        expected_path = REFERENCE / f"Twin2_wire_1_{depth_index}.h5"
        assert actual_path.is_file()
        assert expected_path.is_file()
        with h5py.File(actual_path, "r") as actual, h5py.File(expected_path, "r") as expected:
            np.testing.assert_array_equal(
                actual["entry1/data/data"][...], expected["entry1/data/data"][...]
            )


@pytest.mark.integration
@pytest.mark.slow
def test_native_twin2_point1_matches_portal_reference(tmp_path):
    if not INPUT.is_file() or not GEOMETRY.is_file() or not REFERENCE.is_dir():
        pytest.skip("Twin2 wire-scan fixture not available")

    output_base = tmp_path / "native" / "Twin2_wire_1_"
    result = Reconstructor(
        GEOMETRY, 0, depth_range=(-30.0, 90.0), resolution=1.0,
        num_threads=16, rows_per_stripe=256,
    ).reconstruct(INPUT, output_base)
    assert result.success, result.error
    for depth_index in range(121):
        with h5py.File(f"{output_base}{depth_index}.h5") as actual, h5py.File(
            REFERENCE / f"Twin2_wire_1_{depth_index}.h5"
        ) as expected:
            np.testing.assert_array_equal(
                actual["entry1/data/data"], expected["entry1/data/data"]
            )


@pytest.mark.integration
@pytest.mark.slow
def test_reconstruct_points_three_twin2_points(tmp_path):
    inputs = [FIXTURE / f"Twin2_wire_{index}.h5" for index in range(1, 4)]
    if (
        not all(path.is_file() for path in inputs)
        or not GEOMETRY.is_file()
        or not REFERENCE.is_dir()
    ):
        pytest.skip("Twin2 wire-scan fixture not available")

    executable = tmp_path / "exe"
    for path in inputs[1:]:
        result = reconstruct(
            path,
            executable / f"{path.stem}_",
            GEOMETRY,
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
        geometry=GEOMETRY,
        detector=0,
        workers=3,
        threads_per_worker=16,
        depth_range=(-30.0, 90.0),
        rows_per_stripe=256,
    )
    assert all(result.success for result in results)

    for point in range(1, 4):
        expected_base = REFERENCE if point == 1 else executable
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
