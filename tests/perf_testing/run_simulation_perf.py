#!/usr/bin/env python3
"""Measure cold and warm reflection-simulation performance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_DIRECTORY = REPOSITORY / "tests" / "data" / "simulation"


def _peak_rss_mib() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return peak / scale


def _git_revision() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _git_dirty() -> bool | None:
    process = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    return bool(process.stdout) if process.returncode == 0 else None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor().strip() or "unknown"


def _load_case(case: str):
    import numpy as np

    from lauelab.indexing import (
        Atom,
        Cell,
        Crystal,
        DetectorGeometry,
        load_crystal,
    )

    with np.load(FIXTURE_DIRECTORY / f"portal_jzt_{case}.npz") as fixture:
        metadata = json.loads(fixture["metadata_json"].item())
    detector_data = metadata["detector"]
    detector = DetectorGeometry(
        nx=detector_data["nx"],
        ny=detector_data["ny"],
        size_x=detector_data["size_x_um"],
        size_y=detector_data["size_y_um"],
        detector_id=detector_data["detector_id"],
        translation=np.asarray(detector_data["translation_um"]),
        rotation_vector=np.asarray(detector_data["rotation_vector_rad"]),
        rotation=np.eye(3),
    )
    if case == "si":
        crystal = Crystal(
            "Si",
            227,
            Cell(0.5431, 0.5431, 0.5431),
            (Atom("Si", (0.0, 0.0, 0.0), label="Si001"),),
        )
    else:
        crystal = load_crystal(REPOSITORY / metadata["crystal"]["source"])
    reciprocal = np.asarray(metadata["reciprocal_rows_1_per_nm"], dtype=float)
    return crystal, reciprocal, detector, metadata


def _measure_worker(case: str) -> dict:
    started = time.perf_counter()
    import numpy as np

    from lauelab.analysis import simulate_reflections

    imported = time.perf_counter()
    crystal, reciprocal, detector, metadata = _load_case(case)
    prepared = time.perf_counter()
    result = simulate_reflections(
        crystal,
        reciprocal,
        detector,
        energy_range_kev=tuple(metadata["energy_range_kev"]),
        depth=float(metadata["depth_um"]),
    )
    completed = time.perf_counter()
    return {
        "import_seconds": imported - started,
        "input_setup_seconds": prepared - imported,
        "first_simulation_seconds": completed - prepared,
        "cold_total_seconds": completed - started,
        "peak_rss_mib": _peak_rss_mib(),
        "spot_count": len(result.hkl),
        "numpy_version": np.__version__,
    }


def _cold_measurement(case: str) -> dict:
    environment = os.environ.copy()
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", case],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def _benchmark(case: str, warm_runs: int) -> dict:
    cold = _cold_measurement(case)

    import numpy as np

    from lauelab.analysis import simulate_reflections
    from lauelab.analysis import simulation as simulation_module

    crystal, reciprocal, detector, metadata = _load_case(case)
    energy_range = tuple(metadata["energy_range_kev"])
    depth = float(metadata["depth_um"])
    warm_seconds = []
    spot_counts = []
    for _ in range(warm_runs):
        started = time.perf_counter()
        result = simulate_reflections(
            crystal,
            reciprocal,
            detector,
            energy_range_kev=energy_range,
            depth=depth,
        )
        warm_seconds.append(time.perf_counter() - started)
        spot_counts.append(len(result.hkl))

    return {
        "schema": "lauelab-simulation-performance-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
        },
        "input": {
            "case": case,
            "crystal": crystal.name,
            "space_group": crystal.space_group,
            "reciprocal_rows_1_per_nm": reciprocal.tolist(),
            "detector_id": detector.detector_id,
            "detector_shape_xy": [detector.nx, detector.ny],
            "detector_size_um": [detector.size_x, detector.size_y],
            "detector_translation_um": detector.translation.tolist(),
            "detector_rotation_vector_rad": detector.rotation_vector.tolist(),
            "depth_um": depth,
            "energy_range_kev": list(energy_range),
            "candidate_limit": simulation_module._CANDIDATE_LIMIT,
        },
        "cold": cold,
        "warm": {
            "runs": warm_runs,
            "seconds": warm_seconds,
            "median_seconds": statistics.median(warm_seconds),
            "minimum_seconds": min(warm_seconds),
            "maximum_seconds": max(warm_seconds),
            "peak_rss_mib": _peak_rss_mib(),
            "spot_counts": spot_counts,
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report observational reflection-simulation performance as JSON."
    )
    parser.add_argument("--case", choices=("ni", "cdte", "si"), default="ni")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.warm_runs < 1:
        parser.error("--warm-runs must be positive")
    return arguments


def main() -> None:
    arguments = _arguments()
    if arguments.worker:
        print(json.dumps(_measure_worker(arguments.case), sort_keys=True))
        return
    report = _benchmark(arguments.case, arguments.warm_runs)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
