#!/usr/bin/env python3
# Copyright (c) 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Measure the reconstruction executable on Twin2 point 1."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from lauelab.reconstruct import Reconstructor, reconstruct


THREADS = (1, 8, 16, 32)
FLOAT = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"


def _times(pattern: str, log: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, log)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=os.environ.get("LAUELAB_TWIN2_FIXTURE"),
        help="directory containing Twin2_wire_1.h5 and its geometry file "
             "(default: the LAUELAB_TWIN2_FIXTURE environment variable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="retain outputs in this directory instead of using temporary storage",
    )
    args = parser.parse_args()

    if args.fixture is None:
        print("SKIP: set LAUELAB_TWIN2_FIXTURE or pass --fixture")
        return 0
    input_file = args.fixture / "Twin2_wire_1.h5"
    geometry = args.fixture / "geoN_2023-04-06_03-07-11_cor6.xml"
    if not input_file.is_file() or not geometry.is_file():
        print(f"SKIP: Twin2 wire-scan fixture not available under {args.fixture}")
        return 0

    temporary = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="lauelab-reconstruct-perf-")
        output_dir = Path(temporary.name)
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    print("subprocess executable")
    print("threads  wall_s  cpu_s  compute/stripe_s  io/stripe_s  stripes")
    executable_times = {}
    try:
        for threads in THREADS:
            run_dir = output_dir / f"threads_{threads}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir()

            start = time.perf_counter()
            result = reconstruct(
                input_file,
                run_dir / "Twin2_wire_1_",
                geometry,
                (-30.0, 90.0),
                resolution=1.0,
                verbose=2,
                percent_brightest=100.0,
                wire_edge="leading",
                memory_limit_mb=50000,
                num_threads=threads,
            )
            wall = time.perf_counter() - start
            if not result.success:
                print(result.log, file=sys.stderr)
                print(result.error or "reconstruction failed", file=sys.stderr)
                return result.return_code or 1

            compute = _times(
                r"depth-resolving done in " + FLOAT + r" seconds", result.log
            )
            reads = _times(
                FLOAT
                + r" seconds spent in reading (?:first stripe|data of the next stripe)",
                result.log,
            )
            writes = _times(
                FLOAT + r" seconds spent in writing previous stripe", result.log
            )
            cpu_match = re.search(r"CPU time of " + FLOAT + r" seconds", result.log)
            cpu = float(cpu_match.group(1)) if cpu_match else float("nan")
            stripes = len(compute)
            io_per_stripe = (
                (sum(reads) + sum(writes)) / stripes if stripes else 0.0
            )
            executable_times[threads] = _mean(compute)
            print(
                f"{threads:7d}  {wall:6.1f}  {cpu:5.1f}  "
                f"{executable_times[threads]:16.3f}  {io_per_stripe:11.3f}  {stripes:7d}"
            )

            if temporary is not None:
                shutil.rmtree(run_dir)

        print("\nin-process driver")
        print("threads  wall_s  compute/stripe_s  io/stripe_s  stripes  kernel/exe")
        for threads in THREADS:
            run_dir = output_dir / f"native_threads_{threads}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir()
            driver = Reconstructor(
                geometry, 0, depth_range=(-30.0, 90.0), resolution=1.0,
                percent_brightest=100.0, num_threads=threads,
                rows_per_stripe=256,
            )
            start = time.perf_counter()
            result = driver.reconstruct(input_file, run_dir / "Twin2_wire_1_")
            wall = time.perf_counter() - start
            if not result.success:
                print(result.error or "native reconstruction failed", file=sys.stderr)
                return 1
            compute = _mean([timing.compute_seconds for timing in result.timings])
            io = _mean([timing.read_seconds + timing.write_seconds for timing in result.timings])
            print(
                f"{threads:7d}  {wall:6.1f}  {compute:16.3f}  {io:11.3f}  "
                f"{len(result.timings):7d}  {compute / executable_times[threads]:10.3f}"
            )
            if temporary is not None:
                shutil.rmtree(run_dir)
    finally:
        if temporary is not None:
            temporary.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
