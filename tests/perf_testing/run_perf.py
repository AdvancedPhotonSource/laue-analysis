#!/usr/bin/env python3
"""
Standalone performance runner for laueanalysis reconstruction.

- Does NOT use pytest and won't be auto-collected.
- Requires two parameters: input HDF5 file and a staging directory.
- Copies the HDF5 file into N replicas (one per parallel runner) once before all tests.
- For each test point, clears the output folder only (input replicas stay the same).
- Runs CPU and/or GPU reconstructions across a grid of parallelism (and threads for CPU).
- Saves runtimes and parameters to a JSONL results file for later analysis.

Example:
  python tests/perf_testing/run_perf.py \
    --h5 /data/sample.h5 \
    --staging-dir /mnt/ssd/stage \
    --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
    --depth-range 0:300 \
    --mode cpu \
    --cpu-parallel 1,2,4 \
    --cpu-threads 1,2 \
    --runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# laueanalysis is assumed to be installed in the environment (as per user request).
from laueanalysis.reconstruct import reconstruct, reconstruct_gpu


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int list: {text}")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"All values must be positive: {text}")
    return values


def parse_depth_range(text: str) -> Tuple[float, float]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Depth range must be in the form start:end")
    a, b = text.split(":", 1)
    try:
        start = float(a)
        end = float(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid depth range: {text}")
    if start >= end:
        raise argparse.ArgumentTypeError("Depth range start must be less than end")
    return start, end


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_input_replicas(input_h5: Path, input_dir: Path, nmax: int) -> List[Path]:
    """
    Create N input replicas: basename_0.h5 .. basename_{nmax-1}.h5
    If a replica already exists, leave it in place to avoid unnecessary I/O.
    """
    ensure_dir(input_dir)
    stem = input_h5.stem
    suffix = input_h5.suffix or ".h5"
    replicas = []
    for i in range(nmax):
        dst = input_dir / f"{stem}_{i}{suffix}"
        if not dst.exists():
            shutil.copy2(input_h5, dst)
        replicas.append(dst)
    return replicas


def clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)


def get_git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def worker_cpu(i: int,
               input_path: str,
               out_base: str,
               geometry: str,
               depth_start: float,
               depth_end: float,
               resolution: float,
               verbose: int,
               num_threads: int,
               percent_brightest: float) -> dict:
    t0 = time.perf_counter()
    res = reconstruct(
        input_file=input_path,
        output_file=out_base,
        geometry_file=geometry,
        depth_range=(depth_start, depth_end),
        resolution=resolution,
        verbose=verbose,
        percent_brightest=percent_brightest,
        num_threads=num_threads,
    )
    t1 = time.perf_counter()
    return {
        "i": i,
        "start_s": t0,
        "end_s": t1,
        "elapsed_s": t1 - t0,
        "success": bool(res.success),
        "return_code": int(res.return_code),
    }


def worker_gpu(i: int,
               input_path: str,
               out_base: str,
               geometry: str,
               depth_start: float,
               depth_end: float,
               resolution: float,
               verbose: int,
               percent_brightest: float) -> dict:
    t0 = time.perf_counter()
    res = reconstruct_gpu(
        input_file=input_path,
        output_file=out_base,
        geometry_file=geometry,
        depth_range=(depth_start, depth_end),
        resolution=resolution,
        verbose=verbose,
        percent_brightest=percent_brightest,
    )
    t1 = time.perf_counter()
    return {
        "i": i,
        "start_s": t0,
        "end_s": t1,
        "elapsed_s": t1 - t0,
        "success": bool(res.success),
        "return_code": int(res.return_code),
    }


def run_matrix_cpu(args, replicas: List[Path], results_path: Path) -> None:
    for parallel in args.cpu_parallel:
        for threads in args.cpu_threads:
            for run_index in range(args.runs):
                print(f"[CPU] parallel={parallel} threads={threads} run={run_index+1}/{args.runs}")
                # Clear outputs
                clear_output_dir(args.output_dir)
                run_id = utc_timestamp()
                run_dir = args.output_dir / f"run_{run_id}"
                ensure_dir(run_dir)

                # Prepare per-proc output dirs and assign replicas 0..parallel-1
                jobs = []
                for i in range(parallel):
                    proc_dir = run_dir / f"proc_{i}"
                    ensure_dir(proc_dir)
                    out_base = proc_dir / "out"
                    jobs.append((i, str(replicas[i]), str(out_base)))

                t0 = time.perf_counter()
                results = []
                try:
                    with ProcessPoolExecutor(max_workers=parallel) as ex:
                        futures = [
                            ex.submit(
                                worker_cpu,
                                i, inpath, outbase,
                                str(args.geometry),
                                args.depth_start, args.depth_end,
                                args.resolution, args.verbose,
                                threads,
                                args.percent_brightest,
                            )
                            for (i, inpath, outbase) in jobs
                        ]
                        for fut in as_completed(futures):
                            results.append(fut.result())
                    success = all(r.get("success", False) for r in results)
                except KeyboardInterrupt:
                    print("Interrupted. Attempting to cancel pending CPU tasks...")
                    success = False
                    raise
                finally:
                    t1 = time.perf_counter()

                record = {
                    "type": "run",
                    "ts": utc_timestamp(),
                    "mode": "cpu",
                    "parallel": parallel,
                    "threads": threads,
                    "run_index": run_index,
                    "total_wall_s": t1 - t0,
                    "per_proc": sorted(results, key=lambda r: r["i"]),
                    "run_id": run_id,
                    "input_replicas": parallel,
                    "success": success,
                }
                write_jsonl(results_path, record)


def run_matrix_gpu(args, replicas: List[Path], results_path: Path) -> None:
    for parallel in args.gpu_parallel:
        for run_index in range(args.runs):
            print(f"[GPU] parallel={parallel} run={run_index+1}/{args.runs}")
            # Clear outputs
            clear_output_dir(args.output_dir)
            run_id = utc_timestamp()
            run_dir = args.output_dir / f"run_{run_id}"
            ensure_dir(run_dir)

            jobs = []
            for i in range(parallel):
                proc_dir = run_dir / f"proc_{i}"
                ensure_dir(proc_dir)
                out_base = proc_dir / "out"
                jobs.append((i, str(replicas[i]), str(out_base)))

            t0 = time.perf_counter()
            results = []
            try:
                with ProcessPoolExecutor(max_workers=parallel) as ex:
                    futures = [
                        ex.submit(
                            worker_gpu,
                            i, inpath, outbase,
                            str(args.geometry),
                            args.depth_start, args.depth_end,
                            args.resolution, args.verbose,
                            args.percent_brightest,
                        )
                        for (i, inpath, outbase) in jobs
                    ]
                    for fut in as_completed(futures):
                        results.append(fut.result())
                success = all(r.get("success", False) for r in results)
            except KeyboardInterrupt:
                print("Interrupted. Attempting to cancel pending GPU tasks...")
                success = False
                raise
            finally:
                t1 = time.perf_counter()

            record = {
                "type": "run",
                "ts": utc_timestamp(),
                "mode": "gpu",
                "parallel": parallel,
                "run_index": run_index,
                "total_wall_s": t1 - t0,
                "per_proc": sorted(results, key=lambda r: r["i"]),
                "run_id": run_id,
                "input_replicas": parallel,
                "success": success,
            }
            write_jsonl(results_path, record)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Standalone performance runner for laueanalysis reconstruction.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to input HDF5 file to benchmark.")
    parser.add_argument("--staging-dir", required=True, type=Path, help="Staging directory to copy inputs and write outputs.")
    parser.add_argument("--geometry", required=True, type=Path, help="Path to geometry XML file.")
    parser.add_argument("--depth-range", required=True, type=parse_depth_range, help="Depth range as start:end (floats).")

    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both", help="Which backends to run.")
    parser.add_argument("--cpu-parallel", type=parse_int_list, default=parse_int_list("1,2"), help="Comma-separated list of concurrent CPU reconstructions.")
    parser.add_argument("--cpu-threads", type=parse_int_list, default=parse_int_list("1,2"), help="Comma-separated list of CPU thread counts.")
    parser.add_argument("--gpu-parallel", type=parse_int_list, default=parse_int_list("1,2"), help="Comma-separated list of concurrent GPU reconstructions.")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per matrix point.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Depth resolution (microns).")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity 0-3.")
    parser.add_argument("--percent-brightest", type=float, default=100.0, help="Process only N%% brightest pixels.")
    parser.add_argument("--results", type=Path, default=None, help="Results JSONL path (default: <staging>/results.jsonl)")
    parser.add_argument("--tag", type=str, default="", help="Optional freeform label.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned matrix and exit.")

    args = parser.parse_args(argv)

    # Normalize and validate
    args.h5 = args.h5.expanduser().resolve()
    args.geometry = args.geometry.expanduser().resolve()
    args.staging_dir = args.staging_dir.expanduser().resolve()
    if not args.h5.is_file():
        print(f"Input HDF5 not found: {args.h5}", file=sys.stderr)
        return 2
    if not args.geometry.is_file():
        print(f"Geometry XML not found: {args.geometry}", file=sys.stderr)
        return 2

    args.input_dir = args.staging_dir / "input"
    args.output_dir = args.staging_dir / "output"
    ensure_dir(args.staging_dir)

    args.depth_start, args.depth_end = args.depth_range

    # Determine maximum parallelism across selected modes
    def max_or_zero(vals: List[int]) -> int:
        return max(vals) if vals else 0

    nmax = 0
    if args.mode in ("cpu", "both"):
        nmax = max(nmax, max_or_zero(args.cpu_parallel))
    if args.mode in ("gpu", "both"):
        nmax = max(nmax, max_or_zero(args.gpu_parallel))

    if nmax <= 0:
        print("No parallelism specified (nmax=0). Provide values via --cpu-parallel/--gpu-parallel.", file=sys.stderr)
        return 2

    # Compute results path
    results_path = args.results or (args.staging_dir / "results.jsonl")

    # Write a parameters header line with environment metadata
    params_header = {
        "type": "params",
        "ts": utc_timestamp(),
        "h5": str(args.h5),
        "staging_dir": str(args.staging_dir),
        "geometry": str(args.geometry),
        "depth_range": [args.depth_start, args.depth_end],
        "resolution": args.resolution,
        "verbose": args.verbose,
        "percent_brightest": args.percent_brightest,
        "mode": args.mode,
        "cpu_parallel": args.cpu_parallel,
        "cpu_threads": args.cpu_threads,
        "gpu_parallel": args.gpu_parallel,
        "runs": args.runs,
        "tag": args.tag,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "git": {
            "sha": get_git_sha(Path(__file__).resolve().parent.parent),  # repo root heuristic: tests/perf_testing/../..
        },
        "session_id": utc_timestamp(),
    }

    # Plan and optionally exit for dry-run
    print("Planned configuration:")
    print(json.dumps(params_header, indent=2))
    if args.dry_run:
        print("Dry-run mode; exiting without executing.")
        return 0

    # Prepare input replicas once
    print(f"Preparing input replicas (N={nmax}) in {args.input_dir} ...")
    replicas = copy_input_replicas(args.h5, args.input_dir, nmax)
    print("Input replicas ready.")

    # Write header
    write_jsonl(results_path, params_header)
    print(f"Writing results to: {results_path}")

    # Execute matrices
    try:
        if args.mode in ("cpu", "both"):
            run_matrix_cpu(args, replicas, results_path)
        if args.mode in ("gpu", "both"):
            run_matrix_gpu(args, replicas, results_path)
    except KeyboardInterrupt:
        print("Interrupted by user. Partial results (if any) remain in results file.")
        return 130

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
