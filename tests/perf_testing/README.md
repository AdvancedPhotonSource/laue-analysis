# Performance Benchmark Runner (Standalone)

This directory contains a minimal, standalone performance test runner for the reconstruction code. It is completely separate from pytest and will not be collected by default test runs.

Key features:
- Requires two inputs: an HDF5 file and a staging directory (to test different filesystems).
- Pre-copies the input HDF5 into N replicas (one per parallel runner) once before all tests.
- Clears only the output folder between tests (input replicas remain).
- Runs CPU and/or GPU benchmarks over a grid:
  - CPU: Cartesian product of parallel reconstructions vs. number of threads per process.
  - GPU: Number of concurrent GPU reconstructions.
- Saves runtimes and test parameters to a JSONL file for later analysis.
- Uses the installed `laueanalysis` python API directly; no build steps required.

## Script

- `run_perf.py`: the CLI entry point.

## Usage

Basic example (CPU):
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode cpu \
  --cpu-parallel 1,2,4 \
  --cpu-threads 1,2 \
  --runs 2
```

GPU example:
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode gpu \
  --gpu-parallel 1,2,4 \
  --runs 3
```

Both CPU and GPU:
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode both \
  --cpu-parallel 1,2,4 \
  --cpu-threads 1,2 \
  --gpu-parallel 1,2 \
  --runs 1
```

Dry-run (print planned configuration and exit):
```
python tests/perf_testing/run_perf.py --h5 ... --staging-dir ... --geometry ... --depth-range 0:300 --dry-run
```

## Arguments (minimal set)

Required:
- `--h5`: Path to the input HDF5 file to benchmark.
- `--staging-dir`: Staging directory for input replicas and outputs (allows testing IO on different filesystems).
- `--geometry`: Path to geometry XML file.
- `--depth-range`: Depth range as `start:end` (floats), e.g., `0:300`.

Optional (sane defaults):
- `--mode`: `cpu|gpu|both` (default: `both`)
- `--cpu-parallel`: comma-separated ints (default: `1,2`)
- `--cpu-threads`: comma-separated ints (default: `1,2`)
- `--gpu-parallel`: comma-separated ints (default: `1,2`)
- `--runs`: repetitions per grid point (default: `1`)
- `--resolution`: depth resolution microns (default: `1.0`)
- `--verbose`: verbosity 0–3 (default: `1`)
- `--percent-brightest`: percent brightest pixels (default: `100.0`)
- `--results`: output results JSONL path (default: `<staging>/results.jsonl`)
- `--tag`: freeform label for metadata
- `--dry-run`: print configuration only

## Outputs

- Input replicas: `<staging>/input/<basename>_0.h5`, `<basename>_1.h5`, …
- Per-run outputs: `<staging>/output/run_<timestamp>/proc_<i>/out*` (created for each process).
- Results file: `<staging>/results.jsonl` (or custom via `--results`)

Results JSONL includes:
- One `params` record with the test parameters and environment info.
- One `run` record per grid point repetition with:
  - mode, parallel, threads (CPU only), run_index
  - total_wall_s
  - per_proc timings with success/return_code
  - run_id and input_replicas

Example JSONL entries:
```
{"type":"params", "... test configuration and environment metadata ..."}
{"type":"run","mode":"cpu","parallel":4,"threads":2,"run_index":0,"total_wall_s":12.345,"per_proc":[{"i":0,"elapsed_s":...},...],"run_id":"20250812T150102Z","input_replicas":4,"success":true}
```

## Notes

- The script uses the installed `laueanalysis.reconstruct` Python API (`reconstruct` and `reconstruct_gpu`). Ensure the environment can resolve these.
- GPU tests will simply launch multiple concurrent GPU reconstructions; if you need per-GPU pinning on multi-GPU systems, this script can be extended to set `CUDA_VISIBLE_DEVICES` per process.
- Long runs are expected; use `--dry-run` first to verify the plan.
