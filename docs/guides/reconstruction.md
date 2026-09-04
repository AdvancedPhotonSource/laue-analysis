# Reconstruct a wire scan

A wire-scan point is a stack of detector frames recorded while a wire moves across the diffracted beams. Reconstruction resolves that stack into one detector image for each sample depth. Depth is measured in µm along the incident beam relative to the silicon origin in the geometry file.

Use {class}`~lauelab.reconstruct.Reconstructor` for in-process reconstruction. Parse the geometry and create the reconstructor once, then reuse it for points with the same detector layout and reconstruction parameters.

```python
from pathlib import Path

from lauelab.reconstruct import Reconstructor

root = Path("tests/data")
reconstructor = Reconstructor(
    root / "geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    depth_range=(-25.0, 25.0),
    resolution=1.0,
    num_threads=4,
)
result = reconstructor.reconstruct(
    root / "reconstruction/synthetic_wire_scan.h5",
    output_base="reconstructed_",
)
peak_depth_um = result.depth_um[result.depth_intensity.argmax()]
```

The synthetic input used by the test suite is normally created in a temporary directory. The paths above illustrate the file layout rather than a shipped input file; see `tests/test_reconstructor.py` for runnable examples that create their inputs.

## File conventions

A 34-ID-E multi-image file stores one bookkeeping frame at slice 0, which is skipped. Slice 1 is both the intensity map used for the cutoff and the first scan frame. The last stored slice is read by the executable but never differenced, so the native reader stops one slice early. Both paths therefore reconstruct from slices 1 … N−2 of an N-slice file.

Stored wire vectors include acquisition bookkeeping entries. The file reader pairs scan frame `f` with stored wire entries `f + 2` and `f + 3`. This offset applies only to HDF5 input. {meth}`~lauelab.reconstruct.Reconstructor.reconstruct_array` instead accepts already aligned numeric arrays with image shape `(N, rows, columns)` and raw wire-position shape `(N + 1, 3)`. Equal `depth_range` endpoints request a single output depth.

The cutoff mask retains a requested percentage of the brightest intensity-map pixels. Header-vector normalization supports `mA` and `cnt3` with their fixed beamline divisors. Exponent normalization uses the intensity map and can derive its threshold from the lowest half of the pixels.

The `positioner` argument to `reconstruct_array` selects `"none"`, `"pm500"`, or `"alio"` correction. File reconstruction selects the historical correction from `file_time`; files recorded on or after October 2009 use the Alio identity correction. Matching the executable, a `file_time` containing the ISO `T` separator is not parsed and receives no positioner correction.

For array input, the default `ImageGeometry` describes an unbinned full frame whose dimensions equal the array dimensions. Pass an explicit `ImageGeometry` for a binned image or detector ROI. Its zero-based `start` and `group` values are in unbinned pixels. The constructor's HDF5 `normalization` vector does not apply to array input; pass one dimensionless `scale` value per frame instead. `output_pixel_type` also does not apply because the array path writes no files and returns `numpy.float64` images.

## Memory and threads

Reconstruction processes rows in stripes. When `rows_per_stripe` is `None`, the default is the smaller of 256 rows, matching the executable's `-R` default, and the number that fits within `memory_limit_mb` (8192 MiB by default):

`2 * input bytes + 2 * output bytes <= memory_limit_mb * 2**20`

At least one row is processed. Input stripes use 2 bytes per pixel per image for `uint16` input. Every other numeric input dtype is converted to float64 and uses 8 bytes per pixel per image in the stripe buffers. When `output_pixel_type` is omitted, supported input dtypes retain their corresponding WinView output type; dtypes without a WinView code, such as `uint32`, default to float64 output. `return_images=True` retains a complete float64 array with shape `(n_depths, rows, columns)` in addition to the stripe buffers. A 121-depth full 2048 by 2048 result needs approximately 4 GiB.

`num_threads` controls OpenMP threads for each reconstruction call. Its default estimates physical cores by inspecting Linux SMT topology and otherwise uses `os.cpu_count()`. For multiple points, use {func}`~lauelab.reconstruct.reconstruct_points`; it starts processes with the spawn method so workers do not inherit an initialized OpenMP runtime. Set per-process threads with `threads_per_worker`, not `num_threads`. One expected input, memory, or I/O failure returns `success=False` for that point without stopping the batch, so check every result. Avoid multiplying worker and thread counts beyond the available physical cores.

## Outputs and cross-checks

`output_base` is a filename prefix, not a directory. With `output_base="run/point_"`, reconstruction writes `run/point_0.h5`, `run/point_1.h5`, and one file for each subsequent depth, plus `run/point_summary.txt`; the directory is created when needed and existing files with those names are overwritten. Metadata is copied from the source except for image data and wire-position data. HDF5 performs float-to-integer conversion, including saturation, when an integer output type is requested.

When `norm_exponent` is set and an integer output type is selected, the written images are multiplied by the rescale factor recorded in `entry1/microDiffraction/norm_rescale` before conversion. The `images` and `depth_intensity` fields of the result keep the unscaled `numpy.float64` values, which matches the intensity totals in the executable's summary file.

The returned {class}`~lauelab.reconstruct.ReconstructionResult` contains depth coordinates in µm, intensity totals, stripe timings, and optional retained images. Argument and setup errors raise; failures after the first stripe starts return a result with `success=False`.

The supported {func}`~lauelab.reconstruct.reconstruct` subprocess path provides an independent cross-check through `reconstructN_cpu`. Use it when validating a new acquisition configuration against the in-process path.
