# Reconstruct a wire scan

A wire-scan point is a stack of detector frames recorded while a wire moves across the diffracted beams at one sample position. Reconstruction resolves that stack into one detector image for each sample depth. Depth is in µm along the incident beam, measured from the Si origin in the geometry file. [Wire-scan reconstruction](../concepts/wire-scan-reconstruction.md) explains the model; this page covers the decisions needed to run it.

## Choose the inputs

Reconstruction needs a geometry file with a complete wire section, a detector slot, and a depth grid. Geometry files without a wire section load for indexing but raise {class}`~lauelab.indexing.InputError` here.

```{warning}
`detector` is a physical slot in the geometry file, not an ordinal position among active detectors. See [Geometry](geometry.md) before selecting a slot.
```

The constructor options control the depth grid, the wire edge, and the normalizations.

| Option | Default | Units | Effect |
| --- | --- | --- | --- |
| `depth_range` | required | µm | Inclusive `(start, end)`. Endpoints are rounded to multiples of `resolution`. Equal endpoints give one depth. |
| `resolution` | `1.0` | µm | Spacing between reconstructed depths. |
| `wire_edge` | `"leading"` | | `"leading"`, `"trailing"`, or `"both"`. |
| `percent_brightest` | `100.0` | % | Share of intensity-map pixels to reconstruct. Pixels below 1 count are always skipped. |
| `normalization` | `None` | | HDF5 vector below `entry1` that scales each frame. File input only. |
| `norm_exponent` | `None` | | Exponent normalization from the intensity map. |
| `norm_threshold` | `None` | counts | Threshold for exponent normalization. `None` derives it from the intensity map. |
| `cosmic_filter` | `False` | | Remove single-frame spikes before differencing. |
| `output_pixel_type` | `None` | | Pixel type code for written files. |
| `num_threads` | `None` | | OpenMP threads per call. `None` estimates physical cores. |
| `rows_per_stripe` | `None` | rows | Rows per stripe. `None` uses at most 256, fewer to respect the memory limit. |
| `memory_limit_mb` | `8192` | MiB | Stripe-buffer limit used to derive the default stripe size. |

See the [reference](../reference/reconstruction.md) for accepted ranges and the [algorithm page](../concepts/algorithms/depth-reconstruction.md) for how each option enters the calculation.

## Reconstruct one point

Create the {class}`~lauelab.reconstruct.Reconstructor` once and reuse it for every point with the same detector layout and options. The repository ships a geometry file but no wire-scan file, so replace `point_1.h5` with your own point. The test suite generates a synthetic point from `tests/data/reconstruction/generate_reference.py`.

```python
from pathlib import Path

from lauelab.reconstruct import Reconstructor

reconstructor = Reconstructor(
    Path("tests/data/geo/geoN_2022-03-29_14-15-05.xml"),
    detector=0,
    depth_range=(-25.0, 25.0),
    resolution=1.0,
    num_threads=4,
)
result = reconstructor.reconstruct("point_1.h5", return_images=True)

assert result.success, result.error
peak_depth_um = result.depth_um[result.depth_intensity.argmax()]
brightest_image = result.images[result.depth_intensity.argmax()]
```

`depth_um` has shape `(n_depths,)`. `depth_intensity` holds the sum of each reconstructed image and has the same shape. `images` has shape `(n_depths, rows, columns)` and dtype `numpy.float64`. It is present only with `return_images=True`; a full 2048 by 2048 detector at 121 depths needs about 4 GiB. `timings` lists read, compute, and write seconds for each stripe.

## Write depth files

Pass `output_base` to write one HDF5 file per depth and a summary text file. It is a filename prefix, not a directory.

```python
result = reconstructor.reconstruct("point_1.h5", output_base="run/point_1_")
```

This writes `run/point_1_0.h5`, `run/point_1_1.h5`, one file for each further depth, and `run/point_1_summary.txt`. The directory is created when needed, and existing files with those names are overwritten. Each file copies the source metadata except the image data and wire positions.

`output_pixel_type` selects the stored dtype. By default, file output keeps the input dtype when it has a code, `wire_edge="both"` selects `numpy.int32`, and other dtypes fall back to `numpy.float64`. HDF5 performs the float-to-integer conversion, including saturation. When `norm_exponent` is set and an integer type is written, the written images are multiplied by the factor stored in `entry1/microDiffraction/norm_rescale`. The `images` and `depth_intensity` fields keep the unscaled values.

## Reconstruct arrays

{meth}`~lauelab.reconstruct.Reconstructor.reconstruct_array` accepts frames and wire positions that are already aligned:

- `images` with shape `(N, rows, columns)`. `numpy.uint16` is used as is. Any other numeric dtype is converted to `numpy.float64`.
- `wire_xyz` with shape `(N + 1, 3)` in the acquisition coordinate system. No file-format offset is applied.
- `positioner`: `"none"`, `"pm500"`, or `"alio"`.
- `scale` with shape `(N,)`: dimensionless per-frame factors, the array equivalent of `normalization`.
- `intensity_map` with shape `(rows, columns)`, defaulting to the first frame.

The default {class}`~lauelab.reconstruct.ImageGeometry` describes an unbinned full frame whose detector size equals the array size. Pass an explicit `ImageGeometry` for a binned image or a detector ROI. Its `start` and `group` are zero-based unbinned pixels. The array path writes no files and always returns `numpy.float64` images, so `normalization` and `output_pixel_type` do not apply.

## How a point file is read

A 34-ID-E multi-image file stores a bookkeeping frame at slice 0, which is skipped. Slice 1 is both the intensity map and the first scan frame. The last stored slice is never differenced. An `N`-slice file therefore reconstructs scan frames from slices 1 to `N - 2`, and needs at least 5 slices.

Stored wire vectors include acquisition bookkeeping entries. Scan frame `f` pairs with stored wire entries `f + 2` and `f + 3`. This offset applies to file input only.

The positioner correction comes from the file's `file_time` attribute. Files before May 2006 get no correction, files before October 2009 get the PM500 correction, and later files get the Alio identity correction. Matching the executable, a `file_time` written with the ISO `T` separator is not parsed and receives no correction.

`normalization` reads `entry1/<tag>` and divides by 102 for `mA` and 88100 for `cnt3`. A missing or short vector raises `InputError`. The executable silently skips normalization in that case, so a file that reconstructs with the executable can still fail here.

## Control memory and threads

The kernel processes rows in stripes. With `rows_per_stripe=None`, the stripe is at most 256 rows and is reduced until the stripe buffers fit in `memory_limit_mb`:

```{math}
2 \times \text{input bytes} + 2 \times \text{output bytes} \le \text{memory\_limit\_mb} \times 2^{20}
```

Input stripes use 2 bytes per pixel per frame for `numpy.uint16` and 8 bytes for every other dtype. The limit excludes retained images and HDF5 library buffers. The result does not depend on the stripe size or the thread count.

`num_threads` sets the OpenMP threads for each call. The default estimates physical cores from Linux SMT topology and otherwise uses `os.cpu_count()`.

## Reconstruct many points

{func}`~lauelab.reconstruct.reconstruct_points` processes point files in worker processes. Each worker creates one reconstructor and uses `threads_per_worker` OpenMP threads. Workers start with the spawn method so they do not inherit an initialized OpenMP runtime.

```python
from lauelab.reconstruct import reconstruct_points

paths = ["point_1.h5", "point_2.h5", "point_3.h5"]
results = reconstruct_points(
    paths,
    "run",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    depth_range=(-25.0, 25.0),
    workers=2,
    threads_per_worker=8,
)
failed = [path for path, result in zip(paths, results) if not result.success]
```

Output files use `<output_dir>/<stem>_` as their prefix. Pass reconstruction options as keyword arguments, and set threads with `threads_per_worker` rather than `num_threads`. Keep `workers × threads_per_worker` within the physical core count; oversubscription slows every worker.

An expected input, memory, or I/O failure in one point returns `success=False` for that point and does not stop the batch. Check every result.

## Handle failures

Invalid arguments, geometry, file metadata, and array shapes raise `InputError` before any work starts. A native allocation failure during setup raises `MemoryError`. After the first stripe starts, an expected failure returns a result with `success=False`, the message in `error`, and `last_completed_stripe` marking the last stripe fully written to every output file. See [Error handling](error-handling.md).

## Cross-check with the executable

{func}`~lauelab.reconstruct.reconstruct` runs the `reconstructN_cpu` program in a subprocess and returns the same result type. The in-process path reproduces its output bit for bit on the regression references. Use the executable as an independent check when validating a new acquisition configuration.
