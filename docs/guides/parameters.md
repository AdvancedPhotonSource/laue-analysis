# Parameters

{class}`~laueanalysis.indexing.PeakParams` controls peak detection and fitting. {class}`~laueanalysis.indexing.IndexParams` controls orientation indexing. Both are immutable dataclasses that the `Indexer` validates during construction.

## Start with the defaults

Use default values for an initial API check, but do not assume that they are appropriate for every detector, exposure, or material.

```python
from laueanalysis.indexing import Indexer, IndexParams, PeakParams

indexer = Indexer(
    "geometry.xml",
    "crystal.xml",
    peak_params=PeakParams(),
    index_params=IndexParams(),
)
```

Record the complete parameter objects with analysis output. A result alone does not contain every setting needed to reproduce processing.

## Peak-search parameters

| Parameter | Default | Units | Effect and constraint |
|---|---:|---|---|
| `boxsize` | `5` | px | Half-width of the square fitting region. Must be positive. |
| `max_rfactor` | `2.0` | dimensionless | Maximum accepted fit residual factor. Must be positive. |
| `min_size` | `3` | px | Minimum accepted peak size. Must be positive. |
| `min_separation` | `10` | px | Minimum separation between accepted peaks. Must be positive. |
| `threshold` | `100.0` | detector counts | Absolute detection threshold. Use `None` for an automatically derived threshold. |
| `threshold_ratio` | `4.0` | dimensionless | Scale applied to the frame standard deviation for automatic thresholding. |
| `peak_shape` | `"Lorentzian"` | none | Fit model. Values beginning with `L` or `G`, case-insensitively, select Lorentzian or Gaussian. |
| `max_peaks` | `50` | peaks | Maximum number of returned peaks. Must be positive. |
| `smooth` | `False` | none | Applies native image smoothing before detection and fitting. |

When `threshold` is not `None`, `threshold_ratio` does not determine the threshold. When `threshold` is `None`, the native stage calculates the threshold from frame statistics and `threshold_ratio`.

## Indexing parameters

| Parameter | Default | Units | Effect and constraint |
|---|---:|---|---|
| `kev_max_calc` | `30.0` | keV | Maximum energy used to calculate candidate reflections. Must be positive. |
| `kev_max_test` | `35.0` | keV | Maximum energy used when testing candidate reflections. Must be positive. |
| `angle_tolerance_deg` | `0.12` | deg | Angular matching tolerance. Must be positive. |
| `cone_deg` | `72.0` | deg | Search-cone angle. Must be positive. |
| `hkl_prefer` | `(0, 0, 1)` | Miller indices | Preferred direction. Must contain exactly three integers. |
| `max_data` | `250` | peaks | Maximum detected peaks supplied to orientation indexing. Must be at least two. |

These fields configure the native orientation search. Their scientifically appropriate values depend on the experiment and crystal. This guide does not prescribe universal tuning values.

## Derive a configuration

Use {func}`dataclasses.replace` instead of mutating a parameter object:

```python
from dataclasses import replace

base_peaks = PeakParams()
automatic_threshold = replace(
    base_peaks,
    threshold=None,
    threshold_ratio=4.0,
    max_peaks=200,
)
```

Construct a new indexer or use `Indexer.replace()` to validate changed settings:

```python
updated = indexer.replace(peak_params=automatic_threshold)
```

`Indexer.replace()` constructs independent native geometry and crystal state for the new instance.

## Parameter interactions

Several interactions follow directly from processing behavior:

- `threshold` selects absolute or automatic thresholding.
- `threshold_ratio` affects automatic thresholding only.
- `max_peaks` limits the output of peak search.
- `max_data` limits how many detected peaks enter orientation indexing.
- Orientation indexing runs only when a crystal is present and at least two peaks were detected.
- `start`, `group`, and `depth` affect pixel-to-q conversion rather than peak fitting.
- A mask changes which frame pixels peak search can use.

Changes to peak acceptance can change the scattering vectors available to indexing. Compare the intermediate peak count and fit fields before attributing a changed pattern result only to `IndexParams`.

## A conservative tuning workflow

1. Save the original frame, geometry, crystal description, and parameter objects.
2. Run peak search with a fixed configuration.
3. Inspect `threshold_used`, `n_peaks`, fitted positions, widths, residuals, and masked regions.
4. Change one peak-search parameter at a time.
5. Hold the accepted peak set fixed before comparing indexing parameters.
6. Compare pattern count, assignments, and angular errors across runs.
7. Record the chosen values and the reason for each change.

This process isolates parameter effects. Numerical thresholds and acceptance criteria still require experiment-specific review.

## Performance and reproducibility

Do not infer speed from `elapsed_seconds` alone. It sums recorded peak-search and orientation-indexing time but excludes pixel-to-q conversion and Python setup.

For a reproducible comparison, record:

- Package version or Git commit
- Frame identifier and checksum when possible
- Geometry and crystal file versions
- Detector selection
- `start`, `group`, and `depth`
- Mask identity
- Complete `PeakParams` and `IndexParams`
- Hardware and process configuration for timing comparisons

## Invalid configurations

The `Indexer` raises {class}`~laueanalysis.indexing.InputError` for invalid parameter ranges, unsupported peak models, unsupported detection binning, malformed `hkl_prefer`, and `max_data` below two. See [Configuration](../reference/configuration.md) for exact field definitions.
