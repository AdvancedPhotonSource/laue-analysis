# Results

{class}`~laueanalysis.indexing.FrameResult` contains the output and provenance for one processed frame. Native allocations are released before the object is returned. Peak and pattern data are copied into NumPy arrays.

## Frame-level status

Use these properties for a summary:

| Property | Meaning |
|---|---|
| `indexed` | `True` when at least one pattern was identified |
| `n_peaks` | Number of detected peaks |
| `n_indexed` | Total assignments across all patterns |
| `n_patterns` | Number of identified patterns |
| `elapsed_seconds` | Sum of recorded peak-search and orientation-indexing times |

`n_indexed` counts assignments, not necessarily unique peaks. Use `indexed_peak_indices` when you need the unique set of peaks assigned to any pattern.

## Peaks

`result.peaks` is a one-dimensional structured array with one row per detected peak.

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `fit_x` | scalar | frame px | Zero-based fitted x coordinate |
| `fit_y` | scalar | frame px | Zero-based fitted y coordinate |
| `intens` | scalar | detector counts | Fitted peak intensity |
| `integral` | scalar | implementation-defined | Integrated fitted intensity |
| `hwhm_x` | scalar | px | Fitted half-width along the x fit axis |
| `hwhm_y` | scalar | px | Fitted half-width along the y fit axis |
| `tilt` | scalar | deg | Fitted peak tilt |
| `chisq` | scalar | dimensionless | Normalized fit residual reported by peak search |
| `background` | scalar | detector counts | Fitted background level |
| `qhat` | `(3,)` | dimensionless | Unit scattering vector in the 34-ID-E laboratory convention |

Extract ordinary arrays by field name:

```python
import numpy as np

xy = np.column_stack((result.peaks["fit_x"], result.peaks["fit_y"]))
qhat = result.peaks["qhat"]
intensity = result.peaks["intens"]
```

A frame coordinate `(x, y)` corresponds to NumPy access `image[y, x]`. Fitted coordinates are floating-point values and can lie between pixel centers.

## Patterns

Each {class}`~laueanalysis.indexing.Pattern` represents one orientation returned by the native indexer.

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `euler_deg` | `(3,)` | deg | Euler-angle representation of the orientation |
| `rotation` | `(3, 3)` | dimensionless | Orientation rotation matrix |
| `recip` | `(3, 3)` | reciprocal length | Reciprocal-lattice matrix |
| `goodness` | scalar | implementation-defined | Native pattern goodness score |
| `rms_error_deg` | scalar | deg | Root-mean-square angular indexing error |
| `hkl` | `(n, 3)` | Miller indices | Assigned reflection indices |
| `pk_index` | `(n,)` | array indices | Zero-based indices into `result.peaks` |
| `err_deg` | `(n,)` | deg | Angular error for each assignment |
| `energy_kev` | `(n,)` | keV | Photon energy for each assignment |
| `pred_intens` | `(n,)` | implementation-defined | Predicted intensity for each assignment |

`pattern.n_indexed` is `len(pattern.pk_index)`. Rows in `hkl`, `err_deg`, `energy_kev`, and `pred_intens` correspond to the same assignments.

The precise Euler-angle convention and reciprocal-matrix basis require 34-ID-E domain verification. Prefer the full rotation matrix when exchanging orientations with software that uses a documented matrix convention.

## Indexed and unindexed peaks

Pattern indices refer to `result.peaks`:

```python
for pattern in result.patterns:
    measured = result.peaks[pattern.pk_index]
    assert len(measured) == pattern.n_indexed
```

`result.indexed_peak_indices` returns sorted, unique indices assigned to at least one pattern. `result.unindexed_peak_indices` returns the complement over all detected peaks.

```python
indexed = result.peaks[result.indexed_peak_indices]
unindexed = result.peaks[result.unindexed_peak_indices]
```

These arrays are useful when a peak can appear in more than one pattern or when `n_indexed` must not be treated as a unique count.

## Frame statistics and timing

The result records:

- `threshold_used`
- `total_sum`
- `sum_above_threshold`
- `num_above_threshold`
- `peaksearch_seconds`
- `indexing_seconds`

`indexing_seconds` measures the orientation-indexing section. It includes the negligible branch when no crystal is supplied or too few peaks are present. Pixel-to-q conversion is not included in either timing field. `elapsed_seconds` therefore does not measure complete call latency.

`metadata` contains supplied values and recognized HDF5 provenance. `input_image` is the HDF5 path for file input and `None` for an in-memory array. `image_shape`, `start`, `group`, and `depth` record the frame geometry used by the call.

## Retained images

`index_frame()` and `Indexer.index()` retain a contiguous `uint16` image by default. Set `keep_image=False` when downstream work only needs processed data.

`Indexer.index_many()` does not retain images by default. This avoids keeping one detector-sized array per result.

The `FrameResult` dataclass is frozen, but its arrays and metadata mapping can remain mutable. Copy them before modification when the unchanged result must remain available.

## Write XML

Write one result in the established Laue XML format:

```python
result.write_xml("indexed-frame.xml")
```

Write several results in iteration order:

```python
indexer.write_many_xml(results, "indexed-scan.xml")
```

Existing destination files are replaced. These methods serialize a snapshot captured when each result was created. Later mutations to result arrays do not update that snapshot.

## Compatibility conversion

`result.to_step()` returns a deep copy of the internal LaueGo XML model. This method is a compatibility bridge for code that must interact with that representation. It is not the preferred analysis model, and the serialization classes are not part of the curated public API.

## Empty results

No detected peaks is a valid result. The structured peak array is empty and no patterns are returned.

Detected peaks with no identified orientation is also valid. `n_peaks` is positive, `patterns` is empty, and `indexed` is `False`.

Exceptions represent invalid input, allocation failure, or failure inside a native stage. Do not convert a scientifically empty result into an exception unless your application requires that policy.
