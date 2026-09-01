# Frame input

The indexing API accepts either an in-memory NumPy array or a path to the supported 34-ID-E HDF5 layout. Both forms produce the same `FrameResult` model.

## NumPy frames

An array must be two-dimensional and have `numpy.uint16` dtype:

```python
import numpy as np

frame = np.zeros((2048, 2048), dtype=np.uint16)
```

The shape is `(ny, nx)`. NumPy accesses a pixel as `frame[y, x]`, while fitted peak coordinates are reported as `(x, y)`.

The indexer makes a C-contiguous copy only when the supplied array is not already contiguous. It does not convert another dtype to `uint16`. Convert intentionally before the call so that clipping or rounding is visible in your application.

## HDF5 frames

Pass a path to read a frame from `entry1/data/data`:

```python
result = indexer.index("frame.h5")
```

This is support for a specific acquisition layout, not arbitrary HDF5. A missing image dataset raises `KeyError`, and a file-open failure raises `OSError`.

When present, the loader reads processing values from:

| Value | HDF5 dataset |
|---|---|
| `start_x` | `entry1/detector/startx` |
| `start_y` | `entry1/detector/starty` |
| `group_x` | `entry1/detector/binx` |
| `group_y` | `entry1/detector/biny` |

HDF5 `start` and `group` values take precedence over values passed to `index()`. The file reader currently does not load a depth value, so an explicit `depth` remains in effect.

## Region and grouping

For an in-memory frame, `start=(x, y)` identifies the frame's zero-based origin on the full detector. `group=(x, y)` gives the number of detector pixels represented by one frame pixel along each axis.

```python
region = np.zeros((512, 512), dtype=np.uint16)
result = indexer.index(
    region,
    start=(100, 200),
    group=(2, 2),
)
```

Both `start` values must be nonnegative integers. Both `group` values must be positive integers. The transformed frame extent must remain within the selected detector.

The pixel-to-q conversion uses the center of each grouped detector region. See [Geometry](geometry.md) for the mapping.

`depth` is an optional finite sample depth in micrometres passed to geometry conversion. The physical sign convention requires 34-ID-E domain verification and is not inferred by this documentation.

## Masks

A mask must have the same shape as the frame:

```python
mask = np.zeros(frame.shape, dtype=np.uint8)
mask[100:120, 300:340] = 1

result = indexer.index(frame, mask=mask)
```

The API converts the mask to contiguous `uint8`. Zero pixels remain available to peak search. Nonzero pixels are masked.

## Metadata

Pass {class}`~laueanalysis.indexing.FrameMetadata` or a mapping when the result needs experiment provenance:

```python
from laueanalysis.indexing import FrameMetadata

metadata = FrameMetadata(
    sample_name="synthetic nickel",
    scan_number=42,
    detector_id="PE1621 723-3335",
    exposure_seconds=0.25,
)

result = indexer.index(frame, metadata=metadata)
```

For HDF5 input, the loader reads recognized metadata fields when their datasets exist. Explicit metadata values override values loaded from the file.

The HDF5 detector identifier is also checked against the selected geometry detector. A mismatch raises `InputError`. For an in-memory frame, `metadata.detector_id` is provenance only and is not used for this validation.

See the {class}`~laueanalysis.indexing.FrameMetadata` reference for the complete field list.

## Image ownership

`index_frame()` and `Indexer.index()` retain the contiguous image in `result.image` by default. Pass `keep_image=False` when later processing only needs peaks and patterns.

```python
result = indexer.index(frame, keep_image=False)
assert result.image is None
```

`Indexer.index_many()` defaults to `keep_images=False` to limit batch memory use. Pass `keep_images=True` only when every result needs its source image.

The result's peak and pattern arrays are Python-owned copies. Native result storage is released before the method returns.

## Common failures

`InputError` reports:

- A frame with the wrong number of dimensions or dtype
- Invalid `start`, `group`, or `depth`
- A frame region outside the selected detector
- A mask with a different shape
- An HDF5 detector identifier that does not match the selected detector

A no-peak result is not an input failure. It returns an empty peak array and no patterns.
