# Quickstart

This example processes one synthetic detector frame. You need a 34-ID-E geometry file that describes the detector and a crystal XML file that describes the material.

## Prepare the inputs

An in-memory frame must be a two-dimensional `numpy.uint16` array. Its shape is `(ny, nx)`, while peak coordinates use `(x, y)` order.

The following frame contains three synthetic bright regions. It demonstrates the API, not a physical diffraction simulation.

```python
from pathlib import Path

import numpy as np

geometry_file = Path("geometry.xml")
crystal_file = Path("crystal.xml")

frame = np.zeros((2048, 2048), dtype=np.uint16)
frame[510:517, 720:727] = 800
frame[980:987, 1310:1317] = 1200
frame[1450:1457, 430:437] = 1000
```

The geometry must match the frame dimensions and detector. The crystal description must match the material if you want orientation indexing.

## Index one frame

Use {func}`~laueanalysis.indexing.index_frame` when you have one frame or when convenience matters more than reusing parsed configuration.

```python
from laueanalysis.indexing import index_frame

result = index_frame(
    frame,
    geometry=geometry_file,
    crystal=crystal_file,
)

print(f"detected peaks: {result.n_peaks}")
print(f"indexed assignments: {result.n_indexed}")
print(f"patterns: {result.n_patterns}")
```

`index_frame` retains the input image in `result.image` by default. Pass `keep_image=False` if the result does not need it.

A valid call can return no crystal patterns. In that case, `result.indexed` is `False`, while peak data and frame statistics remain available.

## Reuse an indexer

Construct {class}`~laueanalysis.indexing.Indexer` once when frames share geometry, crystal, detector selection, and processing parameters. The object retains parsed native configuration between calls.

```python
from laueanalysis.indexing import Indexer

indexer = Indexer(geometry_file, crystal_file)

first = indexer.index(frame)
second = indexer.index(frame.copy(), keep_image=False)
```

Use `indexer.index_many(frames)` for a sequential batch. Results preserve input order, and batch processing does not retain images by default.

```python
results = indexer.index_many([frame, frame.copy()])
assert [item.image_shape for item in results] == [(2048, 2048), (2048, 2048)]
assert all(item.image is None for item in results)
```

## Inspect the result

`result.peaks` is a structured NumPy array. Each row contains a fitted `(x, y)` position, fit measurements, and a three-component `qhat` vector.

```python
xy = np.column_stack((result.peaks["fit_x"], result.peaks["fit_y"]))
qhat = result.peaks["qhat"]

print(xy.shape)    # (n_peaks, 2)
print(qhat.shape)  # (n_peaks, 3)
```

Each item in `result.patterns` describes one indexed orientation. Its `pk_index` array contains zero-based indices into `result.peaks`.

```python
for pattern in result.patterns:
    assigned_peaks = result.peaks[pattern.pk_index]
    print(pattern.euler_deg, assigned_peaks.shape[0])
```

See [Results](guides/results.md) before interpreting the complete peak and pattern schemas.

## Write XML

XML output is explicit in the in-process API. Writing XML does not create peak, pixel-to-q, or indexing text files.

```python
result.write_xml("indexed-frame.xml")
```

This format exists for compatibility with the established 34-ID-E XML workflow. Use `FrameResult` directly for new Python analysis.

## Handle errors

Catch invalid user or data input separately from native numerical failures:

```python
from laueanalysis.indexing import IndexingError, InputError

try:
    result = indexer.index(frame)
except InputError as error:
    print(f"Check the frame and configuration: {error}")
except IndexingError as error:
    print(f"The native indexing stage failed: {error}")
except MemoryError as error:
    print(f"Indexing could not allocate memory: {error}")
```

Do not retry unchanged invalid input. See [Error handling](guides/error-handling.md) for batch strategies and useful diagnostic context.

## Next steps

- [Select and inspect detector geometry](guides/geometry.md)
- [Load or construct a crystal](guides/crystals.md)
- [Prepare frame data and metadata](guides/frame-input.md)
- [Configure processing parameters](guides/parameters.md)
- [Interpret results](guides/results.md)
- [Process batches](guides/batch-indexing.md)
- [Read the API reference](reference/index.md)
