# Error handling

The in-process API separates invalid input from native processing failures. Catch the narrowest exception that your application can handle correctly.

## Exception hierarchy

{class}`~laueanalysis.indexing.LaueError` is the package base class.

- {class}`~laueanalysis.indexing.InputError` inherits from both `LaueError` and `ValueError`.
- {class}`~laueanalysis.indexing.IndexingError` inherits from both `LaueError` and `RuntimeError`.
- Native allocation failures use Python's built-in {class}`MemoryError`.

`ValueError`, XML parse errors, `OSError`, and `KeyError` can also occur while loading geometry, crystal, or HDF5 input. See the relevant API reference for each loader.

## Invalid input

`InputError` reports invalid processing configuration, including:

- Peak or indexing parameters outside supported ranges
- An unknown detector identifier or inactive detector slot
- A frame that is not a two-dimensional `uint16` array
- Invalid `start`, `group`, or `depth`
- A frame region outside detector bounds
- A mask shape that does not match the frame
- An HDF5 detector identifier that does not match the selected geometry

Fix the input before retrying. The exception message names the failed check and includes the received value when useful.

```python
import numpy as np

from laueanalysis.indexing import InputError

bad_frame = np.zeros((128, 128), dtype=np.float32)

try:
    indexer.index(bad_frame)
except InputError as error:
    print(error)
```

## Memory failure

A native stage raises `MemoryError` when it cannot allocate required storage. The message identifies the stage and includes its diagnostic.

Do not assume that immediate retry will succeed. Release unneeded arrays and results, reduce concurrent work, or move the workload to a process with sufficient memory before retrying.

## Native indexing failure

`IndexingError` reports numerical or internal failures in a native processing stage. Its message begins with the stage name, such as `pixel-to-q conversion failed` or `orientation indexing failed`.

Preserve the complete message. It can distinguish a geometry conversion problem from an orientation-indexing problem without exposing native status values as a public API.

## Batch strategy

Use `index_many()` when the batch should stop on its first failure. Use an explicit loop when each frame needs an independent status:

```python
from laueanalysis.indexing import IndexingError, InputError

results = {}
failures = {}

for frame_id, frame in frames.items():
    try:
        results[frame_id] = indexer.index(frame, keep_image=False)
    except (InputError, IndexingError) as error:
        failures[frame_id] = str(error)
```

Decide separately whether to catch `MemoryError`, `OSError`, and `KeyError`. Continuing after those failures may hide a system-wide resource problem or a repeated file-layout error.

## Diagnostic context

Record enough context to reproduce the call:

- Package version or Git commit
- Input identifier, shape, and dtype
- Geometry and crystal identifiers
- Detector slot and detector ID
- `start`, `group`, and `depth`
- Peak and indexing parameters
- Mask identity or generation method
- Exception type and complete message

Do not log full frame arrays. Remove user names, sample names, local paths, and other sensitive acquisition metadata before sharing a report.

## Retry limits

Do not retry unchanged input after `InputError`. A numerical `IndexingError` also needs investigation before retry because the same inputs can reproduce the same failure.

A transient file-access failure can be retried only when the application can identify a transient cause. `laueanalysis` does not classify I/O failures as transient.

No peaks or no patterns is not a failure and does not raise an exception. Apply a separate scientific acceptance policy to those results.
