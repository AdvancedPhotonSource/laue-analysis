# Legacy interfaces

The `index` and `lauego` names refer to the same subprocess-based compatibility function. It runs the `peaksearch`, `pix2qs`, and `euler` programs and writes intermediate text files beneath an output directory.

New code should use the in-process API. The legacy function remains documented so existing workflows can migrate without assuming identical behavior.

## Current interfaces

```python
from laueanalysis.indexing import index, lauego

assert index is lauego
```

The function returns an `IndexingResult`, documented in the [legacy API reference](../reference/legacy.md). It reports success through fields such as `success` and `error` and stores output paths, logs, command history, counts, parsed legacy structures, and an optional XML path.

The legacy pipeline can report overall success after a later stage fails or produces no output. Applications must inspect its returned fields and logs. This differs from the exception model used by `Indexer`.

## Recommended replacement

Use {func}`~laueanalysis.indexing.index_frame` for one frame:

```python
from laueanalysis.indexing import index_frame

result = index_frame(
    "frame.h5",
    geometry="geometry.xml",
    crystal="crystal.xml",
)
```

Use {class}`~laueanalysis.indexing.Indexer` when processing several frames with shared configuration.

The in-process path:

- Returns `FrameResult` instead of `IndexingResult`.
- Raises exceptions for invalid input and native failures.
- Keeps peak and pattern data in memory.
- Does not create intermediate text files.
- Writes XML only when you call an XML method.
- Has no subprocess timeout because it does not launch the three indexing programs.

## Parameter mapping

| Legacy argument | In-process replacement |
|---|---|
| `geo_file` | `geometry` for `index_frame`, or the first `Indexer` argument |
| `crystal_file` | `crystal` for `index_frame`, or the second `Indexer` argument |
| Peak arguments | Fields of `PeakParams` |
| Index arguments | Fields of `IndexParams` |
| `index_h`, `index_k`, `index_l` | `IndexParams.hkl_prefer` |
| `depth_override` | `depth` on `index_frame()` or `Indexer.index()` |
| `cosmic_filter` | `cosmic_filter` on `index_frame()` or `Indexer` |
| `generate_xml` | An explicit `FrameResult.write_xml()` call |
| `xml_output_file` | The path passed to `write_xml()` |
| `timeout` | No replacement in the in-process API |
| `mask_file` | Load or construct an array and pass it as `mask` |

The legacy `peak_shape` commonly uses `"L"` or `"G"`. `PeakParams.peak_shape` defaults to the clearer `"Lorentzian"` spelling but accepts either initial.

Legacy `threshold_ratio=-1` means that the command option is omitted. In the in-process API, `threshold=None` selects automatic thresholding, and `threshold_ratio` controls its scale. Review this setting rather than copying `-1` mechanically.

## Result mapping

| Legacy result | In-process result |
|---|---|
| `n_peaks_found` | `FrameResult.n_peaks` |
| `n_indexed` | `FrameResult.n_indexed` |
| `n_patterns_found` | `FrameResult.n_patterns` |
| `indexing_data` | `FrameResult.patterns` and their arrays |
| `step_data` | `FrameResult.to_step()` for compatibility only |
| `xml_file` | The path your application passes to `write_xml()` |
| `success` and `error` | Normal return or a raised exception |
| `output_files` | No intermediate-file equivalent |
| `log` and `command_history` | No subprocess equivalent |

## Migration procedure

1. Move peak-search values into `PeakParams`.
2. Move orientation values into `IndexParams`.
3. Construct one `Indexer` for each shared geometry, crystal, detector, and parameter configuration.
4. Replace legacy result-field access with `FrameResult` properties and arrays.
5. Catch `InputError`, `IndexingError`, and `MemoryError` at the application boundary.
6. Add explicit XML writes only where a downstream consumer requires them.
7. Remove assumptions about intermediate peak, pixel-to-q, and indexing files.
8. Compare migrated results against representative legacy output before changing scientific settings.

See the [legacy API reference](../reference/legacy.md) for the complete compatibility signature.
