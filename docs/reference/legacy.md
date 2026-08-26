# Legacy API

The legacy API invokes command-line executables and writes intermediate files.
New integrations should use `index_frame` or `Indexer` from the
{doc}`processing` API.

## Subprocess Pipeline

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autofunction:: lauego
```

`index` is a backward-compatible alias for `lauego`.

## Legacy Result

```{py:class} laueanalysis.indexing.index.IndexingResult(success, output_files, n_peaks_found, n_indexed, n_patterns_found, indexing_data, step_data, xml_file, log, error=None, command_history=[])

Result returned by the legacy subprocess indexing interface.

The result contains the overall status, generated output paths, peak and
pattern counts, optional parsed legacy data, the XML path, logs, an optional
error message, and the command history. New code should use `FrameResult`.
```
