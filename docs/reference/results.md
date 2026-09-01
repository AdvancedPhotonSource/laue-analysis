# Results

`FrameResult` owns Python copies of native indexing output. A result with no
identified patterns is valid and can still contain detected peaks and frame
statistics.

## Frame Result

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autoclass:: FrameResult(peaks, patterns, threshold_used, total_sum, sum_above_threshold, num_above_threshold, peaksearch_seconds, indexing_seconds, threshold_ratio=4.0, peak_minwidth=0.0, peak_maxwidth=0.0, peak_max_cent_to_fit=0.0, peak_boxsize=0, metadata={}, input_image=None, image_shape=(0, 0), start=(0, 0), group=(1, 1), depth=None, image=None)
   :members: indexed, n_peaks, n_indexed, n_patterns, elapsed_seconds, indexed_peak_indices, unindexed_peak_indices, write_xml
```

## Indexed Pattern

See the [results guide](../guides/results.md) for the reciprocal-basis convention used by `Pattern.reciprocal`.

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autoclass:: Pattern
   :members: n_indexed
```
