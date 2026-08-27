# Visualization

The public visualization API contains normalized data, prepared view data, Plotly renderers, selection parsing, and typed tables. Preparation and plot functions accept either `ResultSet` or `VisualizationDataset`.

See [Visualization data](../guides/visualization.md) for workflows, coordinate conventions, and examples.

## Data and selection

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autoclass:: DataScope
   :members: pattern_mask

.. autoclass:: ResultSet
   :members: from_indexer, to_visualization

.. autoclass:: VisualizationDataset
   :members: n_frames, n_patterns, n_assignments, pattern_ids

.. autofunction:: load_visualization_xml
```

## Prepared views

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autoclass:: Axis

.. autoclass:: ScalarColor

.. autoclass:: MapData

.. autoclass:: PoleFigureData

.. autoclass:: DetectorPatternData

.. autoclass:: DetectorViewData

.. autofunction:: prepare_map

.. autofunction:: prepare_pole_figure

.. autofunction:: prepare_detector_view
```

Prepared models copy their arrays and mark them read-only. Their frame and pattern identity fields remain aligned with the prepared rows.

## Plotly rendering

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autofunction:: plot_map

.. autofunction:: plot_pole_figure

.. autofunction:: plot_detector_view
```

Each renderer accepts a prepared model or normalized input. `trace_update` changes traces by the roles documented in the [visualization guide](../guides/visualization.md). `layout_update` applies after package defaults.

## Plotly selection

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autoclass:: PlotlySelection

.. autofunction:: selection_from_plotly
```

## Tables

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autoclass:: Table
   :members: to_dataframe

.. autofunction:: peak_table

.. autofunction:: pattern_table

.. autofunction:: assignment_table

.. autofunction:: indexed_peak_table
```

## Built-in choices

`AXIS_OPTIONS`, `COLOR_MODES`, `SURFACE_PRESETS`, and `PALETTE_OPTIONS` are immutable tuples of {class}`~laueanalysis.visualization.Choice` objects. Applications can use them to construct controls without duplicating package option names.

```{eval-rst}
.. currentmodule:: laueanalysis.visualization

.. autoclass:: Choice

.. autodata:: AXIS_OPTIONS

.. autodata:: COLOR_MODES

.. autodata:: SURFACE_PRESETS

.. autodata:: PALETTE_OPTIONS
```

## Surface frames

```{eval-rst}
.. currentmodule:: laueanalysis.analysis

.. autoclass:: SurfaceFrame
   :members: from_vectors, aps_34ide
```
