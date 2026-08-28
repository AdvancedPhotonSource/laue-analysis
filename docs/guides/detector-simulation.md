# Add simulation to a detector view

Detector-view simulation predicts missing reflection directions for selected indexed patterns. It keeps measured peaks, indexed assignments, and simulated reflections in separate prepared layers.

## Prepare the detector data

Pass an energy interval to {func}`~laueanalysis.visualization.prepare_detector_view`:

```python
from laueanalysis.visualization import prepare_detector_view

detector_data = prepare_detector_view(
    result_set,
    frame_id="scan-42-point-7",
    patterns="best",
    image=True,
    simulation_energy_range_kev=(6.0, 30.0),
)
```

The default value of `simulation_energy_range_kev` is `None`. The default performs no simulation and preserves the ordinary detector-view workflow.

Simulation requires the shared `Crystal` and geometry in the source data. {meth}`~laueanalysis.visualization.ResultSet.from_indexer` copies both references from an `Indexer`. LaueGo XML input may need explicit geometry and crystal context before it can simulate reflections.

The `patterns` argument controls both indexed and simulated layers:

| Value | Selected patterns |
|---|---|
| `"best"` | Lowest frame-local pattern rank |
| `"all"` | Every indexed pattern in the frame |
| `(0, 2)` | Explicit frame-local pattern ranks 0 and 2 |

Preparation calls `missing_from()` with each selected pattern's indexed HKLs. Positive indexed harmonics therefore remove the corresponding simulated direction.

## Understand frame coordinates

{func}`~laueanalysis.analysis.simulate_reflections` returns zero-based, unbinned full-detector `(x, y)` pixels. Detector-view preparation converts each point to the selected frame coordinates:

```text
frame_xy = (detector_xy - start - (group - 1) / 2) / group
```

`start` is the zero-based full-detector `(x, y)` origin. `group` contains the detector-pixel grouping factors `(group_x, group_y)`. The half-group term maps each frame pixel to the center of its full-detector pixel group.

The prepared `predicted_xy` values are zero-based frame `(x, y)` pixels. A NumPy image still uses `[y, x]` access and shape `(ny, nx)`. Preparation removes non-finite and off-frame simulated points.

## Reuse prepared simulation data

Each item in `detector_data.simulations` is a {class}`~laueanalysis.visualization.DetectorSimulationData` for one pattern. Its `hkl`, `predicted_xy`, `energy_kev`, and `relative_intensity` arrays are aligned, copied, and read-only.

Prepare once when several render operations use the same scientific result:

```python
from laueanalysis.visualization import plot_detector_view

figure_with_simulation = plot_detector_view(detector_data)
figure_without_simulation = plot_detector_view(
    detector_data,
    show_simulated=False,
)
```

`show_simulated` changes rendering only. Passing prepared data never runs the simulator again. Application-level persistence across processes or callbacks remains the application's responsibility.

An empty simulation layer is valid. The renderer does not create an empty trace.

## Render and customize simulated reflections

The detector renderer groups simulated reflections by pattern. It draws purple open triangles and uses the semantic trace role `"simulated"`:

```python
figure = plot_detector_view(
    detector_data,
    show_hkl_labels=True,
    trace_update={
        "simulated": {
            "marker": {"size": 14},
        },
    },
)
```

Hover data contains the frame ID, pattern rank, HKL, energy in keV, relative intensity, and frame detector position. `show_hkl_labels` controls labels for indexed and simulated reflections.

The complete detector-view role list is `"image"`, `"boundary"`, `"detected"`, `"indexed"`, and `"simulated"`. An unknown role in `trace_update` raises `ValueError`.

## Read reflection selections

A simulated Plotly point stores these `customdata` fields:

```text
[frame_id, pattern_index, None, h, k, l, energy_kev, relative_intensity]
```

The first three positions retain the existing frame, pattern, and peak identity layout. A simulated point has no measured peak index.

Use {func}`~laueanalysis.visualization.selection_from_plotly` with Plotly click or selection data:

```python
from laueanalysis.visualization import selection_from_plotly

selection = selection_from_plotly(event_data)
for reflection_id in selection.reflection_ids:
    frame_id, pattern_index, h, k, l = reflection_id
    print(frame_id, pattern_index, (h, k, l))
```

`reflection_ids` uses stable `(frame_id, pattern_index, h, k, l)` tuples. The parser removes duplicates in event order. Existing `frame_ids`, `pattern_ids`, and `peak_ids` remain unchanged for current traces.

## Propagate simulation failures

Preparation and rendering do not hide simulation errors. Missing crystal context raises `ValueError` only when simulation is requested. Private simulation failures propagate as `RuntimeError`.

Do not replace an error with an empty overlay. An empty simulation layer means that a valid simulation found no missing on-frame directions.

## Keep application integration outside the package

An application can prepare data in one callback and render or inspect it later. A future portal migration can use the same package-only boundary:

```python
prepared = prepare_detector_view(
    result_set,
    frame_id=frame_id,
    patterns="best",
    simulation_energy_range_kev=energy_range_kev,
)
figure = plot_detector_view(prepared, show_simulated=show_simulated)
selection = selection_from_plotly(event_data)
```

The package does not define Dash controls, callback caching, portal overlay models, or backend-selection settings. Those choices belong to the application.

See the [visualization API reference](../reference/visualization.md) for the prepared model and renderer signatures.
