# laueanalysis

`laueanalysis` provides Python interfaces for processing Laue diffraction data at APS beamline 34-ID-E. It indexes crystal orientations, predicts detector reflections, and prepares maps, pole figures, detector overlays, and tables. Visualization functions accept current indexing results or LaueGo XML.

```{note}
`laueanalysis` is alpha software under active development. The documented indexing and visualization APIs are the supported interfaces, but behavior can change before a stable release. The current source build targets Linux and requires native C build tools and scientific libraries.
```

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
concepts/index
guides/index
reference/index
lauego/index
development/index
```

## Start here

Install the package from source, then follow one frame through indexing, visualization, and optional reflection simulation.

- [Install laueanalysis](installation.md)
- [Index your first frame](quickstart.md)
- [Prepare maps, figures, and tables](guides/visualization.md)
- [Simulate reflections for an indexed orientation](guides/simulation.md)

The in-process API accepts a two-dimensional `numpy.uint16` frame or a supported 34-ID-E HDF5 file. It returns a {class}`~laueanalysis.indexing.FrameResult` containing detected peaks, scattering vectors, candidate crystal patterns, timing, and provenance.

## Understand indexing

The [indexing pipeline](concepts/indexing-pipeline.md) explains the three processing stages and the data passed between them. The [algorithm pages](concepts/algorithms/index.md) provide more detail without interrupting the first-use path.

## Configure a workflow

Use the task-oriented guides when you need to select a detector, define a crystal, prepare frame data, tune parameters, or process a sequence.

- [User guides](guides/index.md)
- [API reference](reference/index.md)

The API reference contains the exact supported signatures and result fields. The guides explain when to use them and which conventions affect the result.

## Existing code

The subprocess-based `index` and `lauego` functions remain available for existing workflows. New code should use {func}`~laueanalysis.indexing.index_frame` or {class}`~laueanalysis.indexing.Indexer`. See [LaueGo interfaces](lauego/index.md) for the behavioral differences.
