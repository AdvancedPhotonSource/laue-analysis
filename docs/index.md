# laueanalysis

`laueanalysis` provides Python interfaces for processing and visualizing Laue diffraction data at APS beamline 34-ID-E. The indexing API finds peaks in a detector frame, maps their positions to scattering vectors, and identifies crystal orientations. The visualization API prepares maps, pole figures, detector overlays, and tables from indexing results or legacy XML.

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
legacy/index
development/index
```

## Start here

Install the package from source, then follow one frame through the in-process indexing API.

- [Install laueanalysis](installation.md)
- [Index your first frame](quickstart.md)
- [Prepare visualization data](guides/visualization.md)

The in-process API accepts a two-dimensional `numpy.uint16` frame or a supported 34-ID-E HDF5 file. It returns a {class}`~laueanalysis.indexing.FrameResult` containing detected peaks, scattering vectors, candidate crystal patterns, timing, and provenance.

## Understand indexing

The [indexing pipeline](concepts/indexing-pipeline.md) explains the three processing stages and the data passed between them. The [algorithm pages](concepts/algorithms/index.md) provide more detail without interrupting the first-use path.

## Configure a workflow

Use the task-oriented guides when you need to select a detector, define a crystal, prepare frame data, tune parameters, or process a sequence.

- [User guides](guides/index.md)
- [API reference](reference/index.md)

The API reference contains the exact supported signatures and result fields. The guides explain when to use them and which conventions affect the result.

## Existing code

The subprocess-based `index` and `lauego` functions remain available for existing workflows. New code should use {func}`~laueanalysis.indexing.index_frame` or {class}`~laueanalysis.indexing.Indexer`. See [legacy interfaces](legacy/index.md) for the behavioral differences.
