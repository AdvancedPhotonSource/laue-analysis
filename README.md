# laue-analysis

Python and native tools for Laue diffraction analysis at APS beamline 34-ID-E. The package currently supports indexing and wire-scan reconstruction.

> [!NOTE]
> `laueanalysis` is alpha software under active development. Public behavior can change before a stable release.

Read the [documentation](https://advancedphotonsource.github.io/laue-analysis/) for installation requirements, indexing workflows, concepts, and API details.

## Install

The current source build targets Linux and requires Python 3.11 or later, GCC, `make`, HDF5, and GSL.

```bash
git clone https://github.com/AdvancedPhotonSource/laue-analysis.git
cd laue-analysis
python -m pip install .
```

## Index a frame

```python
from laueanalysis.indexing import index_frame

result = index_frame(
    "frame.h5",
    geometry="geometry.xml",
    crystal="crystal.xml",
)

print(result.n_peaks, result.n_indexed, result.n_patterns)
```

Use `Indexer` to reuse one geometry and crystal configuration across frames. The subprocess-based `index()` and `lauego()` functions remain available for existing LaueGo workflows.

## Test

```bash
python -m pytest
```
