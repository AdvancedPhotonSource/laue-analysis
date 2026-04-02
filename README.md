# laue-analysis

Python package and C binaries for Laue diffraction data analysis at beamline 34IDE, including indexing and wire scan reconstruction.

## Package Structure

```
src/laueanalysis/
├── indexing/
│   ├── index.py             # Functional indexing API
│   ├── parsers.py           # Output file parsers
│   ├── xml_utils.py         # XML generation utilities
│   ├── xmlWriter.py         # XML batch writer
│   ├── mpi_runner.py        # MPI distributed execution
│   ├── lau_dataclasses/     # Data models (step, detector, pattern, etc.)
│   ├── bin/                 # Compiled C executables (euler, peaksearch, pix2qs)
│   └── src/                 # C source trees
├── reconstruct/
│   ├── reconstruct.py       # Reconstruction API (CPU + GPU)
│   ├── bin/                 # Compiled C executable (reconstructN_cpu)
│   └── source/              # C source code
└── __init__.py
```

## Installation

Prerequisites:
- Python >= 3.12
- System: `make`, `gcc`, `h5cc`, GNU Scientific Library (GSL), HDF5 dev libraries
- Linux only (conda packaging)

```bash
git clone https://github.com/AdvancedPhotonSource/laue-analysis.git
cd laue-analysis
python3 -m pip install .
```

The build step compiles C binaries into `indexing/bin/` and `reconstruct/bin/`.

## Usage

### Indexing

```python
from laueanalysis.indexing import index

result = index(
    input_image='path/to/image.h5',
    output_dir='path/to/output',
    geo_file='path/to/geometry.xml',
    crystal_file='path/to/crystal.xml',
)

print(result.n_peaks_found, result.n_indexed)
```

The `index()` function runs the full pipeline (peak search, pixel-to-q conversion, indexing) and returns an `IndexingResult` with output files, statistics, and parsed data. See `index.py` for all available parameters.

### Wire Scan Reconstruction

```python
from laueanalysis.reconstruct import reconstruct

result = reconstruct(...)
```

See test files for detailed usage examples.

## Output Layout

Indexing outputs are written under `output_dir`:

```
xml/                        # Per-image indexed XML files
peaks/peaks_<name>.txt      # Peak search results
p2q/p2q_<name>.txt          # Pixel-to-Q space results
index/index_<name>.txt      # Indexing results
error/                      # Error logs
```

## Testing

```bash
python -m pytest tests/ -v
```
