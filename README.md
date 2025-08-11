# laue-analysis

A Python package and collection of C binaries for Laue diffraction data analysis at beamline 34IDE, including indexing and wire scan reconstruction capabilities.

## Package Structure

```
src/
└── laueanalysis/
    ├── indexing/                   # Laue indexing submodule
    │   ├── pyLaueGo.py            # Main orchestrator for data processing
    │   ├── mpi_runner.py          # MPI-based execution utilities
    │   ├── xmlWriter.py           # XML output writer
    │   ├── lau_dataclasses/       # Data models (atom, detector, HKL, pattern, etc.)
    │   ├── bin/                   # Compiled C executables (euler, peaksearch, pix2qs)
    │   └── src/                   # C source trees (euler, peaksearch, pixels2qs)
    ├── reconstruct/               # Wire scan reconstruction submodule
    │   ├── wirescan_interface.py  # Python interface for reconstruction C program
    │   ├── bin/                   # Compiled C executable (reconstructN_cpu)
    │   └── source/                # C source code for reconstruction
    │       └── recon_cpu/         # CPU-based reconstruction implementation
    └── __init__.py
```

- **Indexing C binaries** (`euler`, `peaksearch`, `pix2qs`) live under `src/laueanalysis/indexing/bin/` after installation.
- **Reconstruction C binary** (`reconstructN_cpu`) lives under `src/laueanalysis/reconstruct/bin/` after installation.
- **Source** under `src/laueanalysis/*/src/` is included for reference; users do not invoke it directly.

## Installation

Prerequisites:

- Python ≥ 3.12  
- System libraries and tools: `[make, gcc, h5cc]`, GNU Scientific Library (GSL), HDF5 development libraries
- Python dependencies: `numpy`, `pyyaml`, `h5py`, `mpi4py`
- NOTE: The conda packaging is compatible only with linux. Many of the underlying dependencies aren't available for other system. 

Install from source:

```bash
git clone https://github.com/yourusername/laue_indexing.git
cd laue_indexing
python3 -m pip install .
```

The `setup.py` build step will compile the C binaries into both `src/laueanalysis/indexing/bin/` and `src/laueanalysis/reconstruct/bin/`.  

## Configuration

Provide a YAML config file (see `tests/data/test_config.yaml` for an example).  Key entries:

- `filefolder`: Directory of input HDF5 files
- `geoFile`: Path to the geometry XML file
- `crystFile`: Path to the crystal-structure file
- `outputFolder`: Directory for generated output  
- Optional range filters: `scanPointStart`, `scanPointEnd`, `depthRangeStart`, `depthRangeEnd`

## Usage

### Laue Indexing

#### Command-line / MPI

Run with MPI to parallelize:

```bash
mpirun -np 32 python -m laueanalysis.indexing.pyLaueGo path/to/config.yml
```

#### Python API

```python
import yaml
from laueanalysis.indexing.pyLaueGo import PyLaueGo

with open('path/to/config.yml') as f:
    config = yaml.safe_load(f)

processor = PyLaueGo(config=config)
processor.run_on_process()  # single-process execution
```

### Wire Scan Reconstruction

The reconstruction module provides a Python interface to the `reconstructN_cpu` C program for depth-resolved wire scan analysis. See the test files for usage examples.

## Output Layout

### Indexing Results
Results are written under `outputFolder` as:

```
input_basename.xml          # Final processed XML
error_TIMESTAMP.log         # Error log
index/index_out_0_0.txt     # Index subprocess outputs
p2q/p2q_out_0_0.txt         # Pixels→Q space outputs
peaks/peaks_out_0_0.txt     # Peak-search outputs
```

### Reconstruction Results
Wire scan reconstruction outputs depth-resolved image files:

```
recon_prefix_depth_XXX.h5   # Reconstructed depth slices
recon_prefix_metadata.xml   # Reconstruction metadata
```

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```
