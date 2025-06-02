# laue-indexing

A Python package and collection of C binaries for indexing Laue diffraction data at beamline 34IDE.

## Package Structure

```
laueindexing/
├── pyLaueGo.py         # Main orchestrator for data processing
├── mpi_runner.py       # MPI-based execution utilities
├── xmlWriter.py        # XML output writer
├── lau_dataclasses/    # Data models (atom, detector, HKL, pattern, etc.)
├── bin/                # Compiled C executables (euler, peaksearch, pix2qs)
└── src/                # C source trees (euler, peaksearch, pixels2qs)
```

- **C binaries** (`euler`, `peaksearch`, `pix2qs`) live under `laueindexing/bin/` after installation.
- **Source** under `laueindexing/src/` is included for reference; users do not invoke it directly.

## Installation

Prerequisites:

- Python ≥ 3.12  
- System libraries and tools: `[make, gcc, h5cc]`, GNU Scientific Library (GSL)  
- Python dependencies: `numpy`, `pyyaml`, `h5py`, `mpi4py`

Install from source:

```bash
git clone https://github.com/yourusername/laue_indexing.git
cd laue_indexing
python3 -m pip install .
```

The `setup.py` build step will compile the C binaries into `laueindexing/bin/`.  

## Configuration

Provide a YAML config file (see `tests/data/test_config.yaml` for an example).  Key entries:

- `filefolder`: Directory of input HDF5 files
- `geoFile`: Path to the geometry XML file
- `crystFile`: Path to the crystal-structure file
- `outputFolder`: Directory for generated output  
- Optional range filters: `scanPointStart`, `scanPointEnd`, `depthRangeStart`, `depthRangeEnd`

## Usage

### Command-line / MPI

Run with MPI to parallelize:

```bash
mpirun -np 32 python -m laueindexing.pyLaueGo path/to/config.yml
```

### Python API

```python
import yaml
from laueindexing.pyLaueGo import PyLaueGo

with open('path/to/config.yml') as f:
    config = yaml.safe_load(f)

processor = PyLaueGo(config=config)
processor.run_on_process()  # single-process execution
```

## Output Layout

Results are written under `outputFolder` as:

```
input_basename.xml          # Final processed XML
error_TIMESTAMP.log         # Error log
index/index_out_0_0.txt     # Index subprocess outputs
p2q/p2q_out_0_0.txt         # Pixels→Q space outputs
peaks/peaks_out_0_0.txt     # Peak-search outputs
```

## Manual C-Binary Compilation

If you need to recompile manually:

```bash
cd laueindexing/src/euler && make
cd ../peaksearch && make linux
cd ../pixels2qs && make
cp euler peaksearch pix2qs ../../laueindexing/bin/
```

## Testing

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```
