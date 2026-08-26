# Installation

`laueanalysis` currently builds from source. The build compiles the native indexing library and command-line programs before it installs the Python package.

## Requirements

Use Linux and Python 3.11 or later. The repository development environment currently uses Python 3.12.

Install these build tools and development libraries before you install the package:

- `make`
- GCC with C99 support
- `h5cc` from an HDF5 development installation
- GNU Scientific Library (GSL), including headers
- HDF5, including headers

CUDA and `nvcc` are optional. Without them, the build skips GPU reconstruction. CUDA is not required for the indexing API described in this documentation.

A conda environment specification is included in `environment.yml`. It records the versions used by the current development environment:

```console
$ conda env create -f environment.yml
$ conda activate laue-analysis
```

## Install the package

Clone the repository and install it from the repository root:

```console
$ git clone https://github.com/AdvancedPhotonSource/laue-analysis.git
$ cd laue-analysis
$ python -m pip install .
```

The source build places compiled indexing programs and `liblaue.so` in the installed `laueanalysis.indexing` package.

```{warning}
The current build script reports some native compilation failures as warnings and can continue the Python installation. Run the native verification below before you process data.
```

## Install for development

Install an editable copy with the test and documentation dependencies:

```console
$ python -m pip install -e '.[test,docs]'
```

See [Development](development/index.md) for test and documentation build commands.

## Verify the installation

First verify the Python package and public indexing API:

```console
$ python -c "from laueanalysis.indexing import Indexer, index_frame; print('import ok')"
import ok
```

Then verify that the native indexing library can load:

```console
$ python -c "from laueanalysis.indexing._liblaue import get_library; get_library(); print('native indexing ok')"
native indexing ok
```

The second command loads a private module only as an installation check. Do not use `_liblaue` as an application API.

## Troubleshooting

**The build lists `make`, `gcc`, or `h5cc` as missing.** Install the missing tool and confirm that it is available on `PATH`. Run `h5cc -show` to verify the HDF5 compiler wrapper.

**The linker cannot find GSL.** Install the GSL development package, including `libgsl` and `libgslcblas`, then rebuild the package.

**The Python import succeeds but native indexing does not.** Review the complete installation output for a failed native compilation. Reinstall after correcting the first compiler or linker error.

**The build cannot find `nvcc`.** Ignore this warning if you only need indexing or CPU reconstruction. The warning means that GPU reconstruction was not built.

## Report a problem

Open an issue in the [laue-analysis repository](https://github.com/AdvancedPhotonSource/laue-analysis/issues). Include:

- The `laueanalysis` version or Git commit
- Operating-system and Python versions
- The installation command
- Output from the failed verification command
- The complete compiler or linker error

Remove sample names, user names, local paths, and other sensitive experiment metadata before posting logs.
