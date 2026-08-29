# Installation

`laueanalysis` builds from source. `python -m pip install .` compiles the native indexing library and the command-line programs with CMake, then installs the Python package.

## Supported platforms

- Linux x86-64 with GCC.
- Python 3.11, 3.12, or 3.13. The maintained environment uses Python 3.12.

macOS, Windows, Linux ARM, and Clang are not supported.

## Native dependencies

The build needs these tools and libraries:

| Dependency | Purpose |
| --- | --- |
| GCC with C99 support | Compiles all native code |
| CMake 3.24 or later and Ninja | Build system (pip installs them from PyPI when they are absent) |
| GSL 2.8 with headers | Indexing and reconstruction |
| HDF5 1.14 with headers and `h5cc` | Peak search and reconstruction |
| zlib with headers | HDF5 compression |
| OpenMP (part of GCC) | CPU reconstruction |
| `patchelf` | Normalizes library search paths in the installed files (pip installs it from PyPI) |

Other minor versions of GSL and HDF5 can work, but the tests run against the versions above.

CUDA is not required. See [GPU reconstruction](#gpu-reconstruction).

## Install with conda (recommended)

Use this path at APS and on any machine where you do not install system packages. `environment.yml` provides the compiler and all native libraries from conda-forge.

1. Clone the repository:

   ```console
   $ git clone https://github.com/AdvancedPhotonSource/laue-analysis.git
   $ cd laue-analysis
   ```

2. Create and activate the environment:

   ```console
   $ conda env create -f environment.yml
   $ conda activate laue-analysis
   ```

3. Install the package:

   ```console
   $ python -m pip install .
   ```

## Install with a virtual environment

Use this path on a Linux system that already provides the native dependencies, for example through the distribution package manager or HPC environment modules.

1. Install the native dependencies. On Debian or Ubuntu:

   ```console
   $ sudo apt-get install build-essential libgsl-dev libhdf5-dev zlib1g-dev
   ```

2. Create and activate a virtual environment:

   ```console
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```

3. Install the package:

   ```console
   $ python -m pip install .
   ```

Do not run `pip` with `sudo`.

The installed programs find GSL and HDF5 through the dynamic loader's default search path. If the libraries come from environment modules or another non-system prefix, make sure that prefix is in `LD_LIBRARY_PATH` at run time. A virtual environment created on top of a conda environment does not see the conda libraries unless you set `LD_LIBRARY_PATH` to `<conda prefix>/lib`.

## Verify the installation

Run these commands after installation.

1. Check the Python package:

   ```console
   $ python -c "from laueanalysis.indexing import Indexer, index_frame; print('import ok')"
   import ok
   ```

2. Check that the native indexing library loads:

   ```console
   $ python -c "from laueanalysis.indexing._liblaue import get_library; get_library(); print('native indexing ok')"
   native indexing ok
   ```

3. Check that the CPU reconstruction program runs:

   ```console
   $ python -c "from laueanalysis.reconstruct import find_executable; print(find_executable())"
   ```

The second command loads a private module only as an installation check. Do not use `_liblaue` as an application API.

## What the build produces

A default installation builds all of these:

| Component | Location in the installed package | Required |
| --- | --- | --- |
| `liblaue.so` | `laueanalysis/indexing/bin/` | Yes. The in-process indexing API needs it. |
| `peaksearch`, `pix2qs`, `euler` | `laueanalysis/indexing/bin/` | Yes by default. Used by the LaueGo compatibility functions. |
| `reconstructN_cpu` | `laueanalysis/reconstruct/bin/` | Yes by default. Used by `reconstruct()`. |

The build stops with an error when a required dependency is missing or a component fails to compile. It does not install a partial package.

Native code is compiled for the generic x86-64 baseline with SSE2. The same installed files run on any x86-64 Linux machine that has compatible GSL and HDF5 libraries.

(gpu-reconstruction)=

## GPU reconstruction

The package build does not compile the CUDA reconstruction program. `reconstruct_gpu()` remains in the API and raises `FileNotFoundError` when `reconstructN_gpu` is not on `PATH`. The CUDA source and a stand-alone Makefile are kept in `src/laueanalysis/reconstruct/source/recon_gpu/` for users who have an NVIDIA toolchain. Use `reconstruct()` for CPU reconstruction.

## Troubleshooting

**The build reports that GSL or HDF5 was not found.** Install the development package that provides headers and `gsl-config` or `h5cc`, then run `pip install .` again. In a conda environment, confirm that the environment is active.

**The build reports that `patchelf` is required.** This appears only with `pip install --no-build-isolation`. Install `patchelf` into the environment (`pip install patchelf` or `conda install -c conda-forge patchelf`).

**`liblaue.so` fails to load with "cannot open shared object file".** The dynamic loader cannot find `libgsl` or `libhdf5`. Activate the environment that provides them, or add their directory to `LD_LIBRARY_PATH`.

**`reconstructN_gpu` is not found.** GPU reconstruction is not built by the package. See [GPU reconstruction](#gpu-reconstruction).

## Report a problem

Open an issue in the [laue-analysis repository](https://github.com/AdvancedPhotonSource/laue-analysis/issues). Include:

- The `laueanalysis` version or Git commit
- Operating system, Python version, and whether you used conda or a virtual environment
- The installation command
- The complete compiler, CMake, or linker error

Remove sample names, user names, local paths, and other sensitive experiment metadata before posting logs.
