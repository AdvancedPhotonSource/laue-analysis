# GPU reconstruction (CUDA) is not part of the package build

`reconstructN_gpu` is **not** built by `CMakeLists.txt` or `pip install .`.
The CUDA source and this directory's `Makefile` are kept for the planned
GPU migration (see `BUILD_DEPLOYMENT_PLAN.md`, "GPU reconstruction"), which
needs an NVIDIA test environment and a separate design pass.

`lauelab.reconstruct.reconstruct_gpu()` still looks for a
`reconstructN_gpu` executable in the package `bin/` directory or on `PATH`
and raises `FileNotFoundError` when it is absent. To experiment locally,
build with this Makefile (`nvcc`, HDF5, GSL required) and put the resulting
`bin/reconstructN_gpu` on `PATH`. Nothing in the supported install paths
depends on it.
