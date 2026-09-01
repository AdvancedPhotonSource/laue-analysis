# Reconstruction

`laueanalysis.reconstruct` runs the native wire-scan reconstruction programs as subprocesses. The CPU executable, `reconstructN_cpu`, ships with the package. The CUDA executable, `reconstructN_gpu`, requires a separate CUDA build and must be available on `PATH` unless you pass its path explicitly.

Both reconstruction functions capture process output and return a {class}`~laueanalysis.reconstruct.ReconstructionResult`. A nonzero process exit, timeout, or process-execution error is reported in that result rather than raised. Executable discovery and validation errors raise exceptions before reconstruction starts.

```{eval-rst}
.. currentmodule:: laueanalysis.reconstruct

.. autoclass:: ReconstructionResult

.. autofunction:: reconstruct

.. autofunction:: reconstruct_gpu

.. autofunction:: find_executable

.. autofunction:: find_gpu_executable

.. autofunction:: gpu_available
```
