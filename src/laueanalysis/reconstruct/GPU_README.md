# GPU Reconstruction for Laue Wire Scan Analysis

This document describes the GPU-accelerated reconstruction functionality for Laue wire scan analysis.

## Overview

The GPU reconstruction provides a CUDA-accelerated version of the wire scan reconstruction algorithm, offering significant performance improvements over the CPU version for large datasets.

## Installation

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 7.0 or higher (V100, A100, etc.)
- CUDA Toolkit (tested with CUDA 11.0+)
- HDF5 library with development headers
- GSL (GNU Scientific Library)

### Building the GPU Executable

```bash
cd src/laueanalysis/reconstruct/source/recon_gpu
make clean
make

# For A100 GPUs:
make GPU_ARCH=sm_80

# To see build configuration:
make show-config
```

The executable `reconstructNGPU` will be created in the `bin/` directory.

## Usage

### Python API

```python
from laueanalysis.reconstruct import (
    reconstruct_gpu,
    batch_gpu,
    depth_scan_gpu,
    gpu_available
)

# Check if GPU reconstruction is available
if gpu_available():
    print("GPU reconstruction is available!")
    
# Single reconstruction
result = reconstruct_gpu(
    input_file='wirescan.h5',
    output_file='output_',
    geometry_file='geometry.xml',
    depth_range=(0, 100),
    resolution=1.0,
    cuda_rows=16  # GPU-specific parameter
)

# Batch processing
configs = [
    {'input_file': f'scan{i}.h5', 'output_file': f'out{i}_',
     'geometry_file': 'geo.xml', 'depth_range': (i*10, (i+1)*10)}
    for i in range(10)
]
results = batch_gpu(configs, parallel=True)

# Depth scanning
depth_ranges = [(0, 50), (50, 100), (100, 150)]
results = depth_scan_gpu(
    'input.h5', 'output_', 'geo.xml',
    depth_ranges, resolution=0.5
)
```

### Command Line

```bash
# Basic usage
reconstructNGPU -i input.h5 -o output_ -g geometry.xml -s 0 -e 100 -r 1 -D 0

# With GPU-specific options
reconstructNGPU -i input.h5 -o output_ -g geometry.xml \
    -s 0 -e 100 -r 1 -D 0 \
    -R 16 \                    # CUDA rows (default: 8)
    -W depth_corrections.dat   # Wire depth corrections file
```

## Differences from CPU Version

### Features NOT Available in GPU Version

1. **Cosmic Ray Filtering** (`-C` flag)
   - The GPU version does not support cosmic ray filtering
   - Use the CPU version if this feature is required

2. **Advanced Normalization** (`-E` and `-T` flags)
   - Normalization by image intensity with exponent
   - Normalization threshold
   - Use the CPU version for these normalization options

3. **OpenMP Threading** (`-N` flag)
   - GPU version uses CUDA parallelization instead
   - The `-N` parameter is not applicable

### GPU-Specific Features

1. **CUDA Rows** (`-R` parameter)
   - Controls the number of rows processed per CUDA kernel launch
   - Default: 8 rows
   - Typical range: 8-32 depending on GPU memory
   - Higher values may improve performance but require more GPU memory

2. **Wire Depth Corrections** (`-W` parameter)
   - Supports per-pixel depth corrections
   - Provide a file with depth correction values

### Parameter Differences

| Parameter | CPU Version | GPU Version | Notes |
|-----------|------------|-------------|-------|
| `-R` | Rows per stripe (default: 256) | CUDA rows (default: 8) | Different meaning! |
| `-C` | Cosmic filter | Not supported | CPU only |
| `-E` | Norm exponent | Not supported | CPU only |
| `-T` | Norm threshold | Not supported | CPU only |
| `-N` | OpenMP threads | Not supported | CPU only |
| `-W` | Not available | Wire depths file | GPU only |

## Performance Considerations

1. **Speed**: GPU version is typically 10-100x faster than CPU version
2. **Memory**: Limited by GPU memory rather than system RAM
3. **Optimal Settings**:
   - `cuda_rows`: Start with 8, increase if GPU has sufficient memory
   - `memory_limit_mb`: Less relevant for GPU, but still affects host-side operations
   - Process largest possible chunks to minimize overhead

## Troubleshooting

### GPU Not Available

If `gpu_available()` returns False:

1. Check if the GPU executable is compiled:
   ```bash
   ls src/laueanalysis/reconstruct/source/recon_gpu/bin/reconstructNGPU
   ```

2. Check if it's in your PATH:
   ```bash
   which reconstructNGPU
   ```

3. Verify CUDA installation:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Build Errors

1. **CUDA not found**: Set CUDA path in environment
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **HDF5 not found**: If using conda:
   ```bash
   conda activate your_env
   make clean && make
   ```

3. **Wrong GPU architecture**: Check your GPU and set appropriate architecture
   ```bash
   nvidia-smi  # Check GPU model
   make GPU_ARCH=sm_XX  # Set appropriate compute capability
   ```

### Runtime Errors

1. **Out of GPU memory**: Reduce `cuda_rows` parameter
2. **CUDA error**: Check `dmesg` for GPU errors
3. **Library not found**: Ensure CUDA libraries are in LD_LIBRARY_PATH

## Example Workflow

```python
from laueanalysis.reconstruct import gpu_available, reconstruct, reconstruct_gpu

# Choose backend based on availability and requirements
if gpu_available() and not need_cosmic_filter:
    # Use GPU for speed
    result = reconstruct_gpu(
        input_file='data.h5',
        output_file='output_',
        geometry_file='geo.xml',
        depth_range=(0, 100),
        cuda_rows=16
    )
else:
    # Fall back to CPU
    result = reconstruct(
        input_file='data.h5',
        output_file='output_',
        geometry_file='geo.xml',
        depth_range=(0, 100),
        cosmic_filter=True,  # CPU-only feature
        num_threads=8
    )

print(f"Reconstruction {'succeeded' if result.success else 'failed'}")
```

## Development Notes

- The GPU version is a CUDA port of the CPU version
- Core algorithm remains the same, but parallelization strategy differs
- GPU version processes multiple pixels simultaneously using CUDA threads
- Memory access patterns optimized for GPU architecture
