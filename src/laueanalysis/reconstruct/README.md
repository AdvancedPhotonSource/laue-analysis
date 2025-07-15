# Laue Analysis Reconstruction API

This module provides a clean, functional API for wire scan reconstruction in Laue diffraction analysis.

## Installation

The reconstruction module requires a compiled C executable. Please ensure the `reconstructN_cpu` executable is either:
1. Installed in the package's `bin` directory
2. Available in your system PATH

## Quick Start

```python
from laueanalysis.reconstruct import reconstruct, batch, depth_scan

# Single reconstruction
result = reconstruct(
    input_file='wirescan.h5',
    output_file='output_',
    geometry_file='geometry.xml',
    depth_range=(-10.0, 10.0),
    resolution=0.5
)

if result.success:
    print(f"Generated {len(result.output_files)} files")
else:
    print(f"Error: {result.error}")
```

## API Reference

### Main Functions

#### `reconstruct()`
Perform a single wire scan reconstruction.

**Parameters:**
- `input_file`: Path to input HDF5 file
- `output_file`: Base path for output files (without extension)
- `geometry_file`: Path to geometry XML file
- `depth_range`: Tuple of (start, end) depths in microns
- `resolution`: Depth resolution in microns (default: 1.0)
- `image_range`: Optional tuple of (first, last) image indices
- `verbose`: Verbosity level 0-3 (default: 1)
- `percent_brightest`: Process only N% brightest pixels (default: 100)
- `wire_edge`: Wire edge to use - 'leading', 'trailing', or 'both' (default: 'leading')
- `memory_limit_mb`: Memory limit in MB (default: 128)
- `executable`: Optional path to executable (default: auto-detect)
- `timeout`: Timeout in seconds (default: 7200)

**Returns:** `ReconstructionResult` with fields:
- `success`: Whether reconstruction succeeded
- `output_files`: List of generated files
- `log`: Execution log
- `error`: Error message if failed
- `command`: Command that was executed
- `return_code`: Process return code

#### `batch()`
Run multiple reconstructions.

**Parameters:**
- `reconstructions`: List of configuration dictionaries
- `parallel`: Whether to run in parallel (default: False)
- `max_workers`: Number of parallel workers (default: CPU count)
- `stop_on_error`: Stop if any reconstruction fails (default: False)
- `progress_callback`: Optional callback function(completed, total)

**Returns:** List of `ReconstructionResult` objects

#### `depth_scan()`
Convenience function for scanning multiple depth ranges.

**Parameters:**
- `input_file`: Input HDF5 file path
- `output_base`: Base path for output files
- `geometry_file`: Geometry XML file path
- `depth_ranges`: List of (start, end) depth tuples
- `resolution`: Depth resolution in microns
- `parallel`: Whether to run in parallel (default: True)
- `**kwargs`: Additional arguments passed to `reconstruct()`

**Returns:** List of `ReconstructionResult` objects

### Examples

#### Batch Processing
```python
configs = [
    {
        'input_file': 'scan1.h5',
        'output_file': 'out1_',
        'geometry_file': 'geo.xml',
        'depth_range': (0, 10)
    },
    {
        'input_file': 'scan2.h5',
        'output_file': 'out2_',
        'geometry_file': 'geo.xml',
        'depth_range': (0, 10)
    }
]

results = batch(configs, parallel=True, max_workers=4)
```

#### Depth Scanning
```python
results = depth_scan(
    'wirescan.h5',
    'output/scan_',
    'geometry.xml',
    [(-20, -10), (-10, 0), (0, 10), (10, 20)],
    resolution=0.5,
    percent_brightest=50.0
)
```

#### Progress Tracking
```python
def show_progress(completed, total):
    print(f"Progress: {completed}/{total} ({100*completed/total:.0f}%)")

results = batch(configs, progress_callback=show_progress)
```

## Migration from Old API

If you were using the old class-based API:

```python
# Old way
from laueanalysis.reconstruct.wirescan_interface import WireScanRecon
interface = WireScanRecon()
result = interface.run_reconstruction(...)

# New way
from laueanalysis.reconstruct import reconstruct
result = reconstruct(...)
```

Key differences:
- No need to instantiate a class
- Cleaner parameter names (e.g., `depth_range` instead of separate start/end)
- More intuitive wire edge names ('leading' instead of 'l')
- Same result format for compatibility

## Advanced Usage

### Custom Executable
```python
result = reconstruct(
    input_file='data.h5',
    output_file='output_',
    geometry_file='geo.xml',
    depth_range=(0, 10),
    executable='/path/to/custom/reconstructN_cpu'
)
```

### Error Handling
```python
result = reconstruct(...)

if not result.success:
    print(f"Command failed: {result.command}")
    print(f"Error: {result.error}")
    print(f"Return code: {result.return_code}")
    
    # Check log for details
    if result.log:
        print(f"Log output:\n{result.log}")
```

### Parallel Processing with Error Recovery
```python
results = batch(configs, parallel=True, stop_on_error=False)

successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Successful: {len(successful)}/{len(results)}")
for r in failed:
    print(f"Failed: {r.error}")
```

## Performance Tips

1. **Use parallel processing** for multiple reconstructions:
   ```python
   results = batch(configs, parallel=True)
   ```

2. **Process only brightest pixels** to speed up reconstruction:
   ```python
   result = reconstruct(..., percent_brightest=50.0)
   ```

3. **Adjust memory limit** based on your system:
   ```python
   result = reconstruct(..., memory_limit_mb=512)
   ```

4. **Use coarser resolution** for initial exploration:
   ```python
   # Quick scan
   result = reconstruct(..., resolution=2.0)
   
   # Fine scan
   result = reconstruct(..., resolution=0.1)
   ```

## Troubleshooting

### Executable Not Found
```python
from laueanalysis.reconstruct import find_executable

try:
    exe_path = find_executable()
    print(f"Found executable at: {exe_path}")
except FileNotFoundError:
    print("Executable not found. Please compile and install it.")
```

### Timeout Issues
```python
# Increase timeout for large datasets
result = reconstruct(..., timeout=14400)  # 4 hours
```

### Memory Issues
```python
# Reduce memory usage
result = reconstruct(
    ...,
    memory_limit_mb=64,
    percent_brightest=25.0
)
