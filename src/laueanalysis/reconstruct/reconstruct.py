"""Wire scan reconstruction functions for Laue analysis."""

from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import subprocess
import os
import shutil
import functools
from concurrent.futures import ProcessPoolExecutor
from importlib import resources

# Result type for reconstruction operations
class ReconstructionResult(NamedTuple):
    """Result from a reconstruction operation."""
    success: bool
    output_files: List[str]
    log: str
    error: Optional[str] = None
    command: str = ""
    return_code: int = 0


# Cache the executable path lookup to avoid repeated filesystem searches.
# Finding the executable involves checking package resources and PATH,
# which can be slow. Since the executable location doesn't change during
# program execution, we cache it after the first lookup.
@functools.lru_cache(maxsize=1)
def _find_executable(name: str = 'reconstructN_cpu') -> str:
    """
    Find and cache the reconstruction executable path.
    
    Args:
        name: Name of the executable to find
        
    Returns:
        Path to the executable
        
    Raises:
        FileNotFoundError: If executable cannot be found
    """
    # First try the package bin directory
    try:
        bin_files = resources.files('laueanalysis.reconstruct.bin')
        exe_path = bin_files / name
        if exe_path.is_file():
            return str(exe_path)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    
    # Try to find in PATH
    system_exe = shutil.which(name)
    if system_exe:
        return system_exe
    
    raise FileNotFoundError(
        f"Reconstruction executable '{name}' not found. "
        "Please compile it first or ensure it's in your PATH."
    )


def _validate_executable(exe_path: str) -> None:
    """
    Validate that executable works by running with --help.
    
    Args:
        exe_path: Path to executable
        
    Raises:
        RuntimeError: If executable doesn't work properly
    """
    try:
        result = subprocess.run(
            [exe_path, '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # WireScan should output help text
    except (subprocess.CalledProcessError, FileNotFoundError, 
            subprocess.TimeoutExpired, PermissionError) as e:
        raise RuntimeError(
            f"Executable validation failed: {exe_path}. Error: {e}"
        )


def _map_wire_edge(edge: str) -> str:
    """Map user-friendly edge names to program flags."""
    edge_map = {
        'leading': 'l',
        'trailing': 't', 
        'both': 'b',
        'l': 'l',
        't': 't',
        'b': 'b'
    }
    edge_lower = edge.lower()
    if edge_lower not in edge_map:
        raise ValueError(
            f"Invalid wire_edge '{edge}'. "
            "Must be 'leading', 'trailing', or 'both'"
        )
    return edge_map[edge_lower]


def _execute_reconstruction(
    cmd: List[str], 
    output_base: str,
    timeout: int = 7200
) -> ReconstructionResult:
    """
    Execute the reconstruction subprocess.
    
    Args:
        cmd: Command and arguments
        output_base: Base path for output files
        timeout: Timeout in seconds (default 2 hours)
        
    Returns:
        ReconstructionResult with execution details
    """
    try:
        # Execute the subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=timeout
        )
        
        success = result.returncode == 0
        
        # Find output files
        output_files = []
        if success:
            output_dir = Path(output_base).parent
            if output_dir.exists():
                # Look for files matching the output pattern
                output_pattern = Path(output_base).name + "*"
                output_files = [str(f) for f in output_dir.glob(output_pattern)]
        
        return ReconstructionResult(
            success=success,
            output_files=output_files,
            log=result.stdout,
            error=result.stderr if not success else None,
            command=' '.join(cmd),
            return_code=result.returncode
        )
        
    except subprocess.TimeoutExpired:
        return ReconstructionResult(
            success=False,
            output_files=[],
            log='',
            error=f'Process timed out after {timeout} seconds',
            command=' '.join(cmd),
            return_code=-1
        )
    except Exception as e:
        return ReconstructionResult(
            success=False,
            output_files=[],
            log='',
            error=str(e),
            command=' '.join(cmd),
            return_code=-1
        )


def reconstruct(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    geometry_file: Union[str, Path],
    depth_range: Tuple[float, float],
    resolution: float = 1.0,
    *,
    image_range: Optional[Tuple[int, int]] = None,
    verbose: int = 1,
    percent_brightest: float = 100.0,
    wire_edge: str = 'leading',
    memory_limit_mb: int = 128,
    executable: Optional[str] = None,
    timeout: int = 7200,
    # Advanced options
    normalization: Optional[str] = None,
    output_pixel_type: Optional[int] = None,
    distortion_map: Optional[str] = None,
    detector_number: int = 0,
    wire_depths_file: Optional[str] = None
) -> ReconstructionResult:
    """
    Reconstruct wire scan data.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Base path for output files (without extension)
        geometry_file: Path to geometry XML file
        depth_range: Tuple of (start, end) depths in microns
        resolution: Depth resolution in microns (default 1.0)
        image_range: Optional tuple of (first, last) image indices
        verbose: Verbosity level 0-3 (default 1)
        percent_brightest: Process only N% brightest pixels (default 100)
        wire_edge: Wire edge to use - 'leading', 'trailing', or 'both' (default 'leading')
        memory_limit_mb: Memory limit in MB (default 128)
        executable: Optional path to executable (default: auto-detect)
        timeout: Timeout in seconds (default 7200 = 2 hours)
        normalization: Optional normalization tag
        output_pixel_type: Optional output pixel type (0-7, WinView numbers)
        distortion_map: Optional distortion map file
        detector_number: Detector number (default 0)
        wire_depths_file: Optional file with depth corrections
        
    Returns:
        ReconstructionResult containing:
            - success: Whether reconstruction succeeded
            - output_files: List of generated files
            - log: Execution log
            - error: Error message if failed
            - command: Command that was executed
            - return_code: Process return code
            
    Raises:
        FileNotFoundError: If executable cannot be found
        ValueError: If parameters are invalid
    """
    # Find executable
    if executable:
        exe_path = executable
    else:
        exe_path = _find_executable()
    
    # Validate executable on first use
    _validate_executable(exe_path)
    
    # Validate parameters
    if depth_range[0] >= depth_range[1]:
        raise ValueError(
            f"Invalid depth range: start ({depth_range[0]}) must be less than end ({depth_range[1]})"
        )
    
    # Build command
    cmd = [
        exe_path,
        '--infile', str(input_file),
        '--outfile', str(output_file),
        '--geofile', str(geometry_file),
        '--depth-start', str(depth_range[0]),
        '--depth-end', str(depth_range[1]),
        '--resolution', str(resolution),
        '--verbose', str(verbose),
        '--percent-to-process', str(percent_brightest),
        '--wire-edges', _map_wire_edge(wire_edge),
        '--memory', str(memory_limit_mb),
        '--detector_number', str(detector_number)
    ]
    
    # Add optional parameters
    if image_range is not None:
        cmd.extend(['--first-image', str(image_range[0])])
        cmd.extend(['--last-image', str(image_range[1])])
    if normalization:
        cmd.extend(['--normalization', normalization])
    if output_pixel_type is not None:
        cmd.extend(['--type-output-pixel', str(output_pixel_type)])
    if distortion_map:
        cmd.extend(['--distortion_map', distortion_map])
    if wire_depths_file:
        cmd.extend(['--wireDepths', wire_depths_file])
    
    return _execute_reconstruction(cmd, str(output_file), timeout)


def batch(
    reconstructions: List[Dict],
    parallel: bool = False,
    max_workers: Optional[int] = None,
    stop_on_error: bool = False,
    progress_callback: Optional[callable] = None
) -> List[ReconstructionResult]:
    """
    Run multiple reconstructions.
    
    Args:
        reconstructions: List of keyword argument dictionaries for reconstruct()
        parallel: Whether to run reconstructions in parallel
        max_workers: Number of parallel workers (None = CPU count)
        stop_on_error: Stop batch processing if any reconstruction fails
        progress_callback: Optional callback function(completed, total) for progress updates
        
    Returns:
        List of ReconstructionResult objects
        
    Example:
        >>> configs = [
        ...     {'input_file': 'scan1.h5', 'output_file': 'out1_', 
        ...      'geometry_file': 'geo.xml', 'depth_range': (0, 10)},
        ...     {'input_file': 'scan2.h5', 'output_file': 'out2_',
        ...      'geometry_file': 'geo.xml', 'depth_range': (0, 10)}
        ... ]
        >>> results = batch(configs, parallel=True, max_workers=4)
    """
    if not reconstructions:
        return []
    
    results = []
    
    if parallel:
        # Use a top-level function for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_reconstruction, config) 
                      for config in reconstructions]
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(reconstructions))
                
                if stop_on_error and not result.success:
                    # Cancel remaining futures
                    for f in futures[i+1:]:
                        f.cancel()
                    break
    else:
        # Sequential processing
        for i, config in enumerate(reconstructions):
            result = _run_single_reconstruction(config)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(reconstructions))
            
            if stop_on_error and not result.success:
                break
    
    return results


def _run_single_reconstruction(config: Dict) -> ReconstructionResult:
    """
    Run a single reconstruction with error handling.
    
    This is a module-level function to support multiprocessing.
    """
    try:
        return reconstruct(**config)
    except Exception as e:
        # Return error result instead of raising
        return ReconstructionResult(
            success=False,
            output_files=[],
            log='',
            error=f"Exception during reconstruction: {str(e)}",
            command='',
            return_code=-1
        )


def depth_scan(
    input_file: Union[str, Path],
    output_base: Union[str, Path],
    geometry_file: Union[str, Path],
    depth_ranges: List[Tuple[float, float]],
    resolution: float = 1.0,
    parallel: bool = True,
    **kwargs
) -> List[ReconstructionResult]:
    """
    Scan multiple depth ranges with automatic output naming.
    
    This is a convenience function that creates appropriate output filenames
    for each depth range and runs the reconstructions.
    
    Args:
        input_file: Input HDF5 file path
        output_base: Base path for output files (depth info will be appended)
        geometry_file: Geometry XML file path
        depth_ranges: List of (start, end) depth tuples in microns
        resolution: Depth resolution in microns
        parallel: Whether to run scans in parallel
        **kwargs: Additional arguments passed to reconstruct()
        
    Returns:
        List of ReconstructionResult objects
        
    Example:
        >>> results = depth_scan(
        ...     'wirescan.h5',
        ...     'output/scan_',
        ...     'geometry.xml',
        ...     [(0, 10), (10, 20), (20, 30)],
        ...     resolution=0.5,
        ...     percent_brightest=50.0
        ... )
    """
    configs = []
    
    for start, end in depth_ranges:
        # Create unique output name for each depth range
        output_file = f"{output_base}depth_{start:.1f}_{end:.1f}_"
        
        config = {
            'input_file': input_file,
            'output_file': output_file,
            'geometry_file': geometry_file,
            'depth_range': (start, end),
            'resolution': resolution,
            **kwargs
        }
        configs.append(config)
    
    return batch(configs, parallel=parallel)


# Utility functions for common use cases
def find_executable() -> str:
    """
    Find the reconstruction executable.
    
    Returns:
        Path to the executable
        
    Raises:
        FileNotFoundError: If executable cannot be found
    """
    return _find_executable()
