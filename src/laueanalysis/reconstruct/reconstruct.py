"""Wire scan reconstruction functions for Laue analysis."""

from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import subprocess
import os
import shutil
import functools
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
@functools.lru_cache(maxsize=2)
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
        # WireScan should output help text - check for expected output
        # The binary returns exit code 1 for help, so we check stdout instead
        if "Usage: WireScan" not in result.stdout and "Usage: WireScan" not in result.stderr:
            raise RuntimeError(
                f"Executable did not produce expected help output. "
                f"stdout: {result.stdout[:100]}... "
                f"stderr: {result.stderr[:100]}..."
            )
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
        # Execute the subprocess - RPATH handles library loading
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
    wire_depths_file: Optional[str] = None,
    # Threaded version parameters
    num_threads: Optional[int] = None,
    rows_per_stripe: Optional[int] = None,
    cosmic_filter: bool = False,
    norm_exponent: Optional[float] = None,
    norm_threshold: Optional[float] = None
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
        num_threads: Number of OpenMP threads (default: auto-detect)
        rows_per_stripe: Rows to process per stripe (default: 256)
        cosmic_filter: Enable cosmic ray filtering
        norm_exponent: Exponent for image intensity scaling (e.g., 0.5)
        norm_threshold: Threshold for image intensity scaling
        
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
        '-i', str(input_file),
        '-o', str(output_file),
        '-g', str(geometry_file),
        '-s', str(depth_range[0]),
        '-e', str(depth_range[1]),
        '-r', str(resolution),
        '-v', str(verbose),
        '-p', str(percent_brightest),
        '-w', _map_wire_edge(wire_edge),
        '-m', str(memory_limit_mb),
        '-D', str(detector_number)
    ]
    
    # Add optional parameters
    if image_range is not None:
        cmd.extend(['-f', str(image_range[0])])
        cmd.extend(['-l', str(image_range[1])])
    if normalization:
        cmd.extend(['-n', normalization])
    if output_pixel_type is not None:
        cmd.extend(['-t', str(output_pixel_type)])
    if distortion_map:
        cmd.extend(['-d', distortion_map])
    if wire_depths_file:
        cmd.extend(['--wireDepths', wire_depths_file])
    
    # Add threaded version parameters
    if num_threads is not None:
        cmd.extend(['-N', str(num_threads)])
    if rows_per_stripe is not None:
        cmd.extend(['-R', str(rows_per_stripe)])
    if cosmic_filter:
        cmd.append('-C')
    if norm_exponent is not None:
        cmd.extend(['-E', str(norm_exponent)])
    if norm_threshold is not None:
        cmd.extend(['-T', str(norm_threshold)])
    
    return _execute_reconstruction(cmd, str(output_file), timeout)




# GPU reconstruction function
def reconstruct_gpu(
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
    # GPU-specific parameters
    wire_depths_file: Optional[str] = None,
    cuda_rows: int = 8
) -> ReconstructionResult:
    """
    Reconstruct wire scan data using GPU (CUDA) implementation.
    
    Note: This GPU version does not support cosmic ray filtering or 
    advanced normalization (norm_exponent/threshold). Use the CPU version
    if these features are required.
    
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
        wire_depths_file: Optional file with depth corrections for each pixel
        cuda_rows: Number of CUDA rows to process (default 8)
        
    Returns:
        ReconstructionResult containing:
            - success: Whether reconstruction succeeded
            - output_files: List of generated files
            - log: Execution log
            - error: Error message if failed
            - command: Command that was executed
            - return_code: Process return code
            
    Raises:
        FileNotFoundError: If GPU executable cannot be found
        ValueError: If parameters are invalid
    """
    # Find GPU executable
    if executable:
        exe_path = executable
    else:
        exe_path = _find_executable('reconstructN_gpu')
    
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
        '-i', str(input_file),
        '-o', str(output_file),
        '-g', str(geometry_file),
        '-s', str(depth_range[0]),
        '-e', str(depth_range[1]),
        '-r', str(resolution),
        '-v', str(verbose),
        '-p', str(percent_brightest),
        '-w', _map_wire_edge(wire_edge),
        '-m', str(memory_limit_mb),
        '-D', str(detector_number),
        '-R', str(cuda_rows)  # GPU uses -R for CUDA rows
    ]
    
    # Add optional parameters
    if image_range is not None:
        cmd.extend(['-f', str(image_range[0])])
        cmd.extend(['-l', str(image_range[1])])
    if normalization:
        cmd.extend(['-n', normalization])
    if output_pixel_type is not None:
        cmd.extend(['-t', str(output_pixel_type)])
    if distortion_map:
        cmd.extend(['-d', distortion_map])
    if wire_depths_file:
        cmd.extend(['-W', wire_depths_file])  # Note: GPU uses -W, not --wireDepths
    
    return _execute_reconstruction(cmd, str(output_file), timeout)




# Utility functions for common use cases
def find_executable() -> str:
    """
    Find the CPU reconstruction executable.
    
    Returns:
        Path to the executable
        
    Raises:
        FileNotFoundError: If executable cannot be found
    """
    return _find_executable()


def find_gpu_executable() -> str:
    """
    Find the GPU reconstruction executable.
    
    Returns:
        Path to the GPU executable
        
    Raises:
        FileNotFoundError: If GPU executable cannot be found
    """
    return _find_executable('reconstructN_gpu')


def gpu_available() -> bool:
    """
    Check if GPU reconstruction is available.
    
    Returns:
        True if GPU reconstruction executable is found and works, False otherwise
    """
    try:
        exe_path = _find_executable('reconstructN_gpu')
        _validate_executable(exe_path)
        return True
    except (FileNotFoundError, RuntimeError):
        return False
