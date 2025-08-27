"""Functional interface for Laue indexing."""

from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import subprocess
import os
import shutil
import functools
import re
from importlib import resources

from laueanalysis.indexing.lau_dataclasses.step import Step
from laueanalysis.indexing.lau_dataclasses.indexing import Indexing


class IndexingResult(NamedTuple):
    """Result from an indexing operation."""
    success: bool
    output_files: Dict[str, str]  # e.g., {'peaks': 'path/to/peaks.txt', ...}
    n_peaks_found: int
    n_indexed: int
    indexing_data: Optional[Indexing]
    step_data: Optional[Step]
    config: Optional[object]  # Keep for compatibility but will be None
    log: str
    error: Optional[str] = None
    command_history: List[str] = []


@functools.lru_cache(maxsize=1)
def _find_executables() -> Dict[str, str]:
    """
    Find and cache paths to indexing executables.
    
    Returns:
        Dictionary mapping executable name to its path.
        
    Raises:
        FileNotFoundError: If an executable cannot be found.
    """
    execs = {}
    for name in ['peaksearch', 'pix2qs', 'euler']:
        try:
            bin_files = resources.files('laueanalysis.indexing.bin')
            exe_path = bin_files / name
            if exe_path.is_file():
                execs[name] = str(exe_path)
                continue
        except (ModuleNotFoundError, FileNotFoundError):
            pass
        
        system_exe = shutil.which(name)
        if system_exe:
            execs[name] = system_exe
            continue
        
        raise FileNotFoundError(
            f"Indexing executable '{name}' not found. "
            "Please compile it first or ensure it's in your PATH."
        )
    return execs


def _run_command(cmd: List[str], timeout: int) -> Tuple[bool, str, str, int]:
    """
    Execute a command and capture output.

    Args:
        cmd: Command and arguments.
        timeout: Timeout in seconds.

    Returns:
        A tuple of (success, stdout, stderr, return_code).
        Note: success is based on return code being 0. Some Laue tools
        have other success codes, which must be handled by the caller.
        This implements graceful degradation - continuing on errors.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        # Log errors but don't fail immediately - graceful degradation
        return result.returncode == 0, result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return False, "", f"Process timed out after {timeout} seconds", -1
    except FileNotFoundError as e:
        return False, "", f"Executable not found: {e}", -1
    except Exception as e:
        return False, "", f"An unexpected error occurred: {e}", -1


def _setup_output_dirs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create the standard output directory structure for indexing.

    Args:
        output_dir: The root output directory.

    Returns:
        A dictionary mapping directory type to its Path object.
    """
    output_dir = Path(output_dir)
    subdirs = {
        'peaks': output_dir / 'peaks',
        'p2q': output_dir / 'p2q',
        'index': output_dir / 'index',
        'error': output_dir / 'error'
    }
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def _run_peaksearch(input_image: str, output_dir: str,
                   boxsize: int, max_rfactor: float, min_size: int,
                   min_separation: int, threshold: int, peak_shape: str,
                   max_peaks: int, mask_file: Optional[str],
                   threshold_ratio: float, smooth: bool,
                   timeout: int = 300) -> Tuple[bool, str, str, int]:
    """
    Run the peaksearch executable.
    
    Args:
        input_image: Path to input image file.
        output_dir: Directory for output files.
        boxsize: Box size for peak search.
        max_rfactor: Maximum R-factor.
        min_size: Minimum peak size.
        min_separation: Minimum separation between peaks.
        threshold: Threshold value.
        peak_shape: Peak shape ('L', 'G', etc.).
        max_peaks: Maximum number of peaks.
        mask_file: Optional mask file path.
        threshold_ratio: Threshold ratio (-1 to disable).
        smooth: Whether to apply smoothing.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code).
    """
    executables = _find_executables()
    
    # Create output filename (remove .h5 extension for consistent naming)
    input_path = Path(input_image)
    if input_path.suffix == '.h5':
        base_name = input_path.stem  # Remove .h5
    else:
        base_name = input_path.stem
    output_file = Path(output_dir) / f'peaks_{base_name}.txt'
    
    # Build peaksearch command
    cmd = [
        executables['peaksearch'],
        '-b', str(boxsize),
        '-R', str(max_rfactor),
        '-m', str(min_size),
        '-s', str(min_separation),
        '-t', str(threshold),
        '-p', peak_shape,
        '-M', str(max_peaks)
    ]
    
    # Add optional parameters
    if mask_file:
        cmd.extend(['-K', mask_file])
    if threshold_ratio != -1:
        cmd.extend(['-T', str(threshold_ratio)])
    if smooth:
        cmd.extend(['-S'])  # Note: -S doesn't take a parameter
    
    # Add input and output files
    cmd.extend([input_image, str(output_file)])
    
    return _run_command(cmd, timeout)


def _run_p2q(peaks_file: str, output_dir: str, geo_file: str, crystal_file: str,
             timeout: int = 300) -> Tuple[bool, str, str, int]:
    """
    Run the pix2qs (p2q) executable.
    
    Args:
        peaks_file: Path to peaks file from peaksearch.
        output_dir: Directory for output files.
        geo_file: Path to geometry file.
        crystal_file: Path to crystal file.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code).
    """
    executables = _find_executables()
    
    # Create output filename
    peaks_path = Path(peaks_file)
    output_file = Path(output_dir) / f'p2q_{peaks_path.stem.replace("peaks_", "")}.txt'
    
    cmd = [
        executables['pix2qs'],
        '-g', geo_file,
        '-x', crystal_file,
        peaks_file,
        str(output_file)
    ]
    return _run_command(cmd, timeout)


def _run_indexing(p2q_file: str, output_dir: str,
                  index_kev_max_calc: float, index_kev_max_test: float,
                  index_angle_tolerance: float, index_cone: float,
                  index_h: int, index_k: int, index_l: int,
                  timeout: int = 300) -> Tuple[bool, str, str, int]:
    """
    Run the euler (indexing) executable.
    
    Args:
        p2q_file: Path to p2q file.
        output_dir: Directory for output files.
        index_kev_max_calc: Maximum keV for calculation.
        index_kev_max_test: Maximum keV for testing.
        index_angle_tolerance: Angle tolerance for indexing.
        index_cone: Cone angle for indexing.
        index_h: H index.
        index_k: K index.
        index_l: L index.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code).
    """
    executables = _find_executables()
    
    # Create output filename
    p2q_path = Path(p2q_file)
    output_file = Path(output_dir) / f'index_{p2q_path.stem.replace("p2q_", "")}.txt'
    
    cmd = [
        executables['euler'],
        '-q',  # quiet mode
        '-k', str(index_kev_max_calc),
        '-t', str(index_kev_max_test),
        '-a', str(index_angle_tolerance),
        '-c', str(index_cone),
        '-f', p2q_file,
        '-h', str(index_h), str(index_k), str(index_l),
        '-o', str(output_file)
    ]
    return _run_command(cmd, timeout)


def _parse_peaks_output(peaks_file: str) -> int:
    """
    Parse peaks file to count number of peaks found.

    Strategy:
    - Prefer $peakList dims if present: "$peakList <params_per_peak> <n_peaks> ..."
      e.g., "$peakList 8 13" means 13 peaks; "$peakList 8 0" means zero peaks.
    - Otherwise, use explicit $Npeaks header if present.
    - Otherwise, count numeric data rows following the $peakList header
      until the next header/comment/blank line.
    - Robust to files that use $-headers and // comments (not just #).
    """
    try:
        with open(peaks_file, "r") as f:
            lines = f.read().splitlines()

        # 1) Prefer $peakList dims: "$peakList <params_per_peak> <n_peaks> ..."
        for line in lines:
            m = re.match(r'^\s*\$peakList\s+(\d+)\s+(\d+)\b', line)
            if m:
                return int(m.group(2))

        # 2) Next prefer explicit header: $Npeaks <int>
        for line in lines:
            m = re.match(r'^\s*\$Npeaks\s+(\d+)', line)
            if m:
                return int(m.group(1))

        # 3) Fallback: count numeric rows after $peakList
        in_table = False
        count = 0
        for line in lines:
            s = line.strip()
            if not s:
                if in_table:
                    break
                continue
            if s.startswith("$peakList"):
                in_table = True
                continue
            if in_table:
                # Stop if we hit another header/comment
                if s.startswith("$") or s.startswith("#") or s.startswith("//"):
                    break
                # Count if the first token is numeric
                parts = s.split()
                try:
                    float(parts[0])
                    count += 1
                except (ValueError, IndexError):
                    # Non-numeric row signals end or malformed table
                    break
        return count
    except (FileNotFoundError, IOError):
        return 0


def _parse_indexing_output(index_file: str) -> int:
    """
    Parse indexing output file to count number of indexed reflections.
    
    Args:
        index_file: Path to indexing output file.
        
    Returns:
        Number of indexed reflections.
    """
    try:
        with open(index_file, 'r') as f:
            lines = f.readlines()
            # Look for indexed reflections count in output
            for line in lines:
                if 'indexed' in line.lower() and 'reflections' in line.lower():
                    # Extract number from line like "123 reflections indexed"
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            return int(word)
            return 0
    except (FileNotFoundError, IOError):
        return 0


def _override_depth_in_peaks(peaks_file: str, depth_value: float) -> None:
    """
    Override or insert the $depth header line in a peaks file so downstream pixels2qs
    uses this value instead of what's in the HDF5.
    """
    try:
        with open(peaks_file, "r") as f:
            lines = f.read().splitlines()
    except Exception as e:
        raise RuntimeError(f"Failed to read peaks file for depth override: {e}")
    replaced = False
    # Replace existing $depth if present before the data table
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("$Npeaks") or stripped.startswith("$peakList"):
            break
        if stripped.startswith("$depth"):
            lines[i] = f"$depth\t\t{depth_value}\t\t// depth for depth resolved images (micron)"
            replaced = True
            break
    if not replaced:
        # Insert before the first header separator line starting with //
        insert_at = None
        for i, line in enumerate(lines):
            if line.strip().startswith("//"):
                insert_at = i
                break
        if insert_at is None:
            insert_at = 0
        lines.insert(insert_at, f"$depth\t\t{depth_value}\t\t// depth for depth resolved images (micron)")
    try:
        with open(peaks_file, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write peaks file for depth override: {e}")


def index(input_image: str, output_dir: str, geo_file: str, crystal_file: str,
          *,
          # Peak search parameters
          boxsize: int = 5,
          max_rfactor: float = 2.0,
          min_size: int = 3,
          min_separation: int = 10,
          threshold: int = 100,
          peak_shape: str = 'L',
          max_peaks: int = 50,
          mask_file: Optional[str] = None,
          threshold_ratio: float = -1,
          smooth: bool = False,
          # Indexing parameters
          index_kev_max_calc: float = 30.0,
          index_kev_max_test: float = 35.0,
          index_angle_tolerance: float = 0.12,
          index_cone: float = 72.0,
          index_h: int = 0,
          index_k: int = 0,
          index_l: int = 1,
          # General parameters
          depth_override: Optional[float] = None,
          timeout: int = 300) -> IndexingResult:
    """
    Perform complete Laue indexing on an input image.
    
    This function runs the full indexing pipeline:
    1. Peak search to find reflections in the image
    2. Pixel-to-q conversion using geometry
    3. Indexing against crystal structure
    
    Args:
        input_image: Path to input diffraction image.
        output_dir: Directory for all output files.
        geo_file: Path to geometry file.
        crystal_file: Path to crystal structure file.
        boxsize: Box size for peak search (default: 5).
        max_rfactor: Maximum R-factor for peak validation (default: 2.0).
        min_size: Minimum peak size in pixels (default: 3).
        min_separation: Minimum separation between peaks (default: 10).
        threshold: Intensity threshold for peak detection (default: 100).
        peak_shape: Peak shape model - 'L' for Lorentzian, 'G' for Gaussian (default: 'L').
        max_peaks: Maximum number of peaks to find (default: 50).
        mask_file: Optional path to mask file.
        threshold_ratio: Threshold ratio, -1 to disable (default: -1).
        smooth: Apply smoothing before peak search (default: False).
        index_kev_max_calc: Maximum keV for indexing calculation (default: 30.0).
        index_kev_max_test: Maximum keV for indexing test (default: 35.0).
        index_angle_tolerance: Angle tolerance in degrees for indexing (default: 0.12).
        index_cone: Cone angle in degrees for indexing (default: 72.0).
        index_h: H component of reference vector (default: 0).
        index_k: K component of reference vector (default: 0).
        index_l: L component of reference vector (default: 1).
        timeout: Timeout for each subprocess in seconds (default: 300).
        
    Returns:
        IndexingResult containing success status, output files, and statistics.
    """
    
    # Set up output directories
    try:
        subdirs = _setup_output_dirs(output_dir)
    except Exception as e:
        return IndexingResult(
            success=False,
            output_files={},
            n_peaks_found=0,
            n_indexed=0,
            indexing_data=None,
            step_data=None,
            config=None,
            log="",
            error=f"Failed to create output directories: {e}",
            command_history=[]
        )
    
    command_history = []
    log_parts = []
    output_files = {}
    
    # Step 1: Run peaksearch
    log_parts.append("Running peak search...")
    success, stdout, stderr, returncode = _run_peaksearch(
        input_image, str(subdirs['peaks']),
        boxsize, max_rfactor, min_size,
        min_separation, threshold, peak_shape,
        max_peaks, mask_file,
        threshold_ratio, smooth,
        timeout
    )
    command_history.append(f"peaksearch {input_image} -> {subdirs['peaks']}")
    log_parts.append(f"Peak search stdout: {stdout}")
    if stderr:
        log_parts.append(f"Peak search stderr: {stderr}")
    
    # Continue even if peaksearch has errors (might still produce output)
    if not success:
        log_parts.append(f"Peak search had errors (return code {returncode}), but checking for output files...")
    
    # Find peaks output file (typically peaks_*.txt)
    peaks_files = list(subdirs['peaks'].glob('peaks_*.txt'))
    if not peaks_files:
        # Only fail if no output file was created at all
        return IndexingResult(
            success=False,
            output_files=output_files,
            n_peaks_found=0,
            n_indexed=0,
            indexing_data=None,
            step_data=None,
            config=None,
            log="\n".join(log_parts),
            error="No peaks output file found - peak search completely failed",
            command_history=command_history
        )
    
    peaks_file = str(peaks_files[0])
    output_files['peaks'] = peaks_file
    n_peaks_found = _parse_peaks_output(peaks_file)
    # Optional: override depth header before pixels2qs step
    if depth_override is not None:
        try:
            _override_depth_in_peaks(peaks_file, depth_override)
            log_parts.append(f"Depth override applied: $depth = {depth_override}")
            command_history.append(f"override_depth {peaks_file} -> {depth_override}")
        except Exception as e:
            log_parts.append(f"Depth override failed: {e}")
    
    # Only run p2q if we have peaks
    if n_peaks_found > 0:
        # Step 2: Run p2q conversion
        log_parts.append("Running pixel-to-q conversion...")
        success, stdout, stderr, returncode = _run_p2q(
            peaks_file, str(subdirs['p2q']), geo_file, crystal_file, timeout
        )
        command_history.append(f"pix2qs {peaks_file} -> {subdirs['p2q']}")
        log_parts.append(f"P2Q stdout: {stdout}")
        if stderr:
            log_parts.append(f"P2Q stderr: {stderr}")
        
        if not success:
            log_parts.append(f"P2Q conversion failed with return code {returncode}, but continuing...")
            # Don't fail completely, continue without p2q results
            return IndexingResult(
                success=True,  # Still success, just no p2q/indexing
                output_files=output_files,
                n_peaks_found=n_peaks_found,
                n_indexed=0,
                indexing_data=None,
                step_data=None,
                config=None,
                log="\n".join(log_parts),
                error=None,
                command_history=command_history
            )
        
        # Find p2q output file
        p2q_files = list(subdirs['p2q'].glob('p2q_*.txt'))
        if not p2q_files:
            log_parts.append("No p2q output file found, skipping indexing...")
            return IndexingResult(
                success=True,  # Still success, just no indexing
                output_files=output_files,
                n_peaks_found=n_peaks_found,
                n_indexed=0,
                indexing_data=None,
                step_data=None,
                config=None,
                log="\n".join(log_parts),
                error=None,
                command_history=command_history
            )
        
        p2q_file = str(p2q_files[0])
        output_files['p2q'] = p2q_file
        
        # Only run indexing if we have at least 2 peaks
        if n_peaks_found > 1:
            # Step 3: Run indexing
            log_parts.append("Running indexing...")
            success, stdout, stderr, returncode = _run_indexing(
                p2q_file, str(subdirs['index']),
                index_kev_max_calc, index_kev_max_test,
                index_angle_tolerance, index_cone,
                index_h, index_k, index_l,
                timeout
            )
            command_history.append(f"euler {p2q_file} -> {subdirs['index']}")
            log_parts.append(f"Indexing stdout: {stdout}")
            if stderr:
                log_parts.append(f"Indexing stderr: {stderr}")
            
            if not success:
                log_parts.append(f"Indexing failed with return code {returncode}, but continuing...")
                # Don't fail completely, return what we have
                return IndexingResult(
                    success=True,  # Still success, just no indexing results
                    output_files=output_files,
                    n_peaks_found=n_peaks_found,
                    n_indexed=0,
                    indexing_data=None,
                    step_data=None,
                    config=None,
                    log="\n".join(log_parts),
                    error=None,
                    command_history=command_history
                )
            
            # Find indexing output file
            index_files = list(subdirs['index'].glob('index_*.txt'))
            if index_files:
                index_file = str(index_files[0])
                output_files['index'] = index_file
                n_indexed = _parse_indexing_output(index_file)
            else:
                n_indexed = 0
                log_parts.append("No indexing output file found")
        else:
            log_parts.append(f"Only {n_peaks_found} peak(s) found, need at least 2 for indexing. Skipping indexing step.")
            n_indexed = 0
    else:
        log_parts.append("No peaks found, skipping p2q and indexing steps.")
        n_indexed = 0
    
    log_parts.append("Indexing completed successfully!")
    
    return IndexingResult(
        success=True,
        output_files=output_files,
        n_peaks_found=n_peaks_found,
        n_indexed=n_indexed,
        indexing_data=None,
        step_data=None,
        config=None,
        log="\n".join(log_parts),
        error=None,
        command_history=command_history
    )
