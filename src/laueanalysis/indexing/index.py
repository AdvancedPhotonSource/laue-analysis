"""Functional interface for Laue indexing."""

from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import subprocess
import os
import shutil
import functools
from importlib import resources

from laueanalysis.indexing.lau_dataclasses.step import Step
from laueanalysis.indexing.lau_dataclasses.indexing import Indexing
from laueanalysis.indexing.lau_dataclasses.config import LaueConfig


class IndexingResult(NamedTuple):
    """Result from an indexing operation."""
    success: bool
    output_files: Dict[str, str]  # e.g., {'peaks': 'path/to/peaks.txt', ...}
    n_peaks_found: int
    n_indexed: int
    indexing_data: Optional[Indexing]
    step_data: Optional[Step]
    config: Optional[LaueConfig]
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
        This matches pyLaueGo's behavior of continuing on errors.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        # Match pyLaueGo behavior: log errors but don't fail immediately
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


def _run_peaksearch(input_image: str, output_dir: str, config: LaueConfig, 
                   timeout: int = 300) -> Tuple[bool, str, str, int]:
    """
    Run the peaksearch executable.
    
    Args:
        input_image: Path to input image file.
        output_dir: Directory for output files.
        config: Configuration containing peaksearch parameters.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code).
    """
    executables = _find_executables()
    
    # Create output filename (match pyLaueGo naming: remove .h5 extension)
    input_path = Path(input_image)
    if input_path.suffix == '.h5':
        base_name = input_path.stem  # Remove .h5
    else:
        base_name = input_path.stem
    output_file = Path(output_dir) / f'peaks_{base_name}.txt'
    
    # Build command like the original pyLaueGo
    cmd = [
        executables['peaksearch'],
        '-b', str(config.boxsize or 5),
        '-R', str(config.maxRfactor or 2.0),
        '-m', str(config.min_size or 3),
        '-s', str(config.min_separation or 10),
        '-t', str(config.threshold or 100),
        '-p', config.peakShape or 'L',
        '-M', str(config.max_peaks or 50)
    ]
    
    # Add optional parameters
    if config.maskFile:
        cmd.extend(['-K', config.maskFile])
    if config.thresholdRatio != -1:
        cmd.extend(['-T', str(config.thresholdRatio)])
    if config.smooth:
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


def _run_indexing(p2q_file: str, output_dir: str, config: LaueConfig,
                  timeout: int = 300) -> Tuple[bool, str, str, int]:
    """
    Run the euler (indexing) executable.
    
    Args:
        p2q_file: Path to p2q file.
        output_dir: Directory for output files.
        config: Configuration containing indexing parameters.
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
        '-k', str(config.indexKeVmaxCalc or 30.0),
        '-t', str(config.indexKeVmaxTest or 35.0),
        '-a', str(config.indexAngleTolerance or 0.12),
        '-c', str(config.indexCone or 72.0),
        '-f', p2q_file,
        '-h', str(config.indexH or 0), str(config.indexK or 0), str(config.indexL or 1),
        '-o', str(output_file)
    ]
    return _run_command(cmd, timeout)


def _parse_peaks_output(peaks_file: str) -> int:
    """
    Parse peaks file to count number of peaks found.
    
    Args:
        peaks_file: Path to peaks output file.
        
    Returns:
        Number of peaks found.
    """
    try:
        with open(peaks_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines and count data lines
            data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            return len(data_lines)
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


def index(input_image: str, output_dir: str, geo_file: str, crystal_file: str,
          config: Optional[LaueConfig] = None, step_data: Optional[Step] = None, 
          indexing_data: Optional[Indexing] = None, timeout: int = 300) -> IndexingResult:
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
        config: LaueConfig with processing parameters (defaults will be used if None).
        step_data: Step configuration (for compatibility, currently unused).
        indexing_data: Indexing configuration (for compatibility, currently unused).
        timeout: Timeout for each subprocess in seconds.
        
    Returns:
        IndexingResult containing success status, output files, and statistics.
    """
    # Set up default config if not provided
    if config is None:
        config = LaueConfig()
        config.geoFile = geo_file
        config.crystFile = crystal_file
    
    # Set up output directories
    try:
        subdirs = _setup_output_dirs(output_dir)
    except Exception as e:
        return IndexingResult(
            success=False,
            output_files={},
            n_peaks_found=0,
            n_indexed=0,
            indexing_data=indexing_data,
            step_data=step_data,
            config=config,
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
        input_image, str(subdirs['peaks']), config, timeout
    )
    command_history.append(f"peaksearch {input_image} -> {subdirs['peaks']}")
    log_parts.append(f"Peak search stdout: {stdout}")
    if stderr:
        log_parts.append(f"Peak search stderr: {stderr}")
    
    # Match pyLaueGo: continue even if peaksearch has errors (might still produce output)
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
            indexing_data=indexing_data,
            step_data=step_data,
            config=config,
            log="\n".join(log_parts),
            error="No peaks output file found - peak search completely failed",
            command_history=command_history
        )
    
    peaks_file = str(peaks_files[0])
    output_files['peaks'] = peaks_file
    n_peaks_found = _parse_peaks_output(peaks_file)
    
    # Match pyLaueGo logic: only run p2q if we have peaks
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
            # Match pyLaueGo: don't fail completely, continue without p2q results
            return IndexingResult(
                success=True,  # Still success, just no p2q/indexing
                output_files=output_files,
                n_peaks_found=n_peaks_found,
                n_indexed=0,
                indexing_data=indexing_data,
                step_data=step_data,
                config=config,
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
                indexing_data=indexing_data,
                step_data=step_data,
                config=config,
                log="\n".join(log_parts),
                error=None,
                command_history=command_history
            )
        
        p2q_file = str(p2q_files[0])
        output_files['p2q'] = p2q_file
        
        # Match pyLaueGo logic: only run indexing if we have at least 2 peaks
        if n_peaks_found > 1:
            # Step 3: Run indexing
            log_parts.append("Running indexing...")
            success, stdout, stderr, returncode = _run_indexing(
                p2q_file, str(subdirs['index']), config, timeout
            )
            command_history.append(f"euler {p2q_file} -> {subdirs['index']}")
            log_parts.append(f"Indexing stdout: {stdout}")
            if stderr:
                log_parts.append(f"Indexing stderr: {stderr}")
            
            if not success:
                log_parts.append(f"Indexing failed with return code {returncode}, but continuing...")
                # Match pyLaueGo: don't fail completely, return what we have
                return IndexingResult(
                    success=True,  # Still success, just no indexing results
                    output_files=output_files,
                    n_peaks_found=n_peaks_found,
                    n_indexed=0,
                    indexing_data=indexing_data,
                    step_data=step_data,
                    config=config,
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
        indexing_data=indexing_data,
        step_data=step_data,
        config=config,
        log="\n".join(log_parts),
        error=None,
        command_history=command_history
    )