#!/usr/bin/env python3
"""
Simple interface for running WireScan reconstruction as a subprocess.

This module provides a clean, simple interface to run Laue wire scan reconstruction
from other Python programs via subprocess calls.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import shutil


class WireScanRecon:
    """
    Simple interface for running WireScan reconstruction via subprocess.
    
    This class provides a clean way to call the WireScan C program
    from other Python programs without direct C integration.
    """
    
    def __init__(self, wirescan_executable: str = None):
        """
        Initialize the interface.
        
        Args:
            wirescan_executable: Path to WireScan executable.
                               If None, looks for 'WireScan' in package bin or PATH.
        """
        if wirescan_executable:
            self.wirescan_executable = wirescan_executable
        else:
            self.wirescan_executable = self._find_wirescan_executable()
        
        self._validate_executable()
    
    def _find_wirescan_executable(self) -> str:
        """Find the WireScan executable in package or PATH."""
        # First try the package bin directory
        try:
            from importlib import resources
            bin_files = resources.files('laueanalysis.reconstruct.bin')
            wirescan_exe = bin_files / 'reconstructN_cpu'
            if wirescan_exe.is_file():
                return str(wirescan_exe)
        except (ModuleNotFoundError, FileNotFoundError):
            pass
        
        # Try to find in PATH
        result = shutil.which('reconstructN_cpu')
        if result:
            return result
            
        # Default assumption - might need compilation
        return 'reconstructN_cpu'
    
    def _validate_executable(self):
        """Validate that the WireScan executable exists and is executable."""
        try:
            # Try to run with --help to check if executable works
            result = subprocess.run([
                self.wirescan_executable, '--help'
            ], capture_output=True, text=True, timeout=30)
            # WireScan should output help text
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as e:
            raise RuntimeError(
                f"WireScan executable not found or not working: {self.wirescan_executable}. "
                f"Error: {e}. You may need to compile the C program first."
            )
    
    def run_reconstruction(self,
                          infile: str,
                          outfile: str, 
                          geofile: str,
                          depth_start: float,
                          depth_end: float,
                          resolution: float = 1.0,
                          first_image: int = None,
                          last_image: int = None,
                          verbose: int = 1,
                          normalization: str = None,
                          percent_to_process: float = 100.0,
                          wire_edges: str = 'l',
                          output_pixel_type: int = None,
                          memory_mb: int = 128,
                          distortion_map: str = None,
                          detector_number: int = 0,
                          wire_depths_file: str = None) -> Dict[str, Any]:
        """
        Run wire scan reconstruction.
        
        Args:
            infile: Base name of input image files
            outfile: Base name of output image files  
            geofile: Full path to geometry file
            depth_start: First depth in reconstruction range (microns)
            depth_end: Last depth in reconstruction range (microns)
            resolution: Depth resolution (microns), default 1.0
            first_image: Index to first input image file
            last_image: Index to last input image file
            verbose: Verbosity level (0-3), default 1
            normalization: Optional tag for normalization
            percent_to_process: Only process the N% brightest pixels, default 100
            wire_edges: Use 'l' (leading), 't' (trailing), or 'b' (both) edges
            output_pixel_type: Type of output pixel (0-7, WinView numbers)
            memory_mb: Memory limit in MiB, default 128
            distortion_map: Optional distortion map file
            detector_number: Detector number, default 0
            wire_depths_file: Optional file with depth corrections
            
        Returns:
            Dictionary with execution results:
                - 'success': Boolean indicating if execution succeeded
                - 'output_files': List of generated output files
                - 'log': Execution log output
                - 'error': Error message if failed
                - 'command': Command that was executed
        """
        
        # Build command line arguments
        cmd = [
            self.wirescan_executable,
            '--infile', infile,
            '--outfile', outfile,
            '--geofile', geofile,
            '--depth-start', str(depth_start),
            '--depth-end', str(depth_end),
            '--resolution', str(resolution),
            '--verbose', str(verbose),
            '--percent-to-process', str(percent_to_process),
            '--wire-edges', wire_edges,
            '--memory', str(memory_mb),
            '--detector_number', str(detector_number)
        ]
        
        # Add optional arguments
        if first_image is not None:
            cmd.extend(['--first-image', str(first_image)])
        if last_image is not None:
            cmd.extend(['--last-image', str(last_image)])
        if normalization:
            cmd.extend(['--normalization', normalization])
        if output_pixel_type is not None:
            cmd.extend(['--type-output-pixel', str(output_pixel_type)])
        if distortion_map:
            cmd.extend(['--distortion_map', distortion_map])
        if wire_depths_file:
            cmd.extend(['--wireDepths', wire_depths_file])
        
        return self._execute_reconstruction(cmd, outfile)
    
    def run_reconstruction_batch(self,
                               reconstruction_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run multiple reconstructions in sequence.
        
        Args:
            reconstruction_configs: List of configuration dictionaries for run_reconstruction()
            
        Returns:
            List of result dictionaries, one for each reconstruction
        """
        results = []
        
        for i, config in enumerate(reconstruction_configs):
            print(f"Running reconstruction {i+1}/{len(reconstruction_configs)}...")
            result = self.run_reconstruction(**config)
            results.append(result)
            
            if not result['success']:
                print(f"Warning: Reconstruction {i+1} failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def _execute_reconstruction(self, cmd: List[str], outfile_base: str) -> Dict[str, Any]:
        """Execute the reconstruction subprocess and return results."""
        
        try:
            # Execute the subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=7200  # 2 hour timeout for reconstruction
            )
            
            success = result.returncode == 0
            
            # Find output files
            output_files = []
            if success:
                output_dir = Path(outfile_base).parent
                if output_dir.exists():
                    # Look for files matching the output pattern
                    output_pattern = Path(outfile_base).name + "*"
                    output_files = [str(f) for f in output_dir.glob(output_pattern)]
            
            return {
                'success': success,
                'output_files': output_files,
                'log': result.stdout,
                'error': result.stderr if not success else None,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output_files': [],
                'log': '',
                'error': 'Process timed out after 2 hours',
                'return_code': -1,
                'command': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'success': False,
                'output_files': [],
                'log': '',
                'error': str(e),
                'return_code': -1,
                'command': ' '.join(cmd)
            }


def create_simple_reconstruction_config(
    input_base: str,
    output_base: str,
    geo_file: str,
    depth_start: float,
    depth_end: float,
    resolution: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a simple reconstruction configuration dictionary.
    
    Args:
        input_base: Base name/path for input image files
        output_base: Base name/path for output image files
        geo_file: Path to geometry file
        depth_start: Start depth in microns
        depth_end: End depth in microns  
        resolution: Depth resolution in microns
        **kwargs: Additional parameters for run_reconstruction()
        
    Returns:
        Configuration dictionary ready for use with run_reconstruction()
    """
    config = {
        'infile': input_base,
        'outfile': output_base,
        'geofile': geo_file,
        'depth_start': depth_start,
        'depth_end': depth_end,
        'resolution': resolution,
        'verbose': 1,
        'wire_edges': 'l',  # leading edge
        'percent_to_process': 100.0,
        'memory_mb': 128
    }
    
    # Override with user-provided values
    config.update(kwargs)
    
    return config


def create_depth_scan_batch(
    input_base: str,
    output_base: str,
    geo_file: str,
    depth_ranges: List[tuple],
    resolution: float = 1.0,
    **common_kwargs
) -> List[Dict[str, Any]]:
    """
    Create a batch of reconstruction configs for multiple depth ranges.
    
    Args:
        input_base: Base name for input files
        output_base: Base name for output files
        geo_file: Geometry file path
        depth_ranges: List of (start, end) depth tuples in microns
        resolution: Depth resolution in microns
        **common_kwargs: Common parameters for all reconstructions
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for i, (start, end) in enumerate(depth_ranges):
        # Create unique output name for each depth range
        range_output = f"{output_base}_depth_{start}_{end}_"
        
        config = create_simple_reconstruction_config(
            input_base=input_base,
            output_base=range_output,
            geo_file=geo_file,
            depth_start=start,
            depth_end=end,
            resolution=resolution,
            **common_kwargs
        )
        configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Simple CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Run WireScan reconstruction via subprocess interface")
    parser.add_argument('--infile', required=True, help="Base name of input image files")
    parser.add_argument('--outfile', required=True, help="Base name of output image files")
    parser.add_argument('--geofile', required=True, help="Path to geometry file")
    parser.add_argument('--depth-start', type=float, required=True, help="Start depth (microns)")
    parser.add_argument('--depth-end', type=float, required=True, help="End depth (microns)")
    parser.add_argument('--resolution', type=float, default=1.0, help="Depth resolution (microns)")
    parser.add_argument('--first-image', type=int, help="First image index")
    parser.add_argument('--last-image', type=int, help="Last image index")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity level (0-3)")
    parser.add_argument('--wire-edges', default='l', help="Wire edges: l, t, or b")
    parser.add_argument('--executable', help="Path to WireScan executable")
    
    args = parser.parse_args()
    
    # Create interface
    interface = WireScanRecon(args.executable)
    
    # Run reconstruction
    result = interface.run_reconstruction(
        infile=args.infile,
        outfile=args.outfile,
        geofile=args.geofile,
        depth_start=args.depth_start,
        depth_end=args.depth_end,
        resolution=args.resolution,
        first_image=args.first_image,
        last_image=args.last_image,
        verbose=args.verbose,
        wire_edges=args.wire_edges
    )
    
    # Print results
    if result['success']:
        print("SUCCESS: Reconstruction completed")
        print(f"Generated {len(result['output_files'])} output files")
        if result['output_files']:
            print("Output files:")
            for f in result['output_files'][:5]:  # Show first 5
                print(f"  {f}")
            if len(result['output_files']) > 5:
                print(f"  ... and {len(result['output_files']) - 5} more")
    else:
        print("FAILED: Reconstruction failed")
        print(f"Error: {result['error']}")
        print(f"Command: {result['command']}")
        
    exit(0 if result['success'] else 1)
