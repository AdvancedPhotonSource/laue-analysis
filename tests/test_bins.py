"""Test that packaged executables are accessible and exist."""

import os
import pytest
from pathlib import Path
from laueanalysis.indexing.lau_dataclasses.config import get_packaged_executable_path
from laueanalysis.reconstruct.wirescan_interface import WireScanReconstructionInterface


@pytest.mark.parametrize("program", ['euler', 'peaksearch', 'pix2qs'])
def test_get_packaged_executable_path_valid_programs(program):
    """Test that get_packaged_executable_path works for valid program names."""
    path = get_packaged_executable_path(program)
    assert isinstance(path, str)
    assert os.path.exists(path), f"Executable {program} does not exist at {path}"
    assert os.access(path, os.X_OK), f"Executable {program} is not executable at {path}"


def test_get_packaged_executable_path_invalid_program():
    """Test that get_packaged_executable_path raises FileNotFoundError for invalid program names."""
    with pytest.raises(FileNotFoundError):
        get_packaged_executable_path('nonexistent_program')


def test_wirescan_executable_exists():
    """Test that the WireScan reconstruction executable exists and is accessible."""
    interface = WireScanReconstructionInterface()
    executable_path = interface.wirescan_executable
    
    assert isinstance(executable_path, str)
    assert os.path.exists(executable_path), f"WireScan executable does not exist at {executable_path}"
    assert os.access(executable_path, os.X_OK), f"WireScan executable is not executable at {executable_path}"


def test_wirescan_executable_validation():
    """Test that WireScanReconstructionInterface properly validates the executable."""
    # Test with a non-existent executable path
    with pytest.raises(RuntimeError, match="not found|not working"):
        WireScanReconstructionInterface(wirescan_executable="/nonexistent/path/WireScan")


def test_wirescan_find_executable_function():
    """Test the _find_wirescan_executable method directly."""
    interface = WireScanReconstructionInterface.__new__(WireScanReconstructionInterface)
    
    # Test the finding logic
    executable_path = interface._find_wirescan_executable()
    assert isinstance(executable_path, str)
    
    # The executable should exist and be executable after proper installation
    assert os.path.exists(executable_path), f"Found WireScan executable does not exist at {executable_path}"
    assert os.access(executable_path, os.X_OK), f"Found WireScan executable is not executable at {executable_path}"


def test_wirescan_package_bin_location():
    """Test that the package bin directory structure exists for WireScan."""
    from importlib import resources
    bin_files = resources.files('laueanalysis.reconstruct.bin')
    
    # The bin directory should exist after installation
    assert bin_files.is_dir(), "laueanalysis.reconstruct.bin directory should exist"
    
    # Check that reconstructN_cpu executable exists in the package
    wirescan_exe = bin_files / 'reconstructN_cpu'
    assert wirescan_exe.is_file(), "reconstructN_cpu executable should exist in package after installation"
    
    # It should be executable
    assert os.access(str(wirescan_exe), os.X_OK), "Packaged reconstructN_cpu should be executable"