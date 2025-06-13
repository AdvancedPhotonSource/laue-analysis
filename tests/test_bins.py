
"""Test that packaged executables are accessible and exist."""

import os
import pytest
from pathlib import Path
from laueanalysis.indexing.lau_dataclasses.config import get_packaged_executable_path


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