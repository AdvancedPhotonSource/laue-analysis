
import unittest
import os
from pathlib import Path
from laueindexing.lau_dataclasses.config import get_packaged_executable_path


class TestPackagedExecutables(unittest.TestCase):
    """Test that packaged executables are accessible and exist."""

    def test_get_packaged_executable_path_valid_programs(self):
        """Test that get_packaged_executable_path works for valid program names."""
        programs = ['euler', 'peaksearch', 'pix2qs']
        
        for program in programs:
            with self.subTest(program=program):
                path = get_packaged_executable_path(program)
                self.assertIsInstance(path, str)
                self.assertTrue(os.path.exists(path), f"Executable {program} does not exist at {path}")
                self.assertTrue(os.access(path, os.X_OK), f"Executable {program} is not executable at {path}")

    def test_get_packaged_executable_path_invalid_program(self):
        """Test that get_packaged_executable_path raises FileNotFoundError for invalid program names."""
        with self.assertRaises(FileNotFoundError):
            get_packaged_executable_path('nonexistent_program')


if __name__ == '__main__':
    unittest.main()