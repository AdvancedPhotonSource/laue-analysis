#!/usr/bin/env python3

import unittest
import os
import tempfile
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import laueindexing.pyLaueGo as pyLaueGo


class TestSmokeRun(unittest.TestCase):
    """Smoke test to verify that the program can run using the test config."""

    def setUp(self):
        """Set up test fixtures."""
        # Path to the test config file
        self.config_path = os.path.join(
            "tests",
            "data", 
            "test_config.yaml"
        )
        # Ensure the config file exists
        self.assertTrue(os.path.exists(self.config_path), f"Config file {self.config_path} not found")
        
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Load the config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update the output path to use our temporary directory
        self.config['outputFolder'] = self.temp_dir.name

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_single_process(self):
        py_laue_go = pyLaueGo.PyLaueGo(config=self.config)

        py_laue_go.run_on_process()

        self._check_output()

    def test_mpi(self):
        """Test running the indexing with MPI using subprocess."""
        # Create a temporary config file for the MPI run
        temp_config_path = os.path.join(self.temp_dir.name, 'mpi_test_config.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Run the indexing script using mpiexec through subprocess
        cmd = ['mpiexec', '-n', '2', '--allow-run-as-root', 'python', '-m', 'laueindexing.pyLaueGo', temp_config_path]
        
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise an exception on non-zero exit
        )
        
        # Check that the process ran successfully
        self.assertEqual(process.returncode, 0, 
                         f"MPI process failed with return code {process.returncode}\n"
                         f"STDOUT: {process.stdout}\n"
                         f"STDERR: {process.stderr}")
        
        # Check the output files
        self._check_output()

    
    def _check_output(self):
        """Verify that the output files and directory structure were created correctly."""
        # Check that the output directory structure was created
        output_dirs = ['peaks', 'p2q', 'index', 'error']
        for dir_name in output_dirs:
            dir_path = os.path.join(self.temp_dir.name, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Output directory {dir_name} not created")
            self.assertTrue(os.path.isdir(dir_path), f"Output path {dir_name} is not a directory")
        
        # Check that the main XML output file was created
        xml_output = os.path.join(self.temp_dir.name, f"{self.config['filenamePrefix']}indexed.xml")
        self.assertTrue(os.path.exists(xml_output), "XML output file not created")
        
        # Check the content of the XML file to ensure it has the expected structure
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_output)
            root = tree.getroot()
            self.assertEqual(root.tag, "AllSteps", "Root element of XML should be 'AllSteps'")
            self.assertTrue(len(root) > 0, "XML should contain at least one step element")
        except Exception as e:
            self.fail(f"Failed to parse XML output file: {str(e)}")
        
        # Check processing output directory contents
        # Since we're doing a smoke test with test config, we should expect files 
        # for each input that was processed based on scanPointStart/End and depthRangeStart/End
        scan_points = range(int(self.config.get('scanPointStart', 0)), 
                            int(self.config.get('scanPointEnd', 1)))
        depth_range = range(int(self.config.get('depthRangeStart', 0)), 
                            int(self.config.get('depthRangeEnd', 1)))
        
        # Check for expected output files for each input file
        for scan in scan_points:
            for depth in depth_range:
                # File naming pattern from the PyLaueGo class
                file_base = f"{self.config['filenamePrefix']}{scan}_{depth}"
                
                # Check peaks output
                peaks_file = os.path.join(self.temp_dir.name, 'peaks', f"peaks_{file_base}.txt")
                self.assertTrue(os.path.exists(peaks_file), f"Peaks file not found: {peaks_file}")
                
                # Read peaks file to determine if any peaks were found
                with open(peaks_file, 'r', encoding='windows-1252', errors='ignore') as f:
                    peaks_content = f.read()
                    peaks_found = 'Npeaks' in peaks_content
                
                # If peaks were found, there should be p2q output
                if peaks_found:
                    p2q_file = os.path.join(self.temp_dir.name, 'p2q', f"p2q_{file_base}.txt")
                    self.assertTrue(os.path.exists(p2q_file), f"P2Q file not found: {p2q_file}")
                    
                    # Check if there are enough peaks for indexing (needs at least 2)
                    with open(p2q_file, 'r', encoding='windows-1252', errors='ignore') as f:
                        p2q_content = f.read()
                        enough_peaks_for_indexing = p2q_content.count('\n') > 50  # Rough estimate
                    
                    # If enough peaks were found, there should be indexing output
                    if enough_peaks_for_indexing:
                        index_file = os.path.join(self.temp_dir.name, 'index', f"index_{file_base}.txt")
                        self.assertTrue(os.path.exists(index_file), f"Index file not found: {index_file}")

if __name__ == '__main__':
    unittest.main()
