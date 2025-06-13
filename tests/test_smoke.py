#!/usr/bin/env python3

"""Smoke test to verify that the program can run using the test config."""

import os
import tempfile
import yaml
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import laueanalysis.indexing.pyLaueGo as pyLaueGo


@pytest.fixture
def config_path():
    """Path to the test config file."""
    path = os.path.join("tests", "data", "test_config.yaml")
    assert os.path.exists(path), f"Config file {path} not found"
    return path


@pytest.fixture
def test_config(config_path):
    """Load and prepare test configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create temporary directory for output
    temp_dir = tempfile.TemporaryDirectory()
    config['outputFolder'] = temp_dir.name
    
    yield config, temp_dir
    
    # Cleanup
    temp_dir.cleanup()


def test_single_process(test_config):
    """Test single process execution."""
    config, temp_dir = test_config
    py_laue_go = pyLaueGo.PyLaueGo(config=config)
    
    py_laue_go.run_on_process()
    
    _check_output(config, temp_dir.name)


def test_mpi(test_config):
    """Test running the indexing with MPI using subprocess."""
    config, temp_dir = test_config
    
    # Create a temporary config file for the MPI run
    temp_config_path = os.path.join(temp_dir.name, 'mpi_test_config.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run the indexing script using mpiexec through subprocess
    cmd = ['mpiexec', '-n', '2', '--allow-run-as-root', 'python', '-m', 'laueanalysis.indexing.mpi_runner', temp_config_path]
    
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False  # Don't raise an exception on non-zero exit
    )
    
    # Check that the process ran successfully
    assert process.returncode == 0, (
        f"MPI process failed with return code {process.returncode}\n"
        f"STDOUT: {process.stdout}\n"
        f"STDERR: {process.stderr}"
    )
    
    # Check the output files
    _check_output(config, temp_dir.name)


def _check_output(config, temp_dir_name):
    """Verify that the output files and directory structure were created correctly."""
    # Check that the output directory structure was created
    output_dirs = ['peaks', 'p2q', 'index', 'error']
    for dir_name in output_dirs:
        dir_path = os.path.join(temp_dir_name, dir_name)
        assert os.path.exists(dir_path), f"Output directory {dir_name} not created"
        assert os.path.isdir(dir_path), f"Output path {dir_name} is not a directory"
    
    # Check that the main XML output file was created
    xml_output = os.path.join(temp_dir_name, f"{config['filenamePrefix']}indexed.xml")
    assert os.path.exists(xml_output), "XML output file not created"
    
    # Check the content of the XML file to ensure it has the expected structure
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_output)
        root = tree.getroot()
        assert root.tag == "AllSteps", "Root element of XML should be 'AllSteps'"
        assert len(root) > 0, "XML should contain at least one step element"
    except Exception as e:
        pytest.fail(f"Failed to parse XML output file: {str(e)}")
    
    # Check processing output directory contents
    # Since we're doing a smoke test with test config, we should expect files 
    # for each input that was processed based on scanPointStart/End and depthRangeStart/End
    scan_points = range(int(config.get('scanPointStart', 0)), 
                        int(config.get('scanPointEnd', 1)))
    depth_range = range(int(config.get('depthRangeStart', 0)), 
                        int(config.get('depthRangeEnd', 1)))
    
    # Check for expected output files for each input file
    for scan in scan_points:
        for depth in depth_range:
            # File naming pattern from the PyLaueGo class
            file_base = f"{config['filenamePrefix']}{scan}_{depth}"
            
            # Check peaks output
            peaks_file = os.path.join(temp_dir_name, 'peaks', f"peaks_{file_base}.txt")
            assert os.path.exists(peaks_file), f"Peaks file not found: {peaks_file}"
            
            # Read peaks file to determine if any peaks were found
            with open(peaks_file, 'r', encoding='windows-1252', errors='ignore') as f:
                peaks_content = f.read()
                peaks_found = 'Npeaks' in peaks_content
            
            # If peaks were found, there should be p2q output
            if peaks_found:
                p2q_file = os.path.join(temp_dir_name, 'p2q', f"p2q_{file_base}.txt")
                assert os.path.exists(p2q_file), f"P2Q file not found: {p2q_file}"
                
                # Check if there are enough peaks for indexing (needs at least 2)
                with open(p2q_file, 'r', encoding='windows-1252', errors='ignore') as f:
                    p2q_content = f.read()
                    enough_peaks_for_indexing = p2q_content.count('\n') > 50  # Rough estimate
                
                # If enough peaks were found, there should be indexing output
                if enough_peaks_for_indexing:
                    index_file = os.path.join(temp_dir_name, 'index', f"index_{file_base}.txt")
                    assert os.path.exists(index_file), f"Index file not found: {index_file}"
