#!/usr/bin/env python3

"""Tests for the new functional index interface."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from argparse import Namespace
import yaml
from pathlib import Path

from laueanalysis.indexing import index, IndexingResult
from laueanalysis.indexing.lau_dataclasses.config import LaueConfig


@pytest.fixture
def test_config_data():
    """Test configuration data fixture."""
    return {
        'outputFolder': '/tmp/test_output',
        'filefolder': '/tmp/test_input', 
        'pathbins': '/tmp/test_bins',
        'filenamePrefix': 'test_',
        'boxsize': 5,
        'maxRfactor': 0.5,
        'min_size': 2,
        'min_separation': 10,
        'threshold': 100,
        'peakShape': 'G',
        'max_peaks': 50,
        'maskFile': '',
        'smooth': False,
        'geoFile': '/tmp/test_geo.xml',
        'crystFile': '/tmp/test_cryst.xml',
        'indexKeVmaxCalc': 30.0,
        'indexKeVmaxTest': 35.0,
        'indexAngleTolerance': 0.1,
        'indexCone': 72.0,
        'indexH': 0,
        'indexK': 0,
        'indexL': 1
    }


@pytest.fixture
def test_config(test_config_data):
    """LaueConfig instance fixture."""
    return LaueConfig.from_dict(test_config_data)


@pytest.fixture
def temp_image_file():
    """Create a temporary test image file."""
    image_fd, image_path = tempfile.mkstemp(suffix='.h5')
    try:
        # Write some dummy data to the file
        with open(image_path, 'wb') as f:
            f.write(b'dummy h5 data for testing')
        yield image_path
    finally:
        os.close(image_fd)
        if os.path.exists(image_path):
            os.unlink(image_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


def test_index_function_with_mocked_executables(test_config, temp_image_file, temp_output_dir):
    """Test the index function with mocked executable calls."""
    
    # Mock the executable finding and running
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd, \
         patch('laueanalysis.indexing.index._parse_peaks_output') as mock_parse_peaks, \
         patch('laueanalysis.indexing.index._parse_indexing_output') as mock_parse_index:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs', 
            'euler': '/mock/euler'
        }
        
        # Mock successful execution for all steps
        mock_run_cmd.return_value = (True, "Mock output", "", 0)
        mock_parse_peaks.return_value = 10  # Found 10 peaks
        mock_parse_index.return_value = 8   # Indexed 8 reflections
        
        # Mock output files being created
        def mock_glob(pattern):
            if 'peaks_' in str(pattern):
                return [Path(temp_output_dir) / 'peaks' / 'peaks_test.txt']
            elif 'p2q_' in str(pattern):
                return [Path(temp_output_dir) / 'p2q' / 'p2q_test.txt']
            elif 'index_' in str(pattern):
                return [Path(temp_output_dir) / 'index' / 'index_test.txt']
            return []
        
        with patch('pathlib.Path.glob', side_effect=mock_glob):
            # Run the index function
            result = index(
                input_image=temp_image_file,
                output_dir=temp_output_dir,
                geo_file=test_config.geoFile,
                crystal_file=test_config.crystFile,
                config=test_config
            )
        
        # Verify the result
        assert isinstance(result, IndexingResult)
        assert result.success is True
        assert result.n_peaks_found == 10
        assert result.n_indexed == 8
        assert 'peaks' in result.output_files
        assert 'p2q' in result.output_files
        assert 'index' in result.output_files
        assert len(result.command_history) == 3  # peaksearch, p2q, indexing
        
        # Verify the commands were called correctly
        assert mock_run_cmd.call_count == 3
        
        # Check peaksearch command
        peaksearch_call = mock_run_cmd.call_args_list[0][0][0]
        assert peaksearch_call[0] == '/mock/peaksearch'
        assert '-b' in peaksearch_call and str(test_config.boxsize) in peaksearch_call
        assert '-R' in peaksearch_call and str(test_config.maxRfactor) in peaksearch_call
        assert '-p' in peaksearch_call and test_config.peakShape in peaksearch_call
        
        # Check p2q command  
        p2q_call = mock_run_cmd.call_args_list[1][0][0]
        assert p2q_call[0] == '/mock/pix2qs'
        assert '-g' in p2q_call and test_config.geoFile in p2q_call
        assert '-x' in p2q_call and test_config.crystFile in p2q_call
        
        # Check indexing command
        index_call = mock_run_cmd.call_args_list[2][0][0]
        assert index_call[0] == '/mock/euler'
        assert '-q' in index_call
        assert '-k' in index_call and str(test_config.indexKeVmaxCalc) in index_call
        assert '-t' in index_call and str(test_config.indexKeVmaxTest) in index_call


def test_index_function_no_peaks_found(test_config, temp_image_file, temp_output_dir):
    """Test the index function when no peaks are found."""
    
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd, \
         patch('laueanalysis.indexing.index._parse_peaks_output') as mock_parse_peaks:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs',
            'euler': '/mock/euler'
        }
        
        mock_run_cmd.return_value = (True, "Mock output", "", 0)
        mock_parse_peaks.return_value = 0  # No peaks found
        
        # Mock peaks file exists but is empty
        def mock_glob(pattern):
            if 'peaks_' in str(pattern):
                return [Path(temp_output_dir) / 'peaks' / 'peaks_test.txt']
            return []
        
        with patch('pathlib.Path.glob', side_effect=mock_glob):
            result = index(
                input_image=temp_image_file,
                output_dir=temp_output_dir,
                geo_file=test_config.geoFile,
                crystal_file=test_config.crystFile,
                config=test_config
            )
        
        # Should still succeed but skip p2q and indexing
        assert result.success is True
        assert result.n_peaks_found == 0
        assert result.n_indexed == 0
        assert 'peaks' in result.output_files
        assert 'p2q' not in result.output_files
        assert 'index' not in result.output_files
        assert len(result.command_history) == 1  # Only peaksearch
        assert "No peaks found, skipping p2q and indexing steps" in result.log


def test_index_function_insufficient_peaks_for_indexing(test_config, temp_image_file, temp_output_dir):
    """Test the index function when only 1 peak is found (insufficient for indexing)."""
    
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd, \
         patch('laueanalysis.indexing.index._parse_peaks_output') as mock_parse_peaks:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs',
            'euler': '/mock/euler'
        }
        
        mock_run_cmd.return_value = (True, "Mock output", "", 0)
        mock_parse_peaks.return_value = 1  # Only 1 peak found
        
        # Mock output files being created for peaks and p2q, but not indexing
        def mock_glob(pattern):
            if 'peaks_' in str(pattern):
                return [Path(temp_output_dir) / 'peaks' / 'peaks_test.txt']
            elif 'p2q_' in str(pattern):
                return [Path(temp_output_dir) / 'p2q' / 'p2q_test.txt']
            return []
        
        with patch('pathlib.Path.glob', side_effect=mock_glob):
            result = index(
                input_image=temp_image_file,
                output_dir=temp_output_dir,
                geo_file=test_config.geoFile,
                crystal_file=test_config.crystFile,
                config=test_config
            )
        
        # Should succeed, run p2q, but skip indexing
        assert result.success is True
        assert result.n_peaks_found == 1
        assert result.n_indexed == 0
        assert 'peaks' in result.output_files
        assert 'p2q' in result.output_files
        assert 'index' not in result.output_files
        assert len(result.command_history) == 2  # peaksearch and p2q
        assert "need at least 2 for indexing" in result.log


def test_index_function_with_errors_continues_processing(test_config, temp_image_file, temp_output_dir):
    """Test that the index function continues processing even when subprocess have errors."""
    
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd, \
         patch('laueanalysis.indexing.index._parse_peaks_output') as mock_parse_peaks:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs',
            'euler': '/mock/euler'
        }
        
        # Mock peaksearch fails but still produces output, p2q fails
        mock_run_cmd.side_effect = [
            (False, "Peak search output", "Peak search error", 1),  # peaksearch fails but produces file
            (False, "P2Q output", "P2Q error", 1),  # p2q fails
        ]
        mock_parse_peaks.return_value = 5  # Found 5 peaks
        
        # Mock peaks file exists (despite peaksearch error)
        def mock_glob(pattern):
            if 'peaks_' in str(pattern):
                return [Path(temp_output_dir) / 'peaks' / 'peaks_test.txt']
            return []
        
        with patch('pathlib.Path.glob', side_effect=mock_glob):
            result = index(
                input_image=temp_image_file,
                output_dir=temp_output_dir,
                geo_file=test_config.geoFile,
                crystal_file=test_config.crystFile,
                config=test_config
            )
        
        # Should still succeed (matches pyLaueGo behavior)
        assert result.success is True
        assert result.n_peaks_found == 5
        assert result.n_indexed == 0
        assert 'peaks' in result.output_files
        assert 'p2q' not in result.output_files
        assert "Peak search had errors" in result.log
        assert "P2Q conversion failed" in result.log


def test_index_function_default_config():
    """Test the index function with default config when none provided."""
    
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd, \
         patch('laueanalysis.indexing.index._parse_peaks_output') as mock_parse_peaks:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs',
            'euler': '/mock/euler'
        }
        
        mock_run_cmd.return_value = (True, "Mock output", "", 0)
        mock_parse_peaks.return_value = 0
        
        # Mock output files
        def mock_glob(pattern):
            if 'peaks_' in str(pattern):
                return [Path('/tmp/output/peaks/peaks_test.txt')]
            return []
        
        with patch('pathlib.Path.glob', side_effect=mock_glob):
            result = index(
                input_image='/tmp/test.h5',
                output_dir='/tmp/output',
                geo_file='/tmp/geo.xml',
                crystal_file='/tmp/crystal.xml'
                # No config provided - should use defaults
            )
        
        # Should succeed with default config
        assert result.success is True
        assert result.config is not None
        assert result.config.geoFile == '/tmp/geo.xml'
        assert result.config.crystFile == '/tmp/crystal.xml'
        
        # Check that default values were used in the command
        peaksearch_call = mock_run_cmd.call_args_list[0][0][0]
        assert '-b' in peaksearch_call and '5' in peaksearch_call  # default boxsize
        assert '-p' in peaksearch_call and 'L' in peaksearch_call  # default peakShape


def test_index_function_complete_failure():
    """Test the index function when peaksearch completely fails."""
    
    with patch('laueanalysis.indexing.index._find_executables') as mock_find_exes, \
         patch('laueanalysis.indexing.index._run_command') as mock_run_cmd:
        
        # Setup mocks
        mock_find_exes.return_value = {
            'peaksearch': '/mock/peaksearch',
            'pix2qs': '/mock/pix2qs',
            'euler': '/mock/euler'
        }
        
        mock_run_cmd.return_value = (False, "", "Command failed", 1)
        
        # Mock no output files created
        with patch('pathlib.Path.glob', return_value=[]):
            result = index(
                input_image='/tmp/test.h5',
                output_dir='/tmp/output',
                geo_file='/tmp/geo.xml',
                crystal_file='/tmp/crystal.xml'
            )
        
        # Should fail completely
        assert result.success is False
        assert result.n_peaks_found == 0
        assert result.n_indexed == 0
        assert result.output_files == {}
        assert "peak search completely failed" in result.error
