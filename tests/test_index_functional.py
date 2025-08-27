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


def test_index_function_with_mocked_executables(test_config_data, temp_image_file, temp_output_dir):
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
        
        # Create expected output files with deterministic names
        base = Path(temp_image_file).stem
        peaks_dir = Path(temp_output_dir) / 'peaks'
        p2q_dir = Path(temp_output_dir) / 'p2q'
        index_dir = Path(temp_output_dir) / 'index'
        peaks_dir.mkdir(parents=True, exist_ok=True)
        p2q_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        (peaks_dir / f'peaks_{base}.txt').write_text('')
        (p2q_dir / f'p2q_{base}.txt').write_text('')
        (index_dir / f'index_{base}.txt').write_text('')
        
        # Run the index function with parameters
        result = index(
            input_image=temp_image_file,
            output_dir=temp_output_dir,
            geo_file=test_config_data['geoFile'],
            crystal_file=test_config_data['crystFile'],
            boxsize=test_config_data['boxsize'],
            max_rfactor=test_config_data['maxRfactor'],
            min_size=test_config_data['min_size'],
            min_separation=test_config_data['min_separation'],
            threshold=test_config_data['threshold'],
            peak_shape=test_config_data['peakShape'],
            max_peaks=test_config_data['max_peaks'],
            smooth=test_config_data['smooth'],
            index_kev_max_calc=test_config_data['indexKeVmaxCalc'],
            index_kev_max_test=test_config_data['indexKeVmaxTest'],
            index_angle_tolerance=test_config_data['indexAngleTolerance'],
            index_cone=test_config_data['indexCone'],
            index_h=test_config_data['indexH'],
            index_k=test_config_data['indexK'],
            index_l=test_config_data['indexL']
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
        assert '-b' in peaksearch_call and str(test_config_data['boxsize']) in peaksearch_call
        assert '-R' in peaksearch_call and str(test_config_data['maxRfactor']) in peaksearch_call
        assert '-p' in peaksearch_call and test_config_data['peakShape'] in peaksearch_call
        
        # Check p2q command  
        p2q_call = mock_run_cmd.call_args_list[1][0][0]
        assert p2q_call[0] == '/mock/pix2qs'
        assert '-g' in p2q_call and test_config_data['geoFile'] in p2q_call
        assert '-x' in p2q_call and test_config_data['crystFile'] in p2q_call
        
        # Check indexing command
        index_call = mock_run_cmd.call_args_list[2][0][0]
        assert index_call[0] == '/mock/euler'
        assert '-q' in index_call
        assert '-k' in index_call and str(test_config_data['indexKeVmaxCalc']) in index_call
        assert '-t' in index_call and str(test_config_data['indexKeVmaxTest']) in index_call


def test_index_function_no_peaks_found(temp_image_file, temp_output_dir):
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
        
        # Create expected peaks file (empty)
        base = Path(temp_image_file).stem
        peaks_dir = Path(temp_output_dir) / 'peaks'
        peaks_dir.mkdir(parents=True, exist_ok=True)
        (peaks_dir / f'peaks_{base}.txt').write_text('')
        
        result = index(
            input_image=temp_image_file,
            output_dir=temp_output_dir,
            geo_file='/tmp/test_geo.xml',
            crystal_file='/tmp/test_cryst.xml'
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


def test_index_function_insufficient_peaks_for_indexing(temp_image_file, temp_output_dir):
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
        
        # Create expected peaks and p2q files
        base = Path(temp_image_file).stem
        peaks_dir = Path(temp_output_dir) / 'peaks'
        p2q_dir = Path(temp_output_dir) / 'p2q'
        peaks_dir.mkdir(parents=True, exist_ok=True)
        p2q_dir.mkdir(parents=True, exist_ok=True)
        (peaks_dir / f'peaks_{base}.txt').write_text('')
        (p2q_dir / f'p2q_{base}.txt').write_text('')
        
        result = index(
            input_image=temp_image_file,
            output_dir=temp_output_dir,
            geo_file='/tmp/test_geo.xml',
            crystal_file='/tmp/test_cryst.xml'
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


def test_index_function_with_errors_continues_processing(temp_image_file, temp_output_dir):
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
        
        # Create expected peaks file (despite peaksearch error)
        base = Path(temp_image_file).stem
        peaks_dir = Path(temp_output_dir) / 'peaks'
        peaks_dir.mkdir(parents=True, exist_ok=True)
        (peaks_dir / f'peaks_{base}.txt').write_text('')
        
        result = index(
            input_image=temp_image_file,
            output_dir=temp_output_dir,
            geo_file='/tmp/test_geo.xml',
            crystal_file='/tmp/test_cryst.xml'
        )
        
        # Should still succeed (graceful degradation)
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
        
        # Create expected output file for defaults
        Path('/tmp/output/peaks').mkdir(parents=True, exist_ok=True)
        Path('/tmp/output/peaks/peaks_test.txt').write_text('')
        
        result = index(
            input_image='/tmp/test.h5',
            output_dir='/tmp/output',
            geo_file='/tmp/geo.xml',
            crystal_file='/tmp/crystal.xml'
            # No config provided - should use defaults
        )
        
        # Should succeed with defaults
        assert result.success is True
        assert result.config is None  # No config object anymore
        
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
        
        # Use a unique temporary output directory and do not create any peaks files
        with tempfile.TemporaryDirectory() as temp_dir:
            result = index(
                input_image='/tmp/test.h5',
                output_dir=temp_dir,
                geo_file='/tmp/geo.xml',
                crystal_file='/tmp/crystal.xml'
            )
        
        # Should fail completely
        assert result.success is False
        assert result.n_peaks_found == 0
        assert result.n_indexed == 0
        assert result.output_files == {}
        assert "peak search completely failed" in result.error
