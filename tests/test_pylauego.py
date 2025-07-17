#!/usr/bin/env python3

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from argparse import Namespace
import yaml
from laueanalysis.indexing.pyLaueGo import PyLaueGo


@pytest.fixture
def test_config_data():
    """Test configuration data fixture."""
    return {
        'outputFolder': '/tmp/test_output',
        'filefolder': '/tmp/test_input',
        'pathbins': '/tmp/test_bins',
        'filenamePrefix': 'test_',
        'boxsize': '5',
        'maxRfactor': '0.5',
        'min_size': '2',
        'min_separation': '10',
        'threshold': '100',
        'peakShape': 'G',
        'max_peaks': '50',
        'maskFile': '',
        'smooth': 'False',
        'geoFile': '/tmp/test_geo.xml',
        'crystFile': '/tmp/test_cryst.xml',
        'indexKeVmaxCalc': '30.0',
        'indexKeVmaxTest': '35.0',
        'indexAngleTolerance': '0.1',
        'indexCone': '72.0',
        'indexH': '0',
        'indexK': '0',
        'indexL': '1'
    }


@pytest.fixture
def temp_config_file(test_config_data):
    """Create a temporary config file for testing."""
    config_fd, config_path = tempfile.mkstemp(suffix='.yml')
    try:
        with open(config_path, 'w') as f:
            yaml.dump(test_config_data, f)
        yield config_fd, config_path
    finally:
        os.close(config_fd)
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.fixture
def pylauego():
    """PyLaueGo instance fixture."""
    return PyLaueGo()


def test_get_input_files_names_list(pylauego, test_config_data):
    """Test getting list of input file names."""
    with patch('laueanalysis.indexing.pyLaueGo.os.path.isfile') as mock_isfile:
        mock_isfile.return_value = True
        
        # Create args with required attributes
        args = Namespace(
            filefolder=test_config_data['filefolder'],
            filenamePrefix=test_config_data['filenamePrefix']
        )
        
        # Test with scanPoint specified
        scan_point = [1, 2, 3]
        depth_range = None

        pylauego._config = args
        
        # Mock os.walk to return test files
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [(args.filefolder, [], ['test_1.h5', 'test_2.h5', 'test_3.h5'])]
            
            files = pylauego.getInputFileNamesList(depth_range, scan_point)
            
            # Should return files matching the scanPoint
            assert len(files) == 3
            assert 'test_1.h5' in files
            assert 'test_2.h5' in files
            assert 'test_3.h5' in files


def test_run_cmd_and_check_output(pylauego):
    """Test running command and checking output."""
    with patch('laueanalysis.indexing.pyLaueGo.sub.check_output') as mock_check_output:
        # Setup mock
        mock_check_output.return_value = b"Command output"
        
        # Setup PyLaueGo instance with an error log and config
        pylauego.errorLog = tempfile.mktemp()
        # Create a mock config with peaksearchPath
        from unittest.mock import MagicMock
        pylauego._config = MagicMock()
        pylauego._config.peaksearchPath = '/path/to/peaksearch'
        
        # Run a test command
        cmd = ["echo", "test"]
        result = pylauego.runCmdAndCheckOutput(cmd)
        
        # Check that check_output was called with the right arguments
        # The new implementation no longer adds an env parameter (RPATH handles library paths)
        mock_check_output.assert_called_once()
        call_args = mock_check_output.call_args
        assert call_args[0][0] == cmd
        assert call_args[1]['stderr'] == -2  # -2 is subprocess.STDOUT
        # env parameter should NOT be present anymore
        assert 'env' not in call_args[1]
        
        # Clean up
        if os.path.exists(pylauego.errorLog):
            os.unlink(pylauego.errorLog)
