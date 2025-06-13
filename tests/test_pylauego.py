#!/usr/bin/env python3

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from argparse import Namespace
import yaml
from laueanalysis.indexing.pyLaueGo import PyLaueGo


class TestPyLaueGo(unittest.TestCase):
    """Tests for the PyLaueGo class in the laue_indexing package."""

    def setUp(self):
        """Set up test fixtures."""
        self.pylauego = PyLaueGo()
        
        # Create a test config file
        self.config_data = {
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
        
        # Create a temporary config file
        self.config_fd, self.config_path = tempfile.mkstemp(suffix='.yml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.close(self.config_fd)
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)


    @patch('laueanalysis.indexing.pyLaueGo.os.path.isfile')
    def test_get_input_files_names_list(self, mock_isfile):
        """Test getting list of input file names."""
        mock_isfile.return_value = True
        
        # Create args with required attributes
        args = Namespace(
            filefolder=self.config_data['filefolder'],
            filenamePrefix=self.config_data['filenamePrefix']
        )
        
        # Test with scanPoint specified
        scan_point = [1, 2, 3]
        depth_range = None

        self.pylauego._config = args
        
        # Mock os.walk to return test files
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [(args.filefolder, [], ['test_1.h5', 'test_2.h5', 'test_3.h5'])]
            
            files = self.pylauego.getInputFileNamesList(depth_range, scan_point)
            
            # Should return files matching the scanPoint
            self.assertEqual(len(files), 3)
            self.assertIn('test_1.h5', files)
            self.assertIn('test_2.h5', files)
            self.assertIn('test_3.h5', files)

    @patch('laueanalysis.indexing.pyLaueGo.sub.check_output')
    def test_run_cmd_and_check_output(self, mock_check_output):
        """Test running command and checking output."""
        # Setup mock
        mock_check_output.return_value = b"Command output"
        
        # Setup PyLaueGo instance with an error log
        self.pylauego.errorLog = tempfile.mktemp()
        
        # Run a test command
        cmd = ["echo", "test"]
        result = self.pylauego.runCmdAndCheckOutput(cmd)
        
        # Check that check_output was called with the right arguments
        mock_check_output.assert_called_once_with(cmd, stderr=-2)  # -2 is subprocess.STDOUT
        
        # Clean up
        if os.path.exists(self.pylauego.errorLog):
            os.unlink(self.pylauego.errorLog)


if __name__ == '__main__':
    unittest.main()