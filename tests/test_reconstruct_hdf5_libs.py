"""Test HDF5 library handling for reconstruction binaries."""

import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

from laueanalysis.reconstruct.reconstruct import (
    _find_hdf5_lib_path,
    _validate_executable,
    _execute_reconstruction,
    ReconstructionResult
)


class TestHDF5LibraryHandling:
    """Test HDF5 library path handling."""
    
    def test_find_hdf5_lib_path_conda(self):
        """Test finding HDF5 libraries in conda environment."""
        with patch.dict(os.environ, {'CONDA_PREFIX': '/fake/conda/env'}):
            with patch('pathlib.Path.exists') as mock_exists:
                with patch('pathlib.Path.glob') as mock_glob:
                    # Mock that lib directory exists and contains HDF5 libraries
                    mock_exists.return_value = True
                    mock_glob.return_value = [Path('/fake/conda/env/lib/libhdf5.so.103')]
                    
                    result = _find_hdf5_lib_path()
                    assert result == '/fake/conda/env/lib'
    
    def test_find_hdf5_lib_path_h5cc(self):
        """Test finding HDF5 libraries via h5cc."""
        # Clear conda environment
        with patch.dict(os.environ, {'CONDA_PREFIX': ''}, clear=True):
            with patch('subprocess.run') as mock_run:
                # Mock h5cc output
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='-I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5'
                )
                
                result = _find_hdf5_lib_path()
                assert result == '/usr/lib/x86_64-linux-gnu/hdf5/serial'
    
    def test_find_hdf5_lib_path_not_found(self):
        """Test when HDF5 libraries cannot be found."""
        with patch.dict(os.environ, {'CONDA_PREFIX': ''}, clear=True):
            with patch('subprocess.run', side_effect=FileNotFoundError):
                result = _find_hdf5_lib_path()
                assert result is None
    
    def test_validate_executable_with_hdf5_libs(self):
        """Test executable validation sets up HDF5 library paths."""
        with patch('laueanalysis.reconstruct.reconstruct._find_hdf5_lib_path') as mock_find:
            mock_find.return_value = '/path/to/hdf5/lib'
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                
                _validate_executable('/path/to/exe')
                
                # Check that subprocess was called with proper environment
                mock_run.assert_called_once()
                call_env = mock_run.call_args[1]['env']
                assert 'LD_LIBRARY_PATH' in call_env
                assert '/path/to/hdf5/lib' in call_env['LD_LIBRARY_PATH']
    
    def test_validate_executable_conda_env(self):
        """Test executable validation in conda environment."""
        with patch.dict(os.environ, {'CONDA_PREFIX': '/conda/env'}):
            with patch('laueanalysis.reconstruct.reconstruct._find_hdf5_lib_path') as mock_find:
                mock_find.return_value = '/conda/env/lib'
                
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    
                    _validate_executable('/path/to/exe')
                    
                    # Check environment setup
                    call_env = mock_run.call_args[1]['env']
                    assert call_env['HDF5_DISABLE_VERSION_CHECK'] == '1'
                    assert '/conda/env/lib' in call_env['LD_LIBRARY_PATH']
    
    def test_execute_reconstruction_with_hdf5_libs(self):
        """Test reconstruction execution sets up HDF5 library paths."""
        with patch('laueanalysis.reconstruct.reconstruct._find_hdf5_lib_path') as mock_find:
            mock_find.return_value = '/path/to/hdf5/lib'
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='Success',
                    stderr=''
                )
                
                result = _execute_reconstruction(
                    ['reconstructN_cpu', '--help'],
                    'output_base',
                    timeout=30
                )
                
                # Check that subprocess was called with proper environment
                call_env = mock_run.call_args[1]['env']
                assert 'LD_LIBRARY_PATH' in call_env
                assert '/path/to/hdf5/lib' in call_env['LD_LIBRARY_PATH']
                
                # Check result
                assert isinstance(result, ReconstructionResult)
                assert result.success is True
    
    def test_execute_reconstruction_preserves_existing_ld_library_path(self):
        """Test that existing LD_LIBRARY_PATH is preserved."""
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/existing/path'}):
            with patch('laueanalysis.reconstruct.reconstruct._find_hdf5_lib_path') as mock_find:
                mock_find.return_value = '/hdf5/lib'
                
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
                    
                    _execute_reconstruction(['cmd'], 'output', 30)
                    
                    call_env = mock_run.call_args[1]['env']
                    # HDF5 lib should be prepended, existing path preserved
                    assert call_env['LD_LIBRARY_PATH'] == '/hdf5/lib:/existing/path'
    
    @pytest.mark.integration
    def test_real_executable_with_hdf5_libs(self):
        """Integration test with real executable and HDF5 library setup."""
        from laueanalysis.reconstruct import find_executable
        
        try:
            exe_path = find_executable()
            
            # This should work now with proper library paths
            _validate_executable(exe_path)
            
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")
