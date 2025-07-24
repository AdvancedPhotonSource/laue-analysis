"""Tests for the reconstruction functional API."""

import os
import tempfile
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import h5py
import shutil

from laueanalysis.reconstruct import (
    reconstruct,
    batch,
    depth_scan,
    find_executable,
    ReconstructionResult,
    # GPU functions
    reconstruct_gpu,
    batch_gpu,
    depth_scan_gpu,
    find_gpu_executable,
    gpu_available
)


class TestReconstruct:
    """Test reconstruction functions."""
    
    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess.run for unit tests."""
        with patch('subprocess.run') as mock:
            # Default return value for validation calls (--help)
            help_response = MagicMock(
                returncode=1,  # WireScan returns 1 for help
                stdout="Usage: WireScan -i <file> -o <file> -g <file>",
                stderr="",
                timeout=None
            )
            # Default return value for actual reconstruction calls
            run_response = MagicMock(
                returncode=0,
                stdout="Reconstruction complete",
                stderr="",
                timeout=None
            )
            
            # Return help response for validation, run response for actual execution
            def side_effect(*args, **kwargs):
                if '--help' in args[0]:
                    return help_response
                return run_response
            
            mock.side_effect = side_effect
            yield mock
    
    @pytest.fixture
    def mock_executable(self):
        """Mock the executable finder."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock:
            mock.return_value = '/path/to/reconstructN_cpu'
            yield mock
    
    def test_reconstruction_result_type(self):
        """Test that ReconstructionResult is properly defined."""
        result = ReconstructionResult(
            success=True,
            output_files=['file1.h5', 'file2.h5'],
            log='Success',
            error=None,
            command='reconstructN_cpu ...',
            return_code=0
        )
        assert result.success is True
        assert len(result.output_files) == 2
        assert result.error is None
    
    def test_reconstruct_basic(self, mock_subprocess, mock_executable):
        """Test basic reconstruction call."""
        result = reconstruct(
            'input.h5',
            'output_',
            'geo.xml',
            (0.0, 10.0),
            resolution=1.0
        )
        
        assert isinstance(result, ReconstructionResult)
        assert result.success is True
        assert result.return_code == 0
        
        # Verify subprocess was called
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        
        # Check basic arguments
        assert '-i' in call_args
        assert 'input.h5' in call_args
        assert '-o' in call_args
        assert 'output_' in call_args
        assert '-g' in call_args
        assert 'geo.xml' in call_args
        assert '-s' in call_args
        assert '0.0' in call_args
        assert '-e' in call_args
        assert '10.0' in call_args
    
    def test_reconstruct_with_all_options(self, mock_subprocess, mock_executable):
        """Test reconstruction with all optional parameters."""
        result = reconstruct(
            'input.h5',
            'output_',
            'geo.xml',
            (-5.0, 5.0),
            resolution=0.5,
            image_range=(1, 100),
            verbose=2,
            percent_brightest=50.0,
            wire_edge='both',
            memory_limit_mb=256,
            normalization='norm_tag',
            output_pixel_type=3,
            distortion_map='distortion.map',
            detector_number=1,
            wire_depths_file='depths.txt'
        )
        
        assert result.success is True
        
        # Check all parameters in command
        call_args = mock_subprocess.call_args[0][0]
        assert '-r' in call_args
        assert '0.5' in call_args
        assert '-f' in call_args
        assert '1' in call_args
        assert '-l' in call_args
        assert '100' in call_args
        assert '-v' in call_args
        assert '2' in call_args
        assert '-p' in call_args
        assert '50.0' in call_args
        assert '-w' in call_args
        assert 'b' in call_args  # 'both' maps to 'b'
        assert '-m' in call_args
        assert '256' in call_args
        assert '-n' in call_args
        assert 'norm_tag' in call_args
        assert '-t' in call_args
        assert '3' in call_args
        assert '-d' in call_args
        assert 'distortion.map' in call_args
        assert '-D' in call_args
        assert '1' in call_args
        assert '--wireDepths' in call_args
        assert 'depths.txt' in call_args
    
    def test_wire_edge_mapping(self, mock_subprocess, mock_executable):
        """Test that wire edge names are properly mapped."""
        # Test user-friendly names
        for user_name, expected in [
            ('leading', 'l'),
            ('trailing', 't'),
            ('both', 'b'),
            ('LEADING', 'l'),  # Case insensitive
            ('l', 'l'),  # Also accept short form
            ('t', 't'),
            ('b', 'b')
        ]:
            result = reconstruct(
                'input.h5', 'output_', 'geo.xml', (0, 10),
                wire_edge=user_name
            )
            call_args = mock_subprocess.call_args[0][0]
            idx = call_args.index('-w')
            assert call_args[idx + 1] == expected
    
    def test_invalid_wire_edge(self, mock_executable):
        """Test that invalid wire edge raises ValueError."""
        with patch('laueanalysis.reconstruct.reconstruct._validate_executable'):
            with pytest.raises(ValueError, match="Invalid wire_edge"):
                reconstruct(
                    'input.h5', 'output_', 'geo.xml', (0, 10),
                    wire_edge='invalid'
                )
    
    def test_invalid_depth_range(self, mock_executable):
        """Test that invalid depth range raises ValueError."""
        with patch('laueanalysis.reconstruct.reconstruct._validate_executable'):
            with pytest.raises(ValueError, match="Invalid depth range"):
                reconstruct(
                    'input.h5', 'output_', 'geo.xml', (10, 5)  # Start > end
                )
    
    def test_batch_sequential(self, mock_subprocess, mock_executable):
        """Test sequential batch processing."""
        configs = [
            {
                'input_file': 'in1.h5',
                'output_file': 'out1_',
                'geometry_file': 'geo.xml',
                'depth_range': (0, 5)
            },
            {
                'input_file': 'in2.h5',
                'output_file': 'out2_',
                'geometry_file': 'geo.xml',
                'depth_range': (5, 10)
            }
        ]
        
        results = batch(configs, parallel=False)
        
        assert len(results) == 2
        assert all(isinstance(r, ReconstructionResult) for r in results)
        assert all(r.success for r in results)
        assert mock_subprocess.call_count == 4  # 2 validations + 2 reconstructions
    
    def test_batch_parallel(self, mock_subprocess, mock_executable):
        """Test parallel batch processing."""
        configs = [
            {
                'input_file': f'in{i}.h5',
                'output_file': f'out{i}_',
                'geometry_file': 'geo.xml',
                'depth_range': (i*10, (i+1)*10)
            }
            for i in range(4)
        ]
        
        results = batch(configs, parallel=True, max_workers=2)
        
        assert len(results) == 4
        assert all(isinstance(r, ReconstructionResult) for r in results)
    
    def test_batch_stop_on_error(self, mock_subprocess, mock_executable):
        """Test batch processing stops on error when requested."""
        # Create custom side effect that makes second reconstruction fail
        call_count = 0
        
        def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if '--help' in args[0]:
                return MagicMock(
                    returncode=1,
                    stdout="Usage: WireScan -i <file> -o <file> -g <file>",
                    stderr="",
                    timeout=None
                )
            
            # Make the second actual reconstruction fail (4th call overall)
            if call_count == 4:
                return MagicMock(returncode=1, stdout="", stderr="Error")
            
            return MagicMock(returncode=0, stdout="OK", stderr="")
        
        mock_subprocess.side_effect = custom_side_effect
        
        configs = [
            {'input_file': f'in{i}.h5', 'output_file': f'out{i}_',
             'geometry_file': 'geo.xml', 'depth_range': (0, 10)}
            for i in range(3)
        ]
        
        results = batch(configs, parallel=False, stop_on_error=True)
        
        assert len(results) == 2  # Should stop after second
        assert results[0].success is True
        assert results[1].success is False
    
    def test_batch_with_progress_callback(self, mock_subprocess, mock_executable):
        """Test batch processing with progress callback."""
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        configs = [
            {'input_file': f'in{i}.h5', 'output_file': f'out{i}_',
             'geometry_file': 'geo.xml', 'depth_range': (0, 10)}
            for i in range(3)
        ]
        
        results = batch(configs, parallel=False, progress_callback=progress_callback)
        
        assert len(results) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    
    def test_depth_scan(self, mock_subprocess, mock_executable):
        """Test depth scanning convenience function."""
        results = depth_scan(
            'input.h5',
            'output_base_',
            'geo.xml',
            [(0, 5), (5, 10), (10, 15)],
            resolution=0.5,
            parallel=False,
            percent_brightest=75.0
        )
        
        assert len(results) == 3
        
        # Check that output names include depth info
        for i, (call_args, _) in enumerate(mock_subprocess.call_args_list[1::2]):  # Skip validation calls
            cmd = call_args[0]
            outfile_idx = cmd.index('-o')
            outfile = cmd[outfile_idx + 1]
            assert 'depth_' in outfile
            assert f'{i*5}.0_{(i+1)*5}.0' in outfile
    
    def test_find_executable_function(self):
        """Test the public find_executable function."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock:
            mock.return_value = '/path/to/exe'
            
            path = find_executable()
            assert path == '/path/to/exe'
            mock.assert_called_once_with()
    
    def test_executable_not_found(self):
        """Test behavior when executable is not found."""
        with patch('laueanalysis.reconstruct.reconstruct.shutil.which', return_value=None):
            with patch('laueanalysis.reconstruct.reconstruct.resources.files', side_effect=ModuleNotFoundError):
                with pytest.raises(FileNotFoundError, match="not found"):
                    reconstruct('in.h5', 'out_', 'geo.xml', (0, 10))
    
    def test_subprocess_timeout(self, mock_executable):
        """Test handling of subprocess timeout."""
        with patch('laueanalysis.reconstruct.reconstruct._validate_executable'):
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('cmd', 30)):
                result = reconstruct(
                    'input.h5', 'output_', 'geo.xml', (0, 10),
                    timeout=30
                )
                
                assert result.success is False
                assert 'timed out' in result.error
                assert result.return_code == -1
    
    def test_subprocess_exception(self, mock_executable):
        """Test handling of subprocess exceptions."""
        with patch('laueanalysis.reconstruct.reconstruct._validate_executable'):
            with patch('subprocess.run', side_effect=Exception("Test error")):
                result = reconstruct(
                    'input.h5', 'output_', 'geo.xml', (0, 10)
                )
                
                assert result.success is False
                assert "Test error" in result.error
                assert result.return_code == -1
    
    @pytest.mark.integration
    def test_real_executable_validation(self):
        """Test with real executable if available."""
        try:
            exe_path = find_executable()
            # If we get here, executable was found
            
            # Test that validation doesn't raise
            from laueanalysis.reconstruct.reconstruct import _validate_executable
            _validate_executable(exe_path)
            
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")
    
    @pytest.mark.integration
    def test_smoke_test_with_synthetic_data(self):
        """Integration test with minimal synthetic data."""
        try:
            exe_path = find_executable()
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal test data
            image_size = 128
            num_images = 5
            
            # Create synthetic HDF5 file
            input_file = os.path.join(tmpdir, "test_wire_scan.h5")
            with h5py.File(input_file, 'w') as f:
                # Create minimal required structure
                facility = f.create_group('Facility')
                facility.create_dataset('facility_name', data=b'TEST')
                facility.create_dataset('facility_beamline', data=b'34ID-E')
                
                entry = f.create_group('entry1')
                data_group = entry.create_group('data')
                
                # Create small synthetic images
                for i in range(num_images):
                    image_data = np.zeros((image_size, image_size), dtype=np.float32)
                    stripe_pos = int(i * image_size / num_images)
                    image_data[:, stripe_pos:stripe_pos+10] = 1000.0 * np.random.rand(image_size, 10)
                    image_data += 10.0 * np.random.rand(image_size, image_size)
                    data_group.create_dataset(f'data_{i:04d}', data=image_data)
                
                data_group.create_dataset('data', data=image_data)
                entry.create_dataset('depth', data=[0.0])
                
                detector = entry.create_group('detector')
                detector.create_dataset('Nx', data=[image_size])
                detector.create_dataset('Ny', data=[image_size])
                detector.create_dataset('binx', data=[1])
                detector.create_dataset('biny', data=[1])
            
            # Create or copy geometry file
            geo_file = os.path.join(tmpdir, "test_geo.xml")
            geo_source = "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
            if os.path.exists(geo_source):
                shutil.copy(geo_source, geo_file)
            else:
                with open(geo_file, 'w') as f:
                    f.write("""<?xml version="1.0"?>
<geometry>
    <detector>
        <distance>100.0</distance>
        <pixelSize>0.1</pixelSize>
        <centerX>64</centerX>
        <centerY>64</centerY>
    </detector>
</geometry>""")
            
            # Run reconstruction
            output_base = os.path.join(tmpdir, "output", "recon_")
            result = reconstruct(
                input_file,
                output_base,
                geo_file,
                (-2.0, 2.0),
                resolution=1.0,
                image_range=(0, num_images-1),
                verbose=1,
                percent_brightest=50.0,
                memory_limit_mb=50
            )
            
            # Check results
            assert isinstance(result, ReconstructionResult)
            assert 'command' in result._asdict()
            
            if not result.success:
                print(f"Command: {result.command}")
                print(f"Error: {result.error}")
                print(f"Log: {result.log[:500]}")


class TestReconstructGPU:
    """Test GPU reconstruction functions."""
    
    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess.run for unit tests."""
        with patch('subprocess.run') as mock:
            # Default return value for validation calls (--help)
            help_response = MagicMock(
                returncode=1,  # WireScan returns 1 for help
                stdout="Usage: WireScan -i <file> -o <file> -g <file>",
                stderr="",
                timeout=None
            )
            # Default return value for actual reconstruction calls
            run_response = MagicMock(
                returncode=0,
                stdout="GPU Reconstruction complete",
                stderr="",
                timeout=None
            )
            
            # Return help response for validation, run response for actual execution
            def side_effect(*args, **kwargs):
                if '--help' in args[0]:
                    return help_response
                return run_response
            
            mock.side_effect = side_effect
            yield mock
    
    @pytest.fixture
    def mock_gpu_executable(self):
        """Mock the GPU executable finder."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock:
            def side_effect(name='reconstructN_cpu'):
                if name == 'reconstructN_gpu':
                    return '/path/to/reconstructN_gpu'
                return '/path/to/reconstructN_cpu'
            mock.side_effect = side_effect
            yield mock
    
    def test_reconstruct_gpu_basic(self, mock_subprocess, mock_gpu_executable):
        """Test basic GPU reconstruction call."""
        result = reconstruct_gpu(
            'input.h5',
            'output_',
            'geo.xml',
            (0.0, 10.0),
            resolution=1.0
        )
        
        assert isinstance(result, ReconstructionResult)
        assert result.success is True
        assert result.return_code == 0
        
        # Verify subprocess was called with GPU executable
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        
        # Check basic arguments
        assert call_args[0] == '/path/to/reconstructN_gpu'
        assert '-i' in call_args
        assert 'input.h5' in call_args
        assert '-o' in call_args
        assert 'output_' in call_args
        assert '-g' in call_args
        assert 'geo.xml' in call_args
        assert '-s' in call_args
        assert '0.0' in call_args
        assert '-e' in call_args
        assert '10.0' in call_args
        assert '-R' in call_args
        assert '8' in call_args  # Default cuda_rows
    
    def test_reconstruct_gpu_with_options(self, mock_subprocess, mock_gpu_executable):
        """Test GPU reconstruction with optional parameters."""
        result = reconstruct_gpu(
            'input.h5',
            'output_',
            'geo.xml',
            (-5.0, 5.0),
            resolution=0.5,
            image_range=(1, 100),
            verbose=2,
            percent_brightest=50.0,
            wire_edge='both',
            memory_limit_mb=256,
            normalization='norm_tag',
            output_pixel_type=3,
            distortion_map='distortion.map',
            detector_number=1,
            wire_depths_file='depths.txt',
            cuda_rows=16
        )
        
        assert result.success is True
        
        # Check all parameters in command
        call_args = mock_subprocess.call_args[0][0]
        assert '-r' in call_args
        assert '0.5' in call_args
        assert '-f' in call_args
        assert '1' in call_args
        assert '-l' in call_args
        assert '100' in call_args
        assert '-v' in call_args
        assert '2' in call_args
        assert '-p' in call_args
        assert '50.0' in call_args
        assert '-w' in call_args
        assert 'b' in call_args  # 'both' maps to 'b'
        assert '-m' in call_args
        assert '256' in call_args
        assert '-n' in call_args
        assert 'norm_tag' in call_args
        assert '-t' in call_args
        assert '3' in call_args
        assert '-d' in call_args
        assert 'distortion.map' in call_args
        assert '-D' in call_args
        assert '1' in call_args
        assert '-W' in call_args  # GPU uses -W not --wireDepths
        assert 'depths.txt' in call_args
        assert '-R' in call_args
        assert '16' in call_args  # Custom cuda_rows
    
    def test_gpu_does_not_have_cpu_only_params(self, mock_subprocess, mock_gpu_executable):
        """Test that GPU version doesn't have CPU-only parameters."""
        result = reconstruct_gpu(
            'input.h5',
            'output_',
            'geo.xml',
            (0.0, 10.0)
        )
        
        call_args = mock_subprocess.call_args[0][0]
        
        # These CPU-only parameters should NOT be in GPU command
        assert '-C' not in call_args  # cosmic_filter
        assert '-E' not in call_args  # norm_exponent
        assert '-T' not in call_args  # norm_threshold
        assert '-N' not in call_args  # num_threads
        # Note: -R is used but for cuda_rows, not rows_per_stripe
    
    def test_batch_gpu(self, mock_subprocess, mock_gpu_executable):
        """Test GPU batch processing."""
        configs = [
            {
                'input_file': 'in1.h5',
                'output_file': 'out1_',
                'geometry_file': 'geo.xml',
                'depth_range': (0, 5),
                'cuda_rows': 16
            },
            {
                'input_file': 'in2.h5',
                'output_file': 'out2_',
                'geometry_file': 'geo.xml',
                'depth_range': (5, 10),
                'cuda_rows': 32
            }
        ]
        
        results = batch_gpu(configs, parallel=False)
        
        assert len(results) == 2
        assert all(isinstance(r, ReconstructionResult) for r in results)
        assert all(r.success for r in results)
        
        # Check that cuda_rows were passed correctly
        call1 = mock_subprocess.call_args_list[1][0][0]  # Skip validation
        call2 = mock_subprocess.call_args_list[3][0][0]  # Skip validation
        
        idx1 = call1.index('-R')
        assert call1[idx1 + 1] == '16'
        
        idx2 = call2.index('-R')
        assert call2[idx2 + 1] == '32'
    
    def test_depth_scan_gpu(self, mock_subprocess, mock_gpu_executable):
        """Test GPU depth scanning convenience function."""
        results = depth_scan_gpu(
            'input.h5',
            'output_base_',
            'geo.xml',
            [(0, 5), (5, 10), (10, 15)],
            resolution=0.5,
            parallel=False,
            percent_brightest=75.0,
            cuda_rows=24
        )
        
        assert len(results) == 3
        
        # Check that output names include depth info
        for i, (call_args, _) in enumerate(mock_subprocess.call_args_list[1::2]):  # Skip validation calls
            cmd = call_args[0]
            outfile_idx = cmd.index('-o')
            outfile = cmd[outfile_idx + 1]
            assert 'depth_' in outfile
            assert f'{i*5}.0_{(i+1)*5}.0' in outfile
            
            # Check cuda_rows
            idx = cmd.index('-R')
            assert cmd[idx + 1] == '24'
    
    def test_find_gpu_executable_function(self):
        """Test the public find_gpu_executable function."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock:
            mock.return_value = '/path/to/gpu/exe'
            
            path = find_gpu_executable()
            assert path == '/path/to/gpu/exe'
            mock.assert_called_once_with('reconstructN_gpu')
    
    def test_gpu_available_true(self):
        """Test gpu_available when GPU is available."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.return_value = '/path/to/reconstructN_gpu'
            with patch('laueanalysis.reconstruct.reconstruct._validate_executable'):
                assert gpu_available() is True
    
    def test_gpu_available_false_not_found(self):
        """Test gpu_available when GPU executable not found."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.side_effect = FileNotFoundError("Not found")
            assert gpu_available() is False
    
    def test_gpu_available_false_validation_fails(self):
        """Test gpu_available when GPU executable validation fails."""
        with patch('laueanalysis.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.return_value = '/path/to/reconstructN_gpu'
            with patch('laueanalysis.reconstruct.reconstruct._validate_executable') as mock_validate:
                mock_validate.side_effect = RuntimeError("Validation failed")
                assert gpu_available() is False
    
    def test_gpu_executable_not_found(self):
        """Test behavior when GPU executable is not found."""
        with patch('laueanalysis.reconstruct.reconstruct.shutil.which', return_value=None):
            with patch('laueanalysis.reconstruct.reconstruct.resources.files', side_effect=ModuleNotFoundError):
                with pytest.raises(FileNotFoundError, match="reconstructN_gpu"):
                    reconstruct_gpu('in.h5', 'out_', 'geo.xml', (0, 10))
    
    @pytest.mark.integration
    def test_real_gpu_executable_validation(self):
        """Test with real GPU executable if available."""
        if not gpu_available():
            pytest.skip("GPU reconstruction executable not available")
        
        try:
            exe_path = find_gpu_executable()
            # If we get here, GPU executable was found
            
            # Test that validation doesn't raise
            from laueanalysis.reconstruct.reconstruct import _validate_executable
            _validate_executable(exe_path)
            
        except FileNotFoundError:
            pytest.skip("GPU reconstruction executable not available")
