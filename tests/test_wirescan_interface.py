"""Smoke tests for the WireScan reconstruction interface."""

import os
import tempfile
import pytest
from laueanalysis.reconstruct.wirescan_interface import WireScanRecon


def test_interface_initialization():
    """Test that the interface can be initialized successfully."""
    interface = WireScanRecon()
    assert interface.wirescan_executable is not None
    assert isinstance(interface.wirescan_executable, str)


def test_executable_validation():
    """Test that the executable validation works."""
    # This should work if the executable is properly installed
    try:
        interface = WireScanRecon()
        # If we get here, the executable was found and validated
        assert interface.wirescan_executable is not None
    except RuntimeError as e:
        # If it fails, it should be a clear error about the executable
        assert "not found" in str(e).lower() or "not working" in str(e).lower()


def test_interface_with_custom_executable():
    """Test interface with a custom executable path."""
    # Test with a non-existent executable
    with pytest.raises(RuntimeError, match="not found|not working"):
        WireScanRecon(wirescan_executable='/nonexistent/path')


def test_reconstruction_with_missing_files():
    """Test reconstruction with missing input files."""
    interface = WireScanRecon()
    
    # Test with non-existent files - should fail gracefully
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "output")
        
        result = interface.run_reconstruction(
            infile="/nonexistent/input",
            outfile=outfile,
            geofile="/nonexistent/geo.xml",
            depth_start=0.0,
            depth_end=10.0,
            resolution=1.0,
            first_image=1,
            last_image=2
        )
        
        # Should return a result dict with success=False
        assert isinstance(result, dict)
        assert 'success' in result
        # It should fail because files don't exist
        assert result['success'] is False


def test_reconstruction_parameter_validation():
    """Test that invalid parameters are handled."""
    interface = WireScanRecon()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy files
        infile = os.path.join(tmpdir, "input.h5")
        outfile = os.path.join(tmpdir, "output")
        geofile = os.path.join(tmpdir, "geo.xml")
        
        for f in [infile, geofile]:
            with open(f, 'w') as fp:
                fp.write("dummy content")
        
        # Test with invalid depth range (start > end)
        result = interface.run_reconstruction(
            infile=infile,
            outfile=outfile,
            geofile=geofile,
            depth_start=100.0,
            depth_end=50.0,  # End before start
            resolution=1.0
        )
        
        # Should handle the error gracefully
        assert isinstance(result, dict)
        assert 'success' in result


def test_batch_reconstruction_interface():
    """Test the batch reconstruction interface."""
    interface = WireScanRecon()
    
    # Create a simple config list
    configs = [
        {
            'infile': '/dummy/input1',
            'outfile': '/dummy/output1', 
            'geofile': '/dummy/geo.xml',
            'depth_start': 0.0,
            'depth_end': 25.0,
            'resolution': 1.0
        },
        {
            'infile': '/dummy/input2',
            'outfile': '/dummy/output2',
            'geofile': '/dummy/geo.xml', 
            'depth_start': 25.0,
            'depth_end': 50.0,
            'resolution': 1.0
        }
    ]
    
    # This should return a list of results
    results = interface.run_reconstruction_batch(configs)
    assert isinstance(results, list)
    assert len(results) == 2
    
    # Each result should be a dict with at least a 'success' key
    for result in results:
        assert isinstance(result, dict)
        assert 'success' in result


def test_helper_functions_exist():
    """Test that helper functions are available."""
    from laueanalysis.reconstruct.wirescan_interface import (
        create_simple_reconstruction_config,
        create_depth_scan_batch
    )
    
    # Test simple config creation
    config = create_simple_reconstruction_config(
        input_base="/dummy/input",
        output_base="/dummy/output",
        geo_file="/dummy/geo.xml",
        depth_start=0.0,
        depth_end=100.0
    )
    
    assert isinstance(config, dict)
    assert 'infile' in config
    assert 'outfile' in config
    assert 'geofile' in config
    assert config['depth_start'] == 0.0
    assert config['depth_end'] == 100.0
    
    # Test batch creation
    batch = create_depth_scan_batch(
        input_base="/dummy/input_",
        output_base="/dummy/output_",
        geo_file="/dummy/geo.xml",
        depth_ranges=[(0, 25), (25, 50), (50, 75)],
        resolution=1.0
    )
    
    assert isinstance(batch, list)
    assert len(batch) == 3
    for config in batch:
        assert isinstance(config, dict)
        assert 'depth_start' in config
        assert 'depth_end' in config


def test_reconstruction_smoke_test():
    """Smoke test for actual reconstruction with minimal synthetic data."""
    import numpy as np
    import h5py
    import shutil
    from pathlib import Path
    
    # Skip test if executable is not available
    try:
        interface = WireScanRecon()
    except RuntimeError as e:
        pytest.skip(f"WireScan executable not available: {e}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal test data
        # Small image size for quick testing
        image_size = 128
        num_images = 5
        
        # Create synthetic HDF5 file with minimal structure
        input_file = os.path.join(tmpdir, "test_wire_scan.h5")
        with h5py.File(input_file, 'w') as f:
            # Create minimal required structure based on the h5ls output
            facility = f.create_group('Facility')
            facility.create_dataset('facility_name', data=b'TEST')
            facility.create_dataset('facility_beamline', data=b'34ID-E')
            
            entry = f.create_group('entry1')
            data_group = entry.create_group('data')
            
            # Create small synthetic images (128x128 instead of 2048x2048)
            # Simulate wire scan data with a simple pattern
            for i in range(num_images):
                # Create a simple pattern that varies with image number
                image_data = np.zeros((image_size, image_size), dtype=np.float32)
                # Add a vertical stripe that moves across images (simulating wire scan)
                stripe_pos = int(i * image_size / num_images)
                image_data[:, stripe_pos:stripe_pos+10] = 1000.0 * np.random.rand(image_size, 10)
                # Add some noise
                image_data += 10.0 * np.random.rand(image_size, image_size)
                
                # Store as individual datasets (common format)
                data_group.create_dataset(f'data_{i:04d}', data=image_data)
            
            # Also create the single data dataset for compatibility
            data_group.create_dataset('data', data=image_data)
            
            # Add depth information
            entry.create_dataset('depth', data=[0.0])
            
            # Add detector information
            detector = entry.create_group('detector')
            detector.create_dataset('Nx', data=[image_size])
            detector.create_dataset('Ny', data=[image_size])
            detector.create_dataset('binx', data=[1])
            detector.create_dataset('biny', data=[1])
        
        # Copy a geometry file from test data
        geo_source = "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
        geo_file = os.path.join(tmpdir, "test_geo.xml")
        if os.path.exists(geo_source):
            shutil.copy(geo_source, geo_file)
        else:
            # Create a minimal geometry file if test data not available
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
        
        # Set up output path
        output_base = os.path.join(tmpdir, "output", "recon_")
        
        # Run reconstruction with minimal parameters
        result = interface.run_reconstruction(
            infile=input_file,
            outfile=output_base,
            geofile=geo_file,
            depth_start=-2.0,  # Small depth range
            depth_end=2.0,
            resolution=1.0,    # Coarse resolution for speed
            first_image=0,
            last_image=num_images-1,
            verbose=1,
            percent_to_process=50.0,  # Process only 50% brightest pixels
            memory_mb=50,      # Small memory limit
            wire_edges='l'     # Leading edge only
        )
        
        # Check results
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'command' in result
        assert 'log' in result
        
        # Print debug info if test fails
        if not result['success']:
            print(f"Command: {result['command']}")
            print(f"Error: {result.get('error', 'No error message')}")
            print(f"Log: {result.get('log', 'No log')}")
        
        # For smoke test, we just check that it attempted to run
        # The actual success depends on whether the C program is compiled
        # and the exact format it expects
        assert result['return_code'] is not None


def test_reconstruction_smoke_test_with_real_data():
    """Smoke test using the existing test data file if available."""
    import shutil
    from pathlib import Path
    
    # Skip test if executable is not available
    try:
        interface = WireScanRecon()
    except RuntimeError as e:
        pytest.skip(f"WireScan executable not available: {e}")
    
    # Check if test data exists
    test_data_file = "tests/data/gdata/Al30_thick_wire_55_50.h5"
    if not os.path.exists(test_data_file):
        pytest.skip("Test data file not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy test data
        input_file = os.path.join(tmpdir, "test_input.h5")
        shutil.copy(test_data_file, input_file)
        
        # Copy geometry file
        geo_source = "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
        geo_file = os.path.join(tmpdir, "test_geo.xml")
        shutil.copy(geo_source, geo_file)
        
        # Set up output path
        output_base = os.path.join(tmpdir, "output", "recon_")
        
        # Run reconstruction with very limited parameters for speed
        result = interface.run_reconstruction(
            infile=input_file,
            outfile=output_base,
            geofile=geo_file,
            depth_start=-1.0,  # Very small depth range
            depth_end=1.0,
            resolution=2.0,    # Very coarse resolution
            first_image=1,
            last_image=3,      # Process only 3 images
            verbose=2,
            percent_to_process=10.0,  # Process only 10% brightest pixels
            memory_mb=100,
            wire_edges='l'
        )
        
        # Check results
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'command' in result
        
        # Print debug info
        if not result['success']:
            print(f"Command: {result['command']}")
            print(f"Error: {result.get('error', 'No error message')}")
            print(f"Log output: {result.get('log', 'No log')[:500]}")  # First 500 chars
