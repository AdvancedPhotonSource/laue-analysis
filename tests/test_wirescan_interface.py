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
