#!/usr/bin/env python3
"""
Example of using GPU reconstruction functions for Laue wire scan analysis.

This example demonstrates:
1. Checking if GPU reconstruction is available
2. Running a single GPU reconstruction
3. Running batch GPU reconstructions
4. Running depth scans with GPU
"""

from laueanalysis.reconstruct import (
    reconstruct_gpu,
    batch_gpu,
    depth_scan_gpu,
    gpu_available,
    find_gpu_executable
)


def main():
    """Run GPU reconstruction examples."""
    
    # Check if GPU reconstruction is available
    print("Checking GPU availability...")
    if gpu_available():
        print("✓ GPU reconstruction is available!")
        try:
            gpu_exe = find_gpu_executable()
            print(f"  GPU executable found at: {gpu_exe}")
        except FileNotFoundError:
            pass
    else:
        print("✗ GPU reconstruction is not available.")
        print("  The GPU executable may not be compiled or not in PATH.")
        print("  Compile it from src/laueanalysis/reconstruct/source/recon_gpu/")
        return
    
    # Example 1: Single GPU reconstruction
    print("\n" + "="*60)
    print("Example 1: Single GPU Reconstruction")
    print("="*60)
    
    result = reconstruct_gpu(
        input_file='wirescan_data.h5',
        output_file='output/gpu_recon_',
        geometry_file='geometry.xml',
        depth_range=(0, 100),  # microns
        resolution=1.0,        # microns
        verbose=2,
        percent_brightest=50.0,
        cuda_rows=16,          # GPU-specific parameter
        memory_limit_mb=512
    )
    
    print(f"Success: {result.success}")
    print(f"Output files: {len(result.output_files)} files generated")
    if result.error:
        print(f"Error: {result.error}")
    
    # Example 2: Batch GPU reconstructions with different parameters
    print("\n" + "="*60)
    print("Example 2: Batch GPU Reconstructions")
    print("="*60)
    
    # Define multiple reconstruction jobs
    batch_configs = [
        {
            'input_file': 'scan1.h5',
            'output_file': 'output/batch1_',
            'geometry_file': 'geo1.xml',
            'depth_range': (0, 50),
            'resolution': 0.5,
            'cuda_rows': 8
        },
        {
            'input_file': 'scan2.h5',
            'output_file': 'output/batch2_',
            'geometry_file': 'geo2.xml',
            'depth_range': (0, 100),
            'resolution': 1.0,
            'cuda_rows': 16
        },
        {
            'input_file': 'scan3.h5',
            'output_file': 'output/batch3_',
            'geometry_file': 'geo3.xml',
            'depth_range': (-50, 50),
            'resolution': 2.0,
            'cuda_rows': 32
        }
    ]
    
    # Run batch processing (sequentially)
    print("Running batch reconstructions sequentially...")
    results = batch_gpu(batch_configs, parallel=False)
    
    for i, result in enumerate(results):
        print(f"  Job {i+1}: {'Success' if result.success else 'Failed'}")
    
    # Example 3: Depth scanning with GPU
    print("\n" + "="*60)
    print("Example 3: GPU Depth Scanning")
    print("="*60)
    
    # Define depth ranges to scan
    depth_ranges = [
        (-100, -50),
        (-50, 0),
        (0, 50),
        (50, 100),
        (100, 150)
    ]
    
    print(f"Scanning {len(depth_ranges)} depth ranges...")
    scan_results = depth_scan_gpu(
        input_file='wirescan_full.h5',
        output_base='output/depth_scan_',
        geometry_file='geometry.xml',
        depth_ranges=depth_ranges,
        resolution=0.5,
        parallel=True,  # Use parallel processing
        percent_brightest=75.0,
        cuda_rows=16,
        wire_depths_file='depth_corrections.txt'  # GPU can use depth corrections
    )
    
    print(f"Completed {len(scan_results)} depth reconstructions")
    successful = sum(1 for r in scan_results if r.success)
    print(f"  Successful: {successful}/{len(scan_results)}")
    
    # Example 4: GPU-specific features
    print("\n" + "="*60)
    print("Example 4: GPU-Specific Features")
    print("="*60)
    
    # The GPU version supports wire depth corrections
    result = reconstruct_gpu(
        input_file='wirescan_data.h5',
        output_file='output/gpu_corrected_',
        geometry_file='geometry.xml',
        depth_range=(0, 100),
        resolution=1.0,
        wire_depths_file='pixel_depth_corrections.dat',  # GPU-specific
        cuda_rows=24,  # Adjust based on GPU memory
        verbose=1
    )
    
    print("GPU reconstruction with depth corrections completed")
    print(f"Command executed: {result.command}")
    
    # Performance comparison note
    print("\n" + "="*60)
    print("Performance Notes:")
    print("="*60)
    print("- GPU reconstruction is typically 10-100x faster than CPU")
    print("- Optimal cuda_rows depends on GPU memory (8-32 typical)")
    print("- GPU version lacks cosmic ray filtering and advanced normalization")
    print("- Use CPU version if these features are needed")


def progress_callback(completed, total):
    """Example progress callback for batch processing."""
    percent = (completed / total) * 100
    print(f"  Progress: {completed}/{total} ({percent:.1f}%)")


if __name__ == "__main__":
    main()
