#!/usr/bin/env python3
"""
PyTorch Wire Scan Reconstruction - Command Line Interface
Matches the functionality of reconstructBP C program with PyTorch implementation.
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
import h5py
from dataclasses import dataclass, field

from laueanalysis.reconstruct.torch_reconstruct import (
    DetectorParams,
    reconstruct_depth_torch,
    rodrigues_rotation_matrix
)


@dataclass
class GeoParams:
    """Geometry parameters extracted from XML file."""
    # Detector parameters
    detector_P: Tuple[float, float, float]
    detector_R: Tuple[float, float, float]
    pixel_size_i: float
    pixel_size_j: float
    xDimDet: int
    yDimDet: int
    
    # Wire parameters
    wire_diameter: float
    wire_axis: Tuple[float, float, float]
    wire_center: Tuple[float, float, float]
    wire_rho: Optional[np.ndarray] = None
    
    # ROI parameters
    starti: int = 0
    startj: int = 0
    endi: int = 0
    endj: int = 0
    bini: int = 1
    binj: int = 1


def parse_geo_xml(geo_file: str, detector_num: int = 0) -> GeoParams:
    """Parse geometry XML file to extract calibration parameters."""
    tree = ET.parse(geo_file)
    root = tree.getroot()
    
    # Remove namespace if present
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    
    # Find detector element - look for Detector with N attribute
    detector = None
    detectors = root.find('.//Detectors')
    if detectors is not None:
        # Look for specific detector number
        for det in detectors.findall('Detector'):
            if det.get('N') == str(detector_num):
                detector = det
                break
        # If not found, use first detector
        if detector is None:
            detector = detectors.find('Detector')
    
    if detector is None:
        raise ValueError(f"No detector found in geometry file")
    
    # Extract detector parameters
    def get_vec3_from_text(text):
        """Parse space-separated vector from text."""
        if text:
            parts = text.strip().split()
            if len(parts) >= 3:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
        return (0, 0, 0)
    
    def get_vec2_from_text(text):
        """Parse space-separated 2D vector from text."""
        if text:
            parts = text.strip().split()
            if len(parts) >= 2:
                return (float(parts[0]), float(parts[1]))
        return (0, 0)
    
    # Detector geometry - P is in mm, need to convert to microns
    P_text = detector.findtext('P', '0 0 0')
    P_mm = get_vec3_from_text(P_text)
    P = (P_mm[0] * 1000, P_mm[1] * 1000, P_mm[2] * 1000)  # Convert mm to microns
    
    # R is in radians
    R_text = detector.findtext('R', '0 0 0')
    R = get_vec3_from_text(R_text)
    
    # Detector dimensions (pixels)
    npixels_text = detector.findtext('Npixels', '2048 2048')
    Ni, Nj = get_vec2_from_text(npixels_text)
    Ni = int(Ni)
    Nj = int(Nj)
    
    # Detector size in mm, convert to microns per pixel
    size_text = detector.findtext('size', '409.6 409.6')
    size_mm = get_vec2_from_text(size_text)
    # Convert from total size in mm to microns per pixel
    size_i = (size_mm[0] * 1000) / Ni  # microns per pixel
    size_j = (size_mm[1] * 1000) / Nj  # microns per pixel
    
    # Wire parameters
    wire = root.find('.//Wire')
    if wire is not None:
        # Wire diameter
        dia_text = wire.findtext('dia', '52')
        diameter = float(dia_text)
        
        # Wire axis (unit vector)
        axis_text = wire.findtext('Axis', '1 0 0')
        axis = get_vec3_from_text(axis_text)
        
        # Wire origin in microns
        origin_text = wire.findtext('Origin', '0 0 0')
        center = get_vec3_from_text(origin_text)
        
        # Wire rotation
        wire_R_text = wire.findtext('R', '0 0 0')
        wire_R = get_vec3_from_text(wire_R_text)
        
        # Calculate rho matrix to align wire axis with x-axis
        # For now, use identity matrix - proper rotation calculation needed
        rho = np.eye(3, dtype=np.float64)
        
        # If wire axis is not along x, compute rotation matrix
        if abs(axis[0] - 1.0) > 1e-6 or abs(axis[1]) > 1e-6 or abs(axis[2]) > 1e-6:
            # Compute rotation matrix to align axis with (1,0,0)
            # Using Rodrigues formula or similar
            # For now, keep identity
            pass
    else:
        diameter = 52.0
        axis = (1, 0, 0)
        center = (0, 0, 0)
        wire_R = (0, 0, 0)
        rho = np.eye(3, dtype=np.float64)
    
    return GeoParams(
        detector_P=P,
        detector_R=R,
        pixel_size_i=size_i,
        pixel_size_j=size_j,
        xDimDet=Ni,
        yDimDet=Nj,
        wire_diameter=diameter,
        wire_axis=axis,
        wire_center=center,
        wire_rho=rho
    )


def load_h5_data_efficient(
    h5_file: str,
    first_image: int = 0,
    last_image: Optional[int] = None,
    percent: float = 100.0,
    min_cutoff: float = 1.0,
    verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Efficiently load HDF5 data with minimal memory overhead.
    
    Returns:
        images: (S, H, W) array of intensities
        wire_positions: (S, 3) array of wire positions
        metadata: Dictionary with ROI and other parameters
        pixel_mask: (H, W) boolean mask of pixels above cutoff
    """
    with h5py.File(h5_file, 'r') as f:
        # Get data shape
        data = f['/entry1/data/data']
        if len(data.shape) == 3:
            n_images, height, width = data.shape
        else:
            height, width = data.shape
            n_images = 1
        
        # Determine image range
        if last_image is None:
            last_image = n_images
        last_image = min(last_image, n_images)
        first_image = max(0, min(first_image - 1, n_images - 1))  # Convert to 0-based
        last_image = max(first_image + 1, last_image)
        
        n_load = last_image - first_image
        
        if verbose > 0:
            print(f"Loading images {first_image+1} to {last_image} ({n_load} images)")
            print(f"Image dimensions: {height} x {width}")
        
        # Load images efficiently using slicing
        if n_images == 1:
            images = data[...].astype(np.float32)
            images = images[np.newaxis, ...]  # Add batch dimension
        else:
            # Use HDF5 slicing for efficient loading
            images = data[first_image:last_image].astype(np.float32)
        
        # Load wire positions
        wire_positions = np.zeros((n_load, 3), dtype=np.float64)
        
        # Try to load wire positions from various possible locations
        wire_paths = [
            ('/entry1/wire/wireX', '/entry1/wire/wireY', '/entry1/wire/wireZ'),
            ('/entry1/sample/wireX', '/entry1/sample/wireY', '/entry1/sample/wireZ'),
        ]
        
        for path_x, path_y, path_z in wire_paths:
            if path_x in f:
                try:
                    wire_x = f[path_x][first_image:last_image]
                    wire_y = f[path_y][first_image:last_image]
                    wire_z = f[path_z][first_image:last_image]
                    wire_positions[:, 0] = wire_x
                    wire_positions[:, 1] = wire_y
                    wire_positions[:, 2] = wire_z
                    break
                except:
                    pass
        
        # Get ROI information
        detector = f.get('/entry1/detector', f.get('/entry1/instrument/detector'))
        metadata = {}
        
        if detector:
            metadata['startx'] = int(detector.attrs.get('startx', 0))
            metadata['starty'] = int(detector.attrs.get('starty', 0))
            metadata['endx'] = int(detector.attrs.get('endx', width - 1))
            metadata['endy'] = int(detector.attrs.get('endy', height - 1))
            metadata['binx'] = int(detector.attrs.get('groupx', 1))
            metadata['biny'] = int(detector.attrs.get('groupy', 1))
        else:
            metadata = {
                'startx': 0, 'starty': 0,
                'endx': width - 1, 'endy': height - 1,
                'binx': 1, 'biny': 1
            }
    
    # Calculate cutoff similar to CPU version
    # Use first image as intensity map (matching CPU's get_intensity_map)
    intensity_map = images[0].copy()
    intensity_sorted = np.sort(intensity_map.flatten())
    
    # Calculate cutoff based on percent (matching CPU logic exactly)
    # CPU code: cutoff = (int)intensity_sorted[ (size_t)floor((double)sort_len * MIN((100.0 - percent)/100.0,1.)) ];
    sort_len = len(intensity_sorted)
    cutoff_ratio = min((100.0 - percent) / 100.0, 1.0)
    cutoff_idx = int(np.floor(sort_len * cutoff_ratio))
    # Ensure index is valid
    cutoff_idx = min(cutoff_idx, sort_len - 1)
    cutoff_idx = max(cutoff_idx, 0)
    
    # Get cutoff value and enforce minimum
    cutoff = float(intensity_sorted[cutoff_idx])
    cutoff = max(cutoff, min_cutoff)
    
    # Create pixel mask for pixels above cutoff
    pixel_mask = intensity_map >= cutoff
    num_valid_pixels = np.sum(pixel_mask)
    total_pixels = pixel_mask.size
    
    if verbose > 0:
        print(f"Ignoring pixels with a value less than {cutoff:.1f}")
        if percent < 100:
            print(f"Processing {percent:.1f}% brightest pixels")
        print(f"Processing {num_valid_pixels}/{total_pixels} pixels ({100*num_valid_pixels/total_pixels:.1f}%)")
    
    # Apply mask to all images (set below-cutoff pixels to 0)
    for i in range(len(images)):
        images[i] = np.where(pixel_mask, images[i], 0)
    
    return images, wire_positions, metadata, pixel_mask


def save_depth_images(
    depth_images: np.ndarray,
    output_base: str,
    depth_values: np.ndarray,
    separate_files: bool = True,
    template_h5: Optional[str] = None,
    verbose: int = 0
) -> None:
    """
    Save depth-resolved images to HDF5 file(s).
    
    Args:
        depth_images: (M, H, W) array of depth-resolved intensities
        output_base: Base path for output files
        depth_values: (M,) array of depth values in microns
        separate_files: If True, save each depth as separate file
        template_h5: Optional template HDF5 file to copy structure from
    """
    n_depths, height, width = depth_images.shape
    
    if separate_files:
        # Save each depth as a separate file (matching C program behavior)
        for i, depth in enumerate(depth_values):
            output_file = f"{output_base}{i}.h5"
            
            if verbose > 0 and i % 10 == 0:
                print(f"Writing depth {i+1}/{n_depths}: {depth:.2f} μm")
            
            with h5py.File(output_file, 'w') as f:
                # Create standard structure
                entry = f.create_group('entry1')
                data_group = entry.create_group('data')
                
                # Save image data
                dataset = data_group.create_dataset(
                    'data',
                    data=depth_images[i],
                    compression='gzip',
                    compression_opts=4,
                    chunks=True
                )
                
                # Add metadata
                entry.attrs['depth'] = depth
                entry.attrs['depth_units'] = 'microns'
                entry.attrs['reconstruction_method'] = 'PyTorch'
                
                # Copy additional structure from template if provided
                if template_h5 and i == 0:  # Only need to check once
                    try:
                        with h5py.File(template_h5, 'r') as template:
                            # Copy relevant metadata groups
                            for key in ['instrument', 'sample']:
                                if key in template.get('entry1', {}):
                                    template.copy(f'/entry1/{key}', entry, name=key)
                    except:
                        pass
    
    else:
        # Save all depths in a single file
        output_file = f"{output_base}reconstructed.h5"
        
        if verbose > 0:
            print(f"Writing all {n_depths} depths to single file: {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # Create standard structure
            entry = f.create_group('entry1')
            data_group = entry.create_group('data')
            
            # Save 3D data block
            dataset = data_group.create_dataset(
                'data',
                data=depth_images,
                compression='gzip',
                compression_opts=4,
                chunks=(1, height, width)  # Chunk by depth slice
            )
            
            # Save depth axis
            depth_dataset = entry.create_dataset('depth', data=depth_values)
            depth_dataset.attrs['units'] = 'microns'
            
            # Add metadata
            entry.attrs['n_depths'] = n_depths
            entry.attrs['depth_start'] = depth_values[0]
            entry.attrs['depth_end'] = depth_values[-1]
            entry.attrs['depth_resolution'] = depth_values[1] - depth_values[0] if n_depths > 1 else 0
            entry.attrs['reconstruction_method'] = 'PyTorch'


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Wire Scan Depth Reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input HDF5 file with wire scan data')
    parser.add_argument('-o', '--output', required=True,
                        help='Output base path for reconstructed images')
    parser.add_argument('-g', '--geo', required=True,
                        help='Geometry XML file')
    
    # Depth parameters
    parser.add_argument('-s', '--depth-start', type=float, required=True,
                        help='Starting depth in microns')
    parser.add_argument('-e', '--depth-end', type=float, required=True,
                        help='Ending depth in microns')
    parser.add_argument('-r', '--resolution', type=float, default=1.0,
                        help='Depth resolution in microns')
    
    # Processing parameters
    parser.add_argument('-f', '--first-image', type=int, default=1,
                        help='First image to process (1-based)')
    parser.add_argument('-l', '--last-image', type=int, default=None,
                        help='Last image to process (1-based)')
    parser.add_argument('-p', '--percent', type=float, default=100,
                        help='Percent of brightest pixels to process')
    parser.add_argument('-w', '--wire-edge', type=int, default=1,
                        choices=[-1, 0, 1],
                        help='Wire edge: 1=leading, 0=trailing, -1=both')
    parser.add_argument('-D', '--detector', type=int, default=0,
                        help='Detector number')
    parser.add_argument('--min-cutoff', type=float, default=1.0,
                        help='Minimum intensity cutoff (matching CPU version)')
    
    # Output options
    parser.add_argument('--single-file', action='store_true',
                        help='Save all depths in single HDF5 file (default: separate files)')
    
    # Performance parameters
    parser.add_argument('-N', '--threads', type=int, default=1,
                        help='Number of threads (for CPU operations)')
    parser.add_argument('-m', '--memory', type=int, default=50000,
                        help='Memory limit in MB (informational)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Computation device')
    parser.add_argument('--chunk-size', type=int, default=256,
                        help='Chunk size for processing pixels (smaller uses less memory)')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbosity level (0-3)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if args.verbose > 0:
        print("=" * 80)
        print("PyTorch Wire Scan Reconstruction")
        print("=" * 80)
        print(f"Input file: {args.input}")
        print(f"Output base: {args.output}")
        print(f"Geometry file: {args.geo}")
        print(f"Depth range: {args.depth_start} to {args.depth_end} μm")
        print(f"Resolution: {args.resolution} μm")
        print(f"Device: {device}")
        print(f"Output mode: {'Single file' if args.single_file else 'Separate files'}")
        print("-" * 80)
    
    start_time = time.time()
    
    # Parse geometry file
    if args.verbose > 0:
        print("Loading geometry...")
    geo = parse_geo_xml(args.geo, args.detector)
    
    # Load HDF5 data
    if args.verbose > 0:
        print("Loading wire scan data...")
    images, wire_positions, metadata, pixel_mask = load_h5_data_efficient(
        args.input,
        args.first_image,
        args.last_image,
        args.percent,
        args.min_cutoff,
        args.verbose
    )
    
    # Update geometry with ROI information
    geo.starti = metadata.get('startx', 0)
    geo.startj = metadata.get('starty', 0)
    geo.endi = metadata.get('endx', geo.xDimDet - 1)
    geo.endj = metadata.get('endy', geo.yDimDet - 1)
    geo.bini = metadata.get('binx', 1)
    geo.binj = metadata.get('biny', 1)
    
    # Create DetectorParams (order matters for dataclass)
    params = DetectorParams(
        xDimDet=geo.xDimDet,
        yDimDet=geo.yDimDet,
        pixel_size_i=geo.pixel_size_i,
        pixel_size_j=geo.pixel_size_j,
        P=geo.detector_P,
        R=geo.detector_R,
        wire_diameter=geo.wire_diameter,
        starti=geo.starti,
        startj=geo.startj,
        bini=geo.bini,
        binj=geo.binj,
        depth_start=args.depth_start,
        depth_end=args.depth_end,
        depth_resolution=args.resolution,
        # Optional parameters last
        rho=torch.tensor(geo.wire_rho, dtype=torch.float64) if geo.wire_rho is not None else None,
        ki=None  # Will use default (0,0,1)
    )
    
    # Convert to torch tensors
    if args.verbose > 0:
        print(f"Converting to PyTorch tensors on {device}...")
    
    images_torch = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    wire_positions_torch = torch.from_numpy(wire_positions).to(device=device, dtype=torch.float64)
    
    # Perform reconstruction
    if args.verbose > 0:
        print(f"Performing depth reconstruction...")
        print(f"  Images shape: {images_torch.shape}")
        print(f"  Wire positions shape: {wire_positions_torch.shape}")
        print(f"  Number of depth bins: {int((args.depth_end - args.depth_start) / args.resolution + 1)}")
        print(f"  Active pixels: {np.sum(pixel_mask)}/{pixel_mask.size} ({100*np.sum(pixel_mask)/pixel_mask.size:.1f}%)")
    
    reconstruction_start = time.time()
    
    # Set number of threads for CPU operations
    if device == 'cpu':
        torch.set_num_threads(args.threads)
    
    # Perform reconstruction
    depth_images = reconstruct_depth_torch(
        images_torch,
        wire_positions_torch,
        params,
        wire_edge=args.wire_edge,
        chunk_size=args.chunk_size
    )
    
    # Ensure computation is complete (for CUDA)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    reconstruction_time = time.time() - reconstruction_start
    
    if args.verbose > 0:
        print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")
        print(f"Output shape: {depth_images.shape}")
    
    # Convert back to numpy for saving
    depth_images_np = depth_images.cpu().numpy()
    
    # Calculate depth values
    n_depths = depth_images_np.shape[0]
    depth_values = np.linspace(args.depth_start, args.depth_end, n_depths)
    
    # Save results
    if args.verbose > 0:
        print(f"Saving results...")
    
    save_depth_images(
        depth_images_np,
        args.output,
        depth_values,
        separate_files=not args.single_file,
        template_h5=args.input,
        verbose=args.verbose
    )
    
    total_time = time.time() - start_time
    
    if args.verbose > 0:
        print("-" * 80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"  Data loading: {reconstruction_start - start_time:.2f} seconds")
        print(f"  Reconstruction: {reconstruction_time:.2f} seconds")
        print(f"  Data saving: {total_time - reconstruction_start - reconstruction_time:.2f} seconds")
        print("=" * 80)
        print("Reconstruction complete!")


if __name__ == '__main__':
    main()
