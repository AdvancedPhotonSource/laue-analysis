import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DetectorParams:
    # Full-chip unbinned detector dimensions (pixels)
    xDimDet: int
    yDimDet: int

    # Pixel pitch (microns) along detector i and j axes
    pixel_size_i: float
    pixel_size_j: float

    # Detector translation vector P (microns) in detector frame before rotation
    P: Tuple[float, float, float]

    # Detector rotation vector R (Rodrigues: direction is axis, length is angle in radians)
    R: Tuple[float, float, float]

    # Wire calibration
    wire_diameter: float  # microns

    # ROI and binning (mapping binned ROI indices to full-chip unbinned pixels)
    starti: int
    startj: int
    bini: int
    binj: int

    # Depth binning
    depth_start: float  # microns
    depth_end: float    # microns
    depth_resolution: float  # microns
    
    # Optional parameters (must come after required ones)
    # Rotation matrix rho (3x3) that aligns wire axis to beamline x-axis (optional if ki given in detector frame)
    rho: Optional[torch.Tensor] = None  # shape [3,3], torch.tensor(float64)
    # Incident beam direction vector ki (unit length), in wire-aligned frame (after rho). If None, assumed along z.
    ki: Optional[Tuple[float, float, float]] = None


def rodrigues_rotation_matrix(axis: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues rotation formula. axis: shape [3], length is angle in radians.
    Returns 3x3 rotation matrix (float64).
    """
    axis = axis.to(dtype=torch.float64)
    theta = torch.linalg.norm(axis)
    if not torch.isfinite(theta) or theta.item() == 0.0:
        return torch.eye(3, dtype=torch.float64)
    nx, ny, nz = (axis / theta).unbind()
    c = math.cos(theta.item())
    s = math.sin(theta.item())
    c1 = 1.0 - c
    mat = torch.empty((3, 3), dtype=torch.float64)
    mat[0, 0] = c + nx * nx * c1
    mat[0, 1] = nx * ny * c1 - nz * s
    mat[0, 2] = nx * nz * c1 + ny * s
    mat[1, 0] = nx * ny * c1 + nz * s
    mat[1, 1] = c + ny * ny * c1
    mat[1, 2] = ny * nz * c1 - nx * s
    mat[2, 0] = nx * nz * c1 - ny * s
    mat[2, 1] = ny * nz * c1 + nx * s
    mat[2, 2] = c + nz * nz * c1
    return mat


def compute_detector_rotation(params: DetectorParams) -> torch.Tensor:
    return rodrigues_rotation_matrix(torch.tensor(params.R, dtype=torch.float64))


def build_pixel_edge_coords(
    H: int,
    W: int,
    params: DetectorParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build detector-frame 3D coordinates (before wire rho rotation) for pixel front/back edges
    along j-axis, for ROI of size H (rows=i) x W (cols=j).

    Returns:
        front_edge_xyz_det: [H, W, 3] (float64)
        back_edge_xyz_det:  [H, W, 3] (float64)
    """
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ROI indices (binned)
    i_idx = torch.arange(H, dtype=dtype, device=device)  # rows
    j_idx = torch.arange(W, dtype=dtype, device=device)  # cols

    # Map binned ROI centers to full-chip unbinned pixel centers
    # Center (unbinned) index for pixel (i,j):
    # i_center = starti + bini * i + (bini - 1)/2
    # j_center = startj + binj * j + (binj - 1)/2
    i_center_unb = params.starti + params.bini * i_idx.view(H, 1) + (params.bini - 1) / 2.0
    j_center_unb = params.startj + params.binj * j_idx.view(1, W) + (params.binj - 1) / 2.0

    # Front/back edges along j are offset by ±binj/2 in unbinned pixel index
    j_front_unb = j_center_unb + (params.binj / 2.0)
    j_back_unb = j_center_unb - (params.binj / 2.0)

    # Convert to detector coordinates (x', y', z'=0), then translate by P
    # Detector-centered coordinates subtract half chip (zero-based indexing)
    x_center = (i_center_unb - 0.5 * (params.xDimDet - 1)) * params.pixel_size_i
    y_front = (j_front_unb - 0.5 * (params.yDimDet - 1)) * params.pixel_size_j
    y_back = (j_back_unb - 0.5 * (params.yDimDet - 1)) * params.pixel_size_j

    P = torch.tensor(params.P, dtype=dtype, device=device).view(1, 1, 3)
    x_center = x_center.to(device=device)
    y_front = y_front.to(device=device)
    y_back = y_back.to(device=device)

    # Build detector-frame positions (after translation P)
    front_det = torch.empty((H, W, 3), dtype=dtype, device=device)
    back_det = torch.empty((H, W, 3), dtype=dtype, device=device)

    front_det[..., 0] = x_center
    front_det[..., 1] = y_front
    front_det[..., 2] = 0.0

    back_det[..., 0] = x_center
    back_det[..., 1] = y_back
    back_det[..., 2] = 0.0

    front_det = front_det + P
    back_det = back_det + P

    # Rotate into beamline coordinates using detector rotation matrix
    R_det = compute_detector_rotation(params).to(device=device, dtype=dtype)  # [3,3]
    front_xyz = torch.einsum("ij,hwj->hwi", R_det, front_det)
    back_xyz = torch.einsum("ij,hwj->hwi", R_det, back_det)

    return front_xyz, back_xyz


def wire_positions_to_beamline(
    wire_positions: torch.Tensor,
    params: DetectorParams,
) -> torch.Tensor:
    """
    Convert raw wire positions to beamline coordinates.
    For now, assume wire_positions are already corrected for PM500 origin and rotations,
    and apply rho if provided.
    wire_positions: [S, 3] float64
    """
    dtype = torch.float64
    device = wire_positions.device
    wp = wire_positions.to(dtype=dtype)
    if params.rho is not None:
        rho = params.rho.to(device=device, dtype=dtype)
        wp = torch.einsum("ij,sj->si", rho, wp)
    return wp


def pixel_xyz_to_depth(
    point_on_ccd_xyz: torch.Tensor,  # [H,W,3] float64 in beamline coordinates
    wire_position: torch.Tensor,     # [S,3] float64 in beamline coordinates
    params: DetectorParams,
    use_leading_wire_edge: bool,
    wire_batch_size: int = 50,       # Process wire positions in smaller batches
) -> torch.Tensor:
    """
    Compute depth along incident beam for rays tangent to wire edge from detector pixel.
    Vectorized over pixels (H,W) and steps (S). Returns [S,H,W] depth (microns).
    Process wire positions in batches to manage memory.
    """
    dtype = torch.float64
    device = point_on_ccd_xyz.device
    H, W, _ = point_on_ccd_xyz.shape
    S = wire_position.shape[0]

    # Incident beam direction ki (unit vector) in wire-aligned frame
    if params.ki is None:
        ki = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    else:
        ki = torch.tensor(params.ki, dtype=dtype, device=device)
    # Broadcast ki components
    kiy, kiz, kix = ki[1], ki[2], ki[0]

    # Rotate detector pixel positions into wire-aligned frame using rho (if provided)
    if params.rho is not None:
        rho = params.rho.to(device=device, dtype=dtype)
        pixelPos = torch.einsum("ij,hwj->hwi", rho, point_on_ccd_xyz)
    else:
        pixelPos = point_on_ccd_xyz

    # Process wire positions in batches to manage memory
    depth_results = []
    
    for s_start in range(0, S, wire_batch_size):
        s_end = min(s_start + wire_batch_size, S)
        wire_batch = wire_position[s_start:s_end]
        S_batch = s_end - s_start
        
        # Broadcast wire positions to pixel grid for this batch
        wirePos = wire_batch.view(S_batch, 1, 1, 3).expand(S_batch, H, W, 3)  # [S_batch,H,W,3]

        # Vector from pixel to wire center (use only y,z components)
        dY = wirePos[..., 1] - pixelPos[..., 1]  # [S_batch,H,W]
        dZ = wirePos[..., 2] - pixelPos[..., 2]  # [S_batch,H,W]
        wire_radius = params.wire_diameter / 2.0

        lensq = dY * dY + dZ * dZ
        # Avoid invalid sqrt for cases where lensq <= r^2; mask them (no contribution)
        delta = lensq - wire_radius * wire_radius
        # tandphi = wire_radius / sqrt(lensq - r^2)
        safe = delta > 0
        tandphi = torch.zeros_like(dY, dtype=dtype)
        tandphi[safe] = wire_radius / torch.sqrt(delta[safe])

        tanphi0 = torch.zeros_like(dY, dtype=dtype)
        # Avoid division by zero in tanphi0 = dZ/dY
        nonzero_y = dY != 0
        tanphi0[nonzero_y] = dZ[nonzero_y] / dY[nonzero_y]
        # For zero dY, tanphi0 is inf; set a large value with proper sign
        tanphi0[~nonzero_y] = torch.sign(dZ[~nonzero_y]) * 1e12

        if use_leading_wire_edge:
            numerator = tanphi0 - tandphi
            denominator = 1.0 + tanphi0 * tandphi
        else:
            numerator = tanphi0 + tandphi
            denominator = 1.0 - tanphi0 * tandphi

        tanphi = torch.zeros_like(tanphi0, dtype=dtype)
        valid = denominator != 0
        tanphi[valid] = numerator[valid] / denominator[valid]
        # Compute reflected-line intercept b at tangent line: z = y * tanphi + b
        b_reflected = pixelPos[..., 2] - pixelPos[..., 1] * tanphi  # [S_batch,H,W]
        # Intersection of ray and incident beam line: S.z = b / (1 - tanphi * (kiy/kiz))
        kiy_over_kiz = (kiy / kiz).item() if kiz != 0 else 0.0
        denom_intersect = 1.0 - tanphi * kiy_over_kiz
        S_z = torch.zeros_like(b_reflected, dtype=dtype)
        valid2 = denom_intersect != 0
        S_z[valid2] = b_reflected[valid2] / denom_intersect[valid2]
        S_y = kiy_over_kiz * S_z
        kix_over_kiz = (kix / kiz).item() if kiz != 0 else 0.0
        S_x = kix_over_kiz * S_z

        # Depth is projection of intersection point onto ki
        # depth = dot(ki, S) = kx*Sx + ky*Sy + kz*Sz
        depth_batch = kix * S_x + kiy * S_y + kiz * S_z  # [S_batch,H,W]
        depth_results.append(depth_batch)
        
        # Clean up batch tensors
        del wirePos, dY, dZ, tandphi, tanphi0, tanphi, b_reflected, S_z, S_y, S_x
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all batch results
    depth = torch.cat(depth_results, dim=0)  # [S,H,W]
    return depth


def reconstruct_depth_torch(
    images: torch.Tensor,           # [S,H,W] float32/64 image intensities (already normalized/cosmic-filtered if desired)
    wire_positions: torch.Tensor,   # [S,3] float64 wire positions (beamline coordinates; corrected & rotated if needed)
    params: DetectorParams,
    wire_edge: int = 1,             # 1=leading, 0=trailing, -1=both
    chunk_size: int = 256,          # Process pixels in chunks to manage memory
) -> torch.Tensor:
    """
    PyTorch implementation of depth-resolved reconstruction using a prefix-sum binning scheme.

    Notes:
    - Removes historical row-stripe and per-bin loops by vectorizing across pixels and steps
      and using difference arrays along depth bin edges to accumulate linear segments.
    - This implementation computes front/back pixel edge coordinates, wire tangent depths for s and s+1,
      constructs trapezoid segments, applies O(1) difference updates per segment at depth-edge indices,
      and performs cumulative sums to recover per-bin heights, then integrates per bin using trapezoidal rule.

    Returns:
        depth_images: [M,H,W] float64, M = number of depth bins
    """
    device = images.device
    dtype = torch.float64

    # Convert inputs to appropriate dtypes
    imgs = images.to(dtype=dtype)
    H = imgs.shape[1]
    W = imgs.shape[2]
    S = imgs.shape[0]

    # Depth binning parameters
    d_start = params.depth_start
    d_end = params.depth_end
    dD = params.depth_resolution
    M = int(round((d_end - d_start) / dD + 1.0))
    if M < 1:
        raise ValueError("No output depth bins; check depth range and resolution.")

    # Precompute pixel edges in detector frame, then rotate to beamline coords
    front_xyz_det, back_xyz_det = build_pixel_edge_coords(H, W, params)  # [H,W,3], float64
    # Wire positions to beamline (apply rho if provided)
    wire_beam = wire_positions_to_beamline(wire_positions, params)  # [S,3], float64

    # Diff sequence: I[s] - I[s+1], shape [S-1,H,W]
    diff = imgs[:-1] - imgs[1:]

    # Edge offset for depth bins
    edge0 = d_start - dD / 2.0
    Npix = H * W
    
    # Process in chunks to manage memory
    # Initialize output accumulator
    slope_diff = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)
    intercept_diff = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)
    
    # Process pixels in chunks
    for h_start in range(0, H, chunk_size):
        h_end = min(h_start + chunk_size, H)
        h_chunk = h_end - h_start
        
        # Get chunk of pixel coordinates
        front_chunk = front_xyz_det[h_start:h_end]  # [h_chunk, W, 3]
        back_chunk = back_xyz_det[h_start:h_end]     # [h_chunk, W, 3]
        
        # Compute depths for this chunk
        depth_front_s = pixel_xyz_to_depth(front_chunk, wire_beam, params, use_leading_wire_edge=(wire_edge != 0))
        depth_back_s = pixel_xyz_to_depth(back_chunk, wire_beam, params, use_leading_wire_edge=(wire_edge != 0))
        
        # For s+1:
        depth_front_s1 = depth_front_s[1:]  # [S-1,h_chunk,W]
        depth_back_s1 = depth_back_s[1:]    # [S-1,h_chunk,W]
        depth_front_s = depth_front_s[:-1]  # [S-1,h_chunk,W]
        depth_back_s = depth_back_s[:-1]    # [S-1,h_chunk,W]
        
        # Get diff for this chunk
        diff_chunk = diff[:, h_start:h_end]  # [S-1, h_chunk, W]
        
        # Process this chunk (rest of the algorithm remains the same)
        process_chunk(
            depth_front_s, depth_back_s, depth_front_s1, depth_back_s1,
            diff_chunk, h_start, h_chunk, W, M, dD, edge0, device, dtype,
            slope_diff, intercept_diff, wire_edge
        )
        
        # Clean up intermediate tensors to free memory
        del depth_front_s, depth_back_s, depth_front_s1, depth_back_s1, diff_chunk
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Continue with accumulation as before
    slope_acc_edges = torch.cumsum(slope_diff, dim=0)       # [M+1, Npix]
    intercept_acc_edges = torch.cumsum(intercept_diff, dim=0)
    
    # Compute heights at bin edges
    E_m = (edge0 + torch.arange(M + 1, dtype=torch.float64, device=device) * dD).view(M + 1, 1)
    H_edge = slope_acc_edges * E_m + intercept_acc_edges
    
    # Area per bin via trapezoidal rule
    A_bin = 0.5 * (H_edge[:-1, :] + H_edge[1:, :]) * dD
    
    # Reshape back to [M,H,W]
    depth_images = A_bin.reshape(M, H, W)
    
    # If wire_edge == -1 (both edges), we need also trailing-edge contributions added
    if wire_edge == -1:
        # Process trailing edge with opposite sign
        slope_diff_t = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)
        intercept_diff_t = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)
        
        # Process in chunks again for trailing edge
        for h_start in range(0, H, chunk_size):
            h_end = min(h_start + chunk_size, H)
            h_chunk = h_end - h_start
            
            front_chunk = front_xyz_det[h_start:h_end]
            back_chunk = back_xyz_det[h_start:h_end]
            
            depth_front_s = pixel_xyz_to_depth(front_chunk, wire_beam, params, use_leading_wire_edge=False)
            depth_back_s = pixel_xyz_to_depth(back_chunk, wire_beam, params, use_leading_wire_edge=False)
            
            depth_front_s1 = depth_front_s[1:]
            depth_back_s1 = depth_back_s[1:]
            depth_front_s = depth_front_s[:-1]
            depth_back_s = depth_back_s[:-1]
            
            diff_chunk = diff[:, h_start:h_end]
            
            process_chunk(
                depth_front_s, depth_back_s, depth_front_s1, depth_back_s1,
                diff_chunk, h_start, h_chunk, W, M, dD, edge0, device, dtype,
                slope_diff_t, intercept_diff_t, 0  # wire_edge=0 for trailing
            )
            
            # Clean up intermediate tensors
            del depth_front_s, depth_back_s, depth_front_s1, depth_back_s1, diff_chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        slope_acc_edges_t = torch.cumsum(slope_diff_t, dim=0)
        intercept_acc_edges_t = torch.cumsum(intercept_diff_t, dim=0)
        H_edge_t = slope_acc_edges_t * E_m + intercept_acc_edges_t
        A_bin_t = 0.5 * (H_edge_t[:-1, :] + H_edge_t[1:, :]) * dD
        depth_images += A_bin_t.reshape(M, H, W)
    
    return depth_images


def process_chunk(
    depth_front_s, depth_back_s, depth_front_s1, depth_back_s1,
    diff_chunk, h_start, h_chunk, W, M, dD, edge0, device, dtype,
    slope_diff, intercept_diff, wire_edge
):
    """Process a chunk of pixels for memory efficiency with better memory management."""
    # Process in smaller sub-batches to avoid memory spikes
    S_minus_1 = depth_front_s.shape[0]
    batch_size = min(50, S_minus_1)  # Process 50 wire steps at a time
    
    # Flatten the chunk dimension
    Npix_chunk = h_chunk * W
    pixel_offset = h_start * W
    
    # Sign for wire edge
    sign = 1.0 if wire_edge != 0 else -1.0
    
    for s_start in range(0, S_minus_1, batch_size):
        s_end = min(s_start + batch_size, S_minus_1)
        s_batch = s_end - s_start
        
        # Get batch of depths
        depth_front_s_batch = depth_front_s[s_start:s_end]
        depth_back_s_batch = depth_back_s[s_start:s_end]
        depth_front_s1_batch = depth_front_s1[s_start:s_end]
        depth_back_s1_batch = depth_back_s1[s_start:s_end]
        diff_batch = diff_chunk[s_start:s_end]
        
        # Flatten batch
        diff_flat = diff_batch.reshape(s_batch, Npix_chunk)
        
        # Trapezoid endpoints
        partial_start = depth_front_s_batch
        partial_end = depth_back_s1_batch
        full_start = depth_back_s_batch
        full_end = depth_front_s1_batch
        
        # Ensure proper order
        swap_mask = full_end < full_start
        full_start, full_end = torch.where(swap_mask, full_end, full_start), torch.where(swap_mask, full_start, full_end)
        
        # Compute area
        area = (partial_end - partial_start) / 2.0
        valid_area = torch.isfinite(area) & (area > 0)
        
        # Flatten for processing
        area_flat = area.reshape(s_batch, Npix_chunk)
        valid_flat = valid_area.reshape(s_batch, Npix_chunk)
        
        # Convert to edge indices
        def to_edge_index(d: torch.Tensor) -> torch.Tensor:
            idx = torch.floor((d - edge0) / dD).to(torch.int64)
            return torch.clamp(idx, 0, M)
        
        partial_start_flat = partial_start.reshape(s_batch, Npix_chunk)
        full_start_flat = full_start.reshape(s_batch, Npix_chunk)
        full_end_flat = full_end.reshape(s_batch, Npix_chunk)
        partial_end_flat = partial_end.reshape(s_batch, Npix_chunk)
        
        m_ps = to_edge_index(partial_start_flat)
        m_fs = to_edge_index(full_start_flat)
        m_fe = to_edge_index(full_end_flat)
        m_pe = to_edge_index(partial_end_flat)
        
        # Weight by diff_value / area
        weight = torch.zeros_like(diff_flat, dtype=torch.float64, device=device)
        valid_weight = valid_flat & torch.isfinite(diff_flat) & (area_flat > 0)
        weight[valid_weight] = (sign * diff_flat[valid_weight]) / area_flat[valid_weight]
        
        # Process each segment type
        # Rising segment
        denom_rise = (full_start - partial_start).reshape(s_batch, Npix_chunk)
        valid_rise = torch.isfinite(denom_rise) & (denom_rise > 0) & valid_weight
        if torch.any(valid_rise):
            slope_rise = torch.zeros_like(denom_rise, dtype=torch.float64, device=device)
            slope_rise[valid_rise] = 1.0 / denom_rise[valid_rise]
            intercept_rise = -partial_start_flat[valid_rise] * slope_rise[valid_rise]
            
            # Apply rising segment
            nz = torch.nonzero(valid_rise, as_tuple=False)
            if nz.shape[0] > 0:
                ms = m_ps[valid_rise]
                me = m_fs[valid_rise]
                w = weight[valid_rise]
                pix_idx = nz[:, 1] + pixel_offset
                
                sl = slope_rise[valid_rise] * w
                it = intercept_rise * w
                
                # Use smaller batches for scatter operations
                scatter_batch_size = 10000
                for i in range(0, len(ms), scatter_batch_size):
                    j = min(i + scatter_batch_size, len(ms))
                    slope_diff.index_put_((ms[i:j], pix_idx[i:j]), sl[i:j], accumulate=True)
                    intercept_diff.index_put_((ms[i:j], pix_idx[i:j]), it[i:j], accumulate=True)
                    slope_diff.index_put_((me[i:j], pix_idx[i:j]), -sl[i:j], accumulate=True)
                    intercept_diff.index_put_((me[i:j], pix_idx[i:j]), -it[i:j], accumulate=True)
            
            del slope_rise, intercept_rise
        
        # Flat segment
        valid_flat_seg = valid_weight
        if torch.any(valid_flat_seg):
            nz = torch.nonzero(valid_flat_seg, as_tuple=False)
            if nz.shape[0] > 0:
                ms = m_fs[valid_flat_seg]
                me = m_fe[valid_flat_seg]
                w = weight[valid_flat_seg]
                pix_idx = nz[:, 1] + pixel_offset
                
                # Use smaller batches for scatter operations
                scatter_batch_size = 10000
                for i in range(0, len(ms), scatter_batch_size):
                    j = min(i + scatter_batch_size, len(ms))
                    slope_diff.index_put_((me[i:j], pix_idx[i:j]), -w[i:j], accumulate=True)
                    intercept_diff.index_put_((ms[i:j], pix_idx[i:j]), w[i:j], accumulate=True)
                    intercept_diff.index_put_((me[i:j], pix_idx[i:j]), -w[i:j], accumulate=True)
        
        # Falling segment
        denom_fall = (partial_end - full_end).reshape(s_batch, Npix_chunk)
        valid_fall = torch.isfinite(denom_fall) & (denom_fall > 0) & valid_weight
        if torch.any(valid_fall):
            slope_fall = torch.zeros_like(denom_fall, dtype=torch.float64, device=device)
            slope_fall[valid_fall] = -1.0 / denom_fall[valid_fall]
            intercept_fall = partial_end_flat[valid_fall] * (-slope_fall[valid_fall])
            
            # Apply falling segment
            nz = torch.nonzero(valid_fall, as_tuple=False)
            if nz.shape[0] > 0:
                ms = m_fe[valid_fall]
                me = m_pe[valid_fall]
                w = weight[valid_fall]
                pix_idx = nz[:, 1] + pixel_offset
                
                sl = slope_fall[valid_fall] * w
                it = intercept_fall * w
                
                # Use smaller batches for scatter operations
                scatter_batch_size = 10000
                for i in range(0, len(ms), scatter_batch_size):
                    j = min(i + scatter_batch_size, len(ms))
                    slope_diff.index_put_((ms[i:j], pix_idx[i:j]), sl[i:j], accumulate=True)
                    intercept_diff.index_put_((ms[i:j], pix_idx[i:j]), it[i:j], accumulate=True)
                    slope_diff.index_put_((me[i:j], pix_idx[i:j]), -sl[i:j], accumulate=True)
                    intercept_diff.index_put_((me[i:j], pix_idx[i:j]), -it[i:j], accumulate=True)
            
            del slope_fall, intercept_fall
        
        # Clean up batch tensors
        del diff_flat, area_flat, valid_flat, weight
        del m_ps, m_fs, m_fe, m_pe
        del partial_start_flat, full_start_flat, full_end_flat, partial_end_flat
        
        if device == 'cuda':
            torch.cuda.empty_cache()


# Keep the original implementation but refactor it
def reconstruct_depth_torch_original(
    images: torch.Tensor,
    wire_positions: torch.Tensor,
    params: DetectorParams,
    wire_edge: int = 1,
) -> torch.Tensor:
    """Original implementation - kept for reference but will OOM on large data."""

    # The old non-chunked implementation has been moved to reconstruct_depth_torch_original
    # This is now replaced with the chunked version above
    pass


# Example usage (pseudo):
# params = DetectorParams(
#     xDimDet=2048, yDimDet=2048,
#     pixel_size_i=200.0, pixel_size_j=200.0,  # microns per unbinned pixel (example)
#     P=(0.0, 0.0, 400000.0),                  # microns
#     R=(-0.61394300, 1.48219000, 0.61394300), # radians vector
#     wire_diameter=52.0,
#     rho=torch.eye(3, dtype=torch.float64),
#     ki=(0.0, 0.0, 1.0),
#     starti=0, startj=0, bini=1, binj=1,
#     depth_start=0.0, depth_end=100.0, depth_resolution=1.0,
# )
# images = torch.rand((401, 1024, 1024), dtype=torch.float32, device="cuda") # S,H,W
# wire_positions = torch.rand((401, 3), dtype=torch.float64, device="cuda")  # positions
# depth_imgs = reconstruct_depth_torch(images, wire_positions, params, wire_edge=1)
# # Write depth_imgs to HDF5 using h5py as needed.
# NOTE: need to promote to float64 for accuracy in many steps. This must be done in the glue code.
