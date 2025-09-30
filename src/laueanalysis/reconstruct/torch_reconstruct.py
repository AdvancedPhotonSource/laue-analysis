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
    # Rotation matrix rho (3x3) that aligns wire axis to beamline x-axis (optional if ki given in detector frame)
    rho: Optional[torch.Tensor] = None  # shape [3,3], torch.tensor(float64)
    # Incident beam direction vector ki (unit length), in wire-aligned frame (after rho). If None, assumed along z.
    ki: Optional[Tuple[float, float, float]] = None

    # ROI and binning (mapping binned ROI indices to full-chip unbinned pixels)
    starti: int
    startj: int
    bini: int
    binj: int

    # Depth binning
    depth_start: float  # microns
    depth_end: float    # microns
    depth_resolution: float  # microns


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
) -> torch.Tensor:
    """
    Compute depth along incident beam for rays tangent to wire edge from detector pixel.
    Vectorized over pixels (H,W) and steps (S). Returns [S,H,W] depth (microns).
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

    # Broadcast wire positions to pixel grid
    wirePos = wire_position.view(S, 1, 1, 3).expand(S, H, W, 3)  # [S,H,W,3]

    # Vector from pixel to wire center (use only y,z components)
    dY = wirePos[..., 1] - pixelPos[..., 1]  # [S,H,W]
    dZ = wirePos[..., 2] - pixelPos[..., 2]  # [S,H,W]
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
    b_reflected = pixelPos[..., 2] - pixelPos[..., 1] * tanphi  # [S,H,W]
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
    depth = kix * S_x + kiy * S_y + kiz * S_z  # [S,H,W]
    return depth


def reconstruct_depth_torch(
    images: torch.Tensor,           # [S,H,W] float32/64 image intensities (already normalized/cosmic-filtered if desired)
    wire_positions: torch.Tensor,   # [S,3] float64 wire positions (beamline coordinates; corrected & rotated if needed)
    params: DetectorParams,
    wire_edge: int = 1,             # 1=leading, 0=trailing, -1=both
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

    # Compute depths at s and s+1 for front/back edges
    # Shape: [S,H,W]
    depth_front_s = pixel_xyz_to_depth(front_xyz_det, wire_beam, params, use_leading_wire_edge=(wire_edge != 0))
    depth_back_s = pixel_xyz_to_depth(back_xyz_det, wire_beam, params, use_leading_wire_edge=(wire_edge != 0))

    # For s+1:
    depth_front_s1 = depth_front_s[1:]  # [S-1,H,W]
    depth_back_s1 = depth_back_s[1:]    # [S-1,H,W]
    depth_front_s = depth_front_s[:-1]  # [S-1,H,W]
    depth_back_s = depth_back_s[:-1]    # [S-1,H,W]

    # Trapezoid endpoints (partial_start, full_start, full_end, partial_end)
    # as per C code conventions:
    partial_start = depth_front_s
    partial_end = depth_back_s1
    full_start = depth_back_s
    full_end = depth_front_s1

    # If full_end < full_start, swap to ensure proper order
    swap_mask = full_end < full_start
    fs = torch.where(swap_mask, full_end, full_start)
    fe = torch.where(swap_mask, full_start, full_end)
    full_start = fs
    full_end = fe

    # Clip trapezoid to depth range
    # We'll operate on edge indices; edges run from E0 = D_start - dD/2 to EM = D_start - dD/2 + M*dD
    edge0 = d_start - dD / 2.0

    # Compute trapezoid area (height max 1). If area invalid/negative, skip.
    # area = (full_end + partial_end - full_end - partial_start) / 2 = (partial_end - partial_start) / 2
    area = (partial_end - partial_start) / 2.0  # [S-1,H,W]
    valid_area = torch.isfinite(area) & (area > 0)

    # For trailing edge, invert intensity contribution (diff_value sign)
    sign = 1.0 if wire_edge != 0 else -1.0  # leading: +1; trailing: -1

    # Flatten pixel grid to simplify scatter_add operations
    Npix = H * W
    diff_flat = diff.reshape(diff.shape[0], Npix)  # [S-1, Npix]
    area_flat = area.reshape(area.shape[0], Npix)
    valid_flat = valid_area.reshape(valid_area.shape[0], Npix)

    # Convert trapezoid endpoints to edge indices (integer in [0..M])
    def to_edge_index(d: torch.Tensor) -> torch.Tensor:
        idx = torch.floor((d - edge0) / dD).to(torch.int64)
        idx = torch.clamp(idx, 0, M)
        return idx

    m_ps = to_edge_index(partial_start.reshape(partial_start.shape[0], Npix))
    m_fs = to_edge_index(full_start.reshape(full_start.shape[0], Npix))
    m_fe = to_edge_index(full_end.reshape(full_end.shape[0], Npix))
    m_pe = to_edge_index(partial_end.reshape(partial_end.shape[0], Npix))

    # Segment parameters for h(d):
    # Rising segment: h(d) = (d - ps)/(fs - ps), slope = 1/(fs-ps), intercept = -ps/(fs-ps)
    denom_rise = (full_start - partial_start).reshape(full_start.shape[0], Npix)
    valid_rise = torch.isfinite(denom_rise) & (denom_rise > 0)
    slope_rise = torch.zeros_like(denom_rise, dtype=torch.float64, device=device)
    intercept_rise = torch.zeros_like(denom_rise, dtype=torch.float64, device=device)
    slope_rise[valid_rise] = 1.0 / denom_rise[valid_rise]
    intercept_rise[valid_rise] = -partial_start.reshape(partial_start.shape[0], Npix)[valid_rise] * slope_rise[valid_rise]

    # Flat segment: h(d) = 1 on [fs, fe], slope=0, intercept=1
    slope_flat = torch.zeros_like(slope_rise)
    intercept_flat = torch.ones_like(intercept_rise)

    # Falling segment: h(d) = (pe - d)/(pe - fe), slope = -1/(pe - fe), intercept = pe/(pe - fe)
    denom_fall = (partial_end - full_end).reshape(partial_end.shape[0], Npix)
    valid_fall = torch.isfinite(denom_fall) & (denom_fall > 0)
    slope_fall = torch.zeros_like(denom_fall, dtype=torch.float64, device=device)
    intercept_fall = torch.zeros_like(denom_fall, dtype=torch.float64, device=device)
    slope_fall[valid_fall] = -1.0 / denom_fall[valid_fall]
    intercept_fall[valid_fall] = partial_end.reshape(partial_end.shape[0], Npix)[valid_fall] * (-slope_fall[valid_fall])

    # Weight by diff_value / area and wire edge sign
    # Contribution per trapezoid is diff * (area_in_bin / total_area)
    weight = torch.zeros_like(diff_flat, dtype=torch.float64, device=device)
    valid_weight = valid_flat & torch.isfinite(diff_flat) & (area_flat > 0)
    weight[valid_weight] = (sign * diff_flat[valid_weight]) / area_flat[valid_weight]

    # Prepare difference arrays along depth edges for slope and intercept
    # We'll store flattened pixels for difference arrays: [M+1, Npix]
    slope_diff = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)
    intercept_diff = torch.zeros((M + 1, Npix), dtype=torch.float64, device=device)

    def scatter_segment(m_start: torch.Tensor, m_end: torch.Tensor, slope: torch.Tensor, intercept: torch.Tensor, seg_valid: torch.Tensor):
        """
        Apply difference updates along depth edges:
        slope_diff[m_start] += slope * weight
        slope_diff[m_end]   -= slope * weight
        intercept_diff[m_start] += intercept * weight
        intercept_diff[m_end]   -= intercept * weight
        Only for seg_valid and weight-valid pixels.
        """
        mask = seg_valid & valid_weight
        if not torch.any(mask):
            return
        ms = m_start[mask]  # [K]
        me = m_end[mask]    # [K]
        w = weight[mask]    # [K]
        sl = slope[mask] * w
        it = intercept[mask] * w

        # Flatten pixel indices for gather/scatter
        # Each mask corresponds to multiple trapezoids across steps and pixels
        # We need to map them to positions in [M+1, Npix] for scatter_add.
        # To scatter along depth dimension, we use advanced indexing with (edge_idx, pixel_linear_idx)
        # pixel_linear_idx must be known: from mask indices, derive pixel indices
        # Build pixel indices from mask nonzero positions
        # mask comes from 2D [S-1, Npix]; we need corresponding pixel index vector
        # To get pixel indices, compute nonzero of mask and derive col (pix) via modulo
        nz = torch.nonzero(mask, as_tuple=False)  # [K,2] with (s_idx, pix_idx)
        pix_idx = nz[:, 1]  # [K]

        # Scatter updates at start edge
        slope_diff.index_put_((ms, pix_idx), sl, accumulate=True)
        intercept_diff.index_put_((ms, pix_idx), it, accumulate=True)
        # Scatter negative updates at end edge
        slope_diff.index_put_((me, pix_idx), -sl, accumulate=True)
        intercept_diff.index_put_((me, pix_idx), -it, accumulate=True)

    # Apply segments
    scatter_segment(m_ps, m_fs, slope_rise, intercept_rise, valid_rise)
    scatter_segment(m_fs, m_fe, slope_flat, intercept_flat, torch.ones_like(valid_flat, dtype=torch.bool, device=device))
    scatter_segment(m_fe, m_pe, slope_fall, intercept_fall, valid_fall)

    # Accumulate along depth edges (prefix sums)
    slope_acc_edges = torch.cumsum(slope_diff, dim=0)       # [M+1, Npix]
    intercept_acc_edges = torch.cumsum(intercept_diff, dim=0)

    # Compute heights at bin edges H_edge(m) = slope_acc * E_m + intercept_acc
    # Edge depths E_m from edge0 + m*dD, m=0..M
    E_m = (edge0 + torch.arange(M + 1, dtype=torch.float64, device=device) * dD).view(M + 1, 1)  # [M+1,1]
    H_edge = slope_acc_edges * E_m + intercept_acc_edges  # [M+1, Npix]

    # Area per bin via trapezoidal rule: A_bin[m] = (H_edge[m] + H_edge[m+1])/2 * dD
    A_bin = 0.5 * (H_edge[:-1, :] + H_edge[1:, :]) * dD  # [M, Npix]

    # Reshape back to [M,H,W]
    depth_images = A_bin.reshape(M, H, W)  # float64

    # If wire_edge == -1 (both edges), we need also trailing-edge contributions added
    if wire_edge == -1:
        # Compute trailing-edge contributions by recomputing sign and repeating relevant steps
        trailing_params = params  # same params
        trailing_sign = -1.0
        weight_trailing = torch.zeros_like(diff_flat, dtype=torch.float64, device=device)
        weight_trailing[valid_weight] = (trailing_sign * diff_flat[valid_weight]) / area_flat[valid_weight]

        # Reuse precomputed segment indices and slopes/intercepts, but with new weights
        slope_diff_t = torch.zeros_like(slope_diff)
        intercept_diff_t = torch.zeros_like(intercept_diff)

        def scatter_segment_trailing(m_start, m_end, slope, intercept, seg_valid, w_tr):
            mask = seg_valid & valid_weight
            if not torch.any(mask):
                return
            ms = m_start[mask]
            me = m_end[mask]
            w = w_tr[mask]
            sl = slope[mask] * w
            it = intercept[mask] * w
            nz = torch.nonzero(mask, as_tuple=False)
            pix_idx = nz[:, 1]
            slope_diff_t.index_put_((ms, pix_idx), sl, accumulate=True)
            intercept_diff_t.index_put_((ms, pix_idx), it, accumulate=True)
            slope_diff_t.index_put_((me, pix_idx), -sl, accumulate=True)
            intercept_diff_t.index_put_((me, pix_idx), -it, accumulate=True)

        scatter_segment_trailing(m_ps, m_fs, slope_rise, intercept_rise, valid_rise, weight_trailing)
        scatter_segment_trailing(m_fs, m_fe, slope_flat, intercept_flat, torch.ones_like(valid_flat, dtype=torch.bool, device=device), weight_trailing)
        scatter_segment_trailing(m_fe, m_pe, slope_fall, intercept_fall, valid_fall, weight_trailing)

        slope_acc_edges_t = torch.cumsum(slope_diff_t, dim=0)
        intercept_acc_edges_t = torch.cumsum(intercept_diff_t, dim=0)
        H_edge_t = slope_acc_edges_t * E_m + intercept_acc_edges_t
        A_bin_t = 0.5 * (H_edge_t[:-1, :] + H_edge_t[1:, :]) * dD
        depth_images += A_bin_t.reshape(M, H, W)

    return depth_images


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