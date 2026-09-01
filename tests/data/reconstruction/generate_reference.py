# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Deterministic CPU wire-scan reconstruction reference.

This module builds a small synthetic wire scan whose contents are fully
determined by a fixed seed, runs ``reconstructN_cpu`` on it, and stores the
depth-resolved output as the numerical acceptance reference for the build
migration (see ``BUILD_DEPLOYMENT_PLAN.md``, Phase 0 and Phase 3).

The synthetic scan is not a physical simulation. It places three Gaussian
spots on the detector and switches each one off once the wire's leading edge
passes the depth assigned to that spot, using the same pixel-to-depth
geometry as ``WireScan.c`` (``pixel_to_point_xyz`` and
``pixel_xyz_to_depth``). That is enough for the program to exercise its full
input, geometry, depth-binning, and HDF5 output paths with a stable answer.

Run this file directly to regenerate ``cpu_reference.npz`` and
``cpu_reference.json``. Only do that on purpose: the reference is an
acceptance contract and regenerating it is a scientific decision, not a
build step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
GEOMETRY_FILE = HERE.parent / "geo" / "geoN_2022-03-29_14-15-05.xml"
REFERENCE_NPZ = HERE / "cpu_reference.npz"
REFERENCE_JSON = HERE / "cpu_reference.json"

SEED = 20260828
DETECTOR = 0
FULL_PIXELS = 2048           # un-binned detector pixels along each axis
DETECTOR_SIZE_UM = 409.6e3   # detector edge length (micron)
BINNING = 16                 # binned ROI is 128 x 128
WIRE_Y_UM = 1000.0           # wire height above the sample (raw positioner frame, micron)
WIRE_Z_START_UM = -80.0
WIRE_Z_STEP_UM = 2.0
N_SCAN_IMAGES = 81           # images that carry wire positions (slice 0 is skipped by the program)
WIRE_DIAMETER_UM = 101.5

DEPTH_RANGE_UM = (-25.0, 25.0)
DEPTH_RESOLUTION_UM = 1.0
PERCENT_BRIGHTEST = 100.0
OUTPUT_PIXEL_TYPE = 5        # double, so the reference carries no integer rounding
NUM_THREADS = 1

# (binned column i, binned row j, sigma in binned pixels, amplitude, source depth in micron)
SPOTS = (
    (64.0, 64.0, 2.5, 4000.0, 0.0),
    (52.0, 70.0, 2.0, 3000.0, -10.0),
    (68.0, 58.0, 3.0, 2500.0, 12.0),
)
BACKGROUND = 20.0

# Geometry constants copied from detector 0 and the wire section of GEOMETRY_FILE.
DETECTOR_P_UM = np.array([28.720, 3.010, 513.097]) * 1e3
DETECTOR_R = np.array([-1.20127231, -1.21381742, -1.21879073])
WIRE_R = np.array([0.0045, -0.00684, -3.375e-05])
WIRE_AXIS = np.array([0.99999608, 0.0, 0.0028000002])


def rodrigues(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector, matching ``rotationMatFromAxis``."""
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    nx, ny, nz = vector / angle
    c, s = np.cos(angle), np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + nx * nx * c1, nx * ny * c1 - nz * s, nx * nz * c1 + ny * s],
            [nx * ny * c1 + nz * s, c + ny * ny * c1, ny * nz * c1 - nx * s],
            [nx * nz * c1 - ny * s, ny * nz * c1 + nx * s, c + nz * nz * c1],
        ]
    )


class WireGeometry:
    """NumPy replica of the parts of ``geo2calibration`` used to place spots."""

    def __init__(self) -> None:
        self.detector_rotation = rodrigues(DETECTOR_R)
        self.wire_rotation = rodrigues(WIRE_R)
        axis_r = self.wire_rotation @ WIRE_AXIS
        axis_r /= np.linalg.norm(axis_r)
        rvec = np.array([0.0, axis_r[2], -axis_r[1]])
        sin_theta = np.linalg.norm(rvec)
        rvec *= np.arcsin(sin_theta) / sin_theta
        self.rho = rodrigues(rvec)
        self.ki = self.rho @ np.array([0.0, 0.0, 1.0])

    def pixel_xyz(self, i_binned: np.ndarray, j_binned: np.ndarray) -> np.ndarray:
        """Beam-line coordinates of binned pixel centres, following ``pixel_to_point_xyz``."""
        # ``depth_resolve`` loops rows as ``pixel.i`` and columns as ``pixel.j``;
        # ``pixel_to_point_xyz`` then swaps them so the detector x axis runs
        # along image columns.  ``i_binned`` here is the column, ``j_binned``
        # the row, so no further swap is needed before un-binning.
        ci = i_binned * BINNING + (BINNING - 1) / 2.0
        cj = j_binned * BINNING + (BINNING - 1) / 2.0
        pitch = DETECTOR_SIZE_UM / FULL_PIXELS
        x = (ci - 0.5 * (FULL_PIXELS - 1)) * pitch + DETECTOR_P_UM[0]
        y = (cj - 0.5 * (FULL_PIXELS - 1)) * pitch + DETECTOR_P_UM[1]
        z = np.full_like(x, DETECTOR_P_UM[2])
        return np.stack([x, y, z], axis=-1) @ self.detector_rotation.T

    def wire_to_beamline(self, wire_raw: np.ndarray) -> np.ndarray:
        return (wire_raw @ self.wire_rotation.T) @ self.rho.T

    def depth(self, pixel_xyz: np.ndarray, wire_beamline: np.ndarray) -> np.ndarray:
        """Leading-edge depth for each pixel, following ``pixel_xyz_to_depth``."""
        p = pixel_xyz @ self.rho.T
        dy = wire_beamline[1] - p[..., 1]
        dz = wire_beamline[2] - p[..., 2]
        radius = WIRE_DIAMETER_UM / 2.0
        tan_phi0 = dz / dy
        tan_dphi = radius / np.sqrt(dy * dy + dz * dz - radius * radius)
        tan_phi = (tan_phi0 - tan_dphi) / (1.0 + tan_phi0 * tan_dphi)
        b = p[..., 2] - p[..., 1] * tan_phi
        sz = b / (1.0 - tan_phi * self.ki[1] / self.ki[2])
        s = np.stack([self.ki[0] / self.ki[2] * sz, self.ki[1] / self.ki[2] * sz, sz], axis=-1)
        return s @ self.ki


def slice_wire_z(k: np.ndarray | int) -> np.ndarray:
    """Wire z (raw frame, micron) at which stored slice ``k`` was recorded."""
    return WIRE_Z_START_UM + WIRE_Z_STEP_UM * np.asarray(k, dtype=float)


def wire_positions_raw() -> np.ndarray:
    """Raw positioner wire-position vectors as stored in the file.

    ``HDF5ReadDoubleVector`` discards the first entry and ``readHDF5header``
    then requires one more entry than there are slices, so the stored vectors
    hold ``N_SCAN_IMAGES + 3`` values. ``setup_depth_images`` pairs scan image
    ``f`` (stored slice ``f + 1``) with stored entries ``f + 2`` and ``f + 3``,
    and the program attributes the intensity lost between slices ``f + 1`` and
    ``f + 2`` to the wire travelling between those two entries. Stored entry
    ``j`` therefore holds the position at which slice ``j - 1`` was recorded.
    """
    j = np.arange(N_SCAN_IMAGES + 3, dtype=float)
    z = slice_wire_z(j - 1.0)
    return np.stack([np.zeros_like(z), np.full_like(z, WIRE_Y_UM), z], axis=-1)


def synthetic_images() -> tuple[np.ndarray, np.ndarray]:
    """Return (images[N+1, n, n] uint16, wire_raw[N+3, 3])."""
    n = FULL_PIXELS // BINNING
    geometry = WireGeometry()
    jj, ii = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    pixel_xyz = geometry.pixel_xyz(ii, jj)
    wire_raw = wire_positions_raw()
    rng = np.random.default_rng(SEED)

    spot_profiles = []
    for ci, cj, sigma, amplitude, _ in SPOTS:
        spot_profiles.append(amplitude * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * sigma**2)))

    images = np.empty((N_SCAN_IMAGES + 1, n, n), dtype=np.uint16)
    for k in range(N_SCAN_IMAGES + 1):
        frame = np.full((n, n), BACKGROUND)
        if k == 0:
            # Intensity map: every spot fully on, so all pixels pass the cutoff.
            for profile in spot_profiles:
                frame += profile
        else:
            wire_here = np.array([0.0, WIRE_Y_UM, float(slice_wire_z(k))])
            depth_here = geometry.depth(pixel_xyz, geometry.wire_to_beamline(wire_here))
            for profile, (_, _, _, _, source_depth) in zip(spot_profiles, SPOTS):
                frame += profile * (depth_here < source_depth)
        frame = rng.poisson(frame).astype(np.float64)
        images[k] = np.clip(frame, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return images, wire_raw


def write_input_file(path: Path) -> np.ndarray:
    """Write the synthetic wire scan to ``path`` and return the raw wire positions."""
    images, wire_raw = synthetic_images()
    n = images.shape[1]
    with h5py.File(path, "w") as f:
        f.attrs["file_name"] = np.bytes_(b"synthetic_wire_scan.h5")
        f.attrs["file_time"] = np.bytes_(b"2026-08-28T12:00:00-05:00")
        facility = f.create_group("Facility")
        facility.create_dataset("facility_name", data=np.array([b"APS"], dtype="S3"))
        facility.create_dataset("facility_beamline", data=np.array([b"34ID-E"], dtype="S6"))
        entry = f.create_group("entry1")
        data = entry.create_group("data")
        dset = data.create_dataset("data", data=images)
        dset.attrs["signal"] = np.int32(1)
        entry.create_dataset("depth", data=np.array([0.0]))
        entry.create_dataset("scanNum", data=np.array([1], dtype=np.int32))
        detector = entry.create_group("detector")
        for name, value in (
            ("Nx", FULL_PIXELS), ("Ny", FULL_PIXELS),
            ("startx", 0), ("starty", 0),
            ("endx", FULL_PIXELS - 1), ("endy", FULL_PIXELS - 1),
            ("binx", BINNING), ("biny", BINNING),
        ):
            detector.create_dataset(name, data=np.array([value], dtype=np.int32))
        detector.create_dataset("exposure", data=np.array([1.0]))
        detector.create_dataset("ID", data=np.array([b"PE1621 723-3335"], dtype="S15"))
        sample = entry.create_group("sample")
        sample.create_dataset("incident_energy", data=np.array([20.0]))
        for name in ("sampleX", "sampleY", "sampleZ"):
            sample.create_dataset(name, data=np.array([0.0]))
        wire = entry.create_group("wire")
        wire.create_dataset("wireX", data=wire_raw[:, 0])
        wire.create_dataset("wireY", data=wire_raw[:, 1])
        wire.create_dataset("wireZ", data=wire_raw[:, 2])
        wire.create_dataset("wirescan", data=wire_raw[:, 2])
    assert n == FULL_PIXELS // BINNING
    return wire_raw


def run_reconstruction(executable: str, input_file: Path, output_base: Path, num_threads: int = NUM_THREADS):
    """Run the CPU program through the package API with the fixed reference parameters."""
    from lauelab.reconstruct import reconstruct

    return reconstruct(
        str(input_file),
        str(output_base),
        str(GEOMETRY_FILE),
        DEPTH_RANGE_UM,
        resolution=DEPTH_RESOLUTION_UM,
        verbose=1,
        percent_brightest=PERCENT_BRIGHTEST,
        wire_edge="leading",
        memory_limit_mb=256,
        executable=executable,
        output_pixel_type=OUTPUT_PIXEL_TYPE,
        detector_number=DETECTOR,
        num_threads=num_threads,
    )


def load_outputs(output_base: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``<base><n>.h5`` files in index order; return (stack, depths)."""
    n_depths = int(round((DEPTH_RANGE_UM[1] - DEPTH_RANGE_UM[0]) / DEPTH_RESOLUTION_UM)) + 1
    frames, depths = [], []
    for index in range(n_depths):
        path = Path(f"{output_base}{index}.h5")
        with h5py.File(path, "r") as f:
            frames.append(np.asarray(f["entry1/data/data"], dtype=np.float64))
            depths.append(float(np.asarray(f["entry1/depth"]).ravel()[0]))
    return np.stack(frames), np.asarray(depths)


def sha256_of(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run_lines(cmd: list[str]) -> list[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return []
    return [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]


def executable_provenance(executable: str) -> dict:
    """Compiler stamp and NEEDED libraries of the binary (no paths are recorded)."""
    comments = [line.split("]", 1)[-1].strip() for line in _run_lines(["readelf", "-p", ".comment", executable]) if "]" in line]
    needed = [line.split("[", 1)[-1].rstrip("]") for line in _run_lines(["readelf", "-d", executable]) if "NEEDED" in line]
    return {"compiler": comments, "needed": needed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executable", default=None, help="reconstructN_cpu to use (default: package binary)")
    parser.add_argument("--workdir", default=None, help="keep intermediate files here instead of a temp dir")
    args = parser.parse_args(argv)

    import tempfile
    from lauelab.reconstruct import find_executable

    executable = args.executable or find_executable()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="recon_ref_"))
    workdir.mkdir(parents=True, exist_ok=True)
    input_file = workdir / "synthetic_wire_scan.h5"
    write_input_file(input_file)
    input_sha = hashlib.sha256(input_file.read_bytes()).hexdigest()

    output_base = workdir / "out" / "recon_"
    result = run_reconstruction(executable, input_file, output_base)
    if not result.success:
        sys.stderr.write(result.log + "\n" + (result.error or "") + "\n")
        return 1
    stack, depths = load_outputs(output_base)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    provenance = {
        "generated_at_commit": git_commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "toolchain": executable_provenance(str(executable)),
        "seed": SEED,
        "input_sha256": input_sha,
        "output_sha256": sha256_of(stack),
        "output_shape": list(stack.shape),
        "depth_range_um": list(DEPTH_RANGE_UM),
        "depth_resolution_um": DEPTH_RESOLUTION_UM,
        "num_threads": NUM_THREADS,
        "comparison": {"rtol": 1e-12, "atol": 1e-9, "note": "float64 output; expected to match bit-for-bit on the same toolchain"},
        "note": "Generated from a synthetic input; no measured data. Machine paths are intentionally not recorded.",
    }
    np.savez_compressed(REFERENCE_NPZ, depth_um=depths, images=stack)
    REFERENCE_JSON.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    print(f"wrote {REFERENCE_NPZ} ({REFERENCE_NPZ.stat().st_size / 1e6:.2f} MB); workdir={workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
