"""Generate the synthetic Laue frames and LaueGo reference outputs used by the tests.

The frames are simulated: for each named orientation set, the package's own
``simulate_reflections`` places Ni reflections on detector 0 of the public
geometry file, each reflection becomes a Gaussian spot on a constant background
(``BACKGROUND_NOISE_LAMBDA`` adds Poisson noise if a noisy variant is ever
needed; it is zero so the frames stay small). The reference outputs come from the LaueGo
command-line programs (``peaksearch``, ``pix2qs``, ``euler``) run through
``laueanalysis.indexing.lauego`` on those frames, so the tests compare the
in-process indexer against an independent implementation.

Everything is deterministic (fixed seed, fixed orientations). Regenerate only
after a deliberate change to the fixtures:

    python tests/data/synthetic/generate.py

The generator does not record machine-specific paths.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GEOMETRY_FILE = REPO / "tests" / "data" / "geo" / "geoN_2022-03-29_14-15-05.xml"
CRYSTAL_FILE = REPO / "tests" / "config" / "Ni.xml"
FRAMES_DIR = HERE / "frames"
BASELINE_DIR = HERE / "baseline"
PROVENANCE = HERE / "provenance.json"

SEED = 20260828
DETECTOR_ID = "PE1621 723-3335"
NI_A_NM = 0.35238
ENERGY_RANGE_KEV = (6.0, 30.0)
SPOT_SIGMA_PX = 1.8
BACKGROUND = 20
BACKGROUND_NOISE_LAMBDA = 0.0  # constant background: the frames compress to a few tens of KB; peak search still thresholds from the spot statistics
SAMPLE_NAME = "synthetic Ni"

# Frame name -> list of grains; each grain is (rotation axis, angle in degrees, amplitude scale).
FRAMES = {
    "synthetic_ni_grain_a": [((1.0, 0.3, -0.2), 37.0, 1.0)],
    "synthetic_ni_grain_b": [((-0.4, 1.0, 0.5), 112.0, 1.0)],
    "synthetic_ni_two_grains": [((1.0, 0.3, -0.2), 37.0, 1.0), ((0.2, -0.7, 1.0), 64.0, 0.6)],
    "synthetic_ni_empty": [],
}
# Frames stored as a sub-region of the detector: name -> (startx, starty, width, height).
ROI = {"synthetic_ni_empty": (896, 896, 256, 256)}

# Peak-search and indexing settings shared by the CLI baseline and the tests.
PEAK_SEARCH = dict(boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
                   threshold=None, threshold_ratio=4.0, peak_shape="L", max_peaks=200)
INDEXING = dict(index_kev_max_calc=17.2, index_kev_max_test=35.0, index_angle_tolerance=0.1,
                index_cone=72.0, index_h=0, index_k=0, index_l=1)


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def reflections_for(grains):
    from laueanalysis.analysis import lattice_params_to_reciprocal, simulate_reflections
    from laueanalysis.indexing import load_crystal, load_geometry

    crystal = load_crystal(CRYSTAL_FILE)
    geometry = load_geometry(GEOMETRY_FILE)
    detector = geometry.detector(geometry.find_detector(DETECTOR_ID))
    recip0 = lattice_params_to_reciprocal(NI_A_NM, NI_A_NM, NI_A_NM, 90, 90, 90)
    spots = []
    for axis, angle, scale in grains:
        recip = (rotation_matrix(axis, angle) @ recip0.T).T
        sim = simulate_reflections(crystal, recip, detector, energy_range_kev=ENERGY_RANGE_KEV)
        for xy, energy in zip(sim.detector_xy, sim.energy_kev):
            # Brighter at low energy: a simple, monotonic stand-in for a real spectrum.
            amplitude = scale * 6000.0 * (ENERGY_RANGE_KEV[0] / energy) ** 0.5
            spots.append((float(xy[0]), float(xy[1]), amplitude))
    return detector, spots


def render_frame(detector, spots, rng, roi=None) -> np.ndarray:
    startx, starty, nx, ny = roi if roi else (0, 0, detector.nx, detector.ny)
    image = np.full((ny, nx), float(BACKGROUND))
    yy, xx = np.mgrid[starty:starty + ny, startx:startx + nx]
    half = int(6 * SPOT_SIGMA_PX)
    for x, y, amplitude in spots:
        cx, cy = int(round(x)) - startx, int(round(y)) - starty
        xs = slice(max(cx - half, 0), min(cx + half + 1, nx))
        ys = slice(max(cy - half, 0), min(cy + half + 1, ny))
        if xs.start >= xs.stop or ys.start >= ys.stop:
            continue
        dx = xx[ys, xs] - x
        dy = yy[ys, xs] - y
        image[ys, xs] += amplitude * np.exp(-(dx * dx + dy * dy) / (2 * SPOT_SIGMA_PX**2))
    image += rng.poisson(BACKGROUND_NOISE_LAMBDA, size=image.shape)
    return np.clip(np.rint(image), 0, 65535).astype(np.uint16)


def write_frame(path: Path, image: np.ndarray, scan_number: int, detector, roi=None) -> None:
    ny, nx = image.shape
    startx, starty = (roi[0], roi[1]) if roi else (0, 0)
    with h5py.File(path, "w") as f:
        f.attrs["file_time"] = np.bytes_(b"2026-08-28T12:00:00-05:00")
        f.create_dataset("entry1/data/data", data=image, compression="gzip", compression_opts=9,
                         shuffle=True, chunks=(256, nx))
        def text(name: str, value: str) -> None:
            # Null-terminated fixed-length strings, as the beamline writer produces.
            f.create_dataset(name, data=np.array([value.encode()], dtype=f"S{len(value) + 1}"))

        det = f.create_group("entry1/detector")
        text("entry1/detector/ID", DETECTOR_ID)
        for name, value in (("Nx", detector.nx), ("Ny", detector.ny), ("startx", startx), ("starty", starty),
                            ("endx", startx + nx - 1), ("endy", starty + ny - 1), ("binx", 1), ("biny", 1)):
            det.create_dataset(name, data=np.array([value], dtype=np.int32))
        det.create_dataset("exposure", data=np.array([1.0]))
        text("entry1/sample/name", SAMPLE_NAME)
        text("entry1/user/name", "synthetic")
        text("entry1/title", "synthetic Laue frame")
        f.create_dataset("entry1/scanNum", data=np.array([scan_number], dtype=np.int32))
        f.create_dataset("entry1/microDiffraction/CCDshutter", data=np.array([1], dtype=np.int16))


def run_lauego_baseline(frame: Path, out_dir: Path) -> dict:
    from laueanalysis.indexing import lauego

    work = Path(tempfile.mkdtemp(prefix="lauego_"))
    result = lauego(str(frame), str(work), str(GEOMETRY_FILE), str(CRYSTAL_FILE),
                    **PEAK_SEARCH, **INDEXING, generate_xml=False)
    if not result.success:
        raise RuntimeError(f"lauego failed on {frame.name}:\n{result.log}\n{result.error}")
    copied = {}
    for kind in ("peaks", "p2q", "index"):
        src = Path(result.output_files.get(kind, "")) if result.output_files.get(kind) else None
        if src is not None and src.is_file():
            dst = out_dir / kind / f"{kind}_{frame.stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(scrub_paths(src.read_text(), work))
            copied[kind] = dst.name
    shutil.rmtree(work)
    return {"n_peaks": result.n_peaks_found, "n_patterns": result.n_patterns_found,
            "n_indexed": result.n_indexed, "files": copied}


def scrub_paths(text: str, work: Path) -> str:
    """Replace machine-specific paths the LaueGo programs write into their headers."""
    import sys

    text = text.replace(str(work), "<work>")
    text = text.replace(str(Path(sys.prefix)), "<env>")  # before <repo>: the env may live inside the checkout
    text = text.replace(str(REPO), "<repo>")
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import laueanalysis
    from laueanalysis.indexing._liblaue import version as liblaue_version

    rng = np.random.default_rng(SEED)
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(BASELINE_DIR, ignore_errors=True)
    FRAMES_DIR.mkdir(parents=True)
    BASELINE_DIR.mkdir(parents=True)
    record = {}
    for scan_number, (name, grains) in enumerate(FRAMES.items(), start=1):
        detector, spots = reflections_for(grains)
        roi = ROI.get(name)
        image = render_frame(detector, spots, rng, roi)
        frame = FRAMES_DIR / f"{name}.h5"
        write_frame(frame, image, scan_number, detector, roi)
        summary = run_lauego_baseline(frame, BASELINE_DIR)
        summary.update(simulated_reflections=len(spots), sha256=sha256(frame), roi=list(roi) if roi else None,
                       size_bytes=frame.stat().st_size, grains=[list(g[0]) + [g[1], g[2]] for g in grains])
        record[name] = summary
        print(f"{name}: {len(spots)} simulated spots, {frame.stat().st_size/1e6:.2f} MB, "
              f"CLI peaks={summary['n_peaks']} patterns={summary['n_patterns']} indexed={summary['n_indexed']}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    provenance = {
        "generated_at_commit": git,
        "laueanalysis_version": laueanalysis.__version__ if hasattr(laueanalysis, "__version__") else "0.1.0",
        "liblaue_version": liblaue_version(),
        "numpy": np.__version__, "h5py": h5py.__version__,
        "seed": SEED, "geometry_file": str(GEOMETRY_FILE.relative_to(REPO)),
        "crystal_file": str(CRYSTAL_FILE.relative_to(REPO)), "detector_id": DETECTOR_ID,
        "energy_range_kev": list(ENERGY_RANGE_KEV), "peak_search": PEAK_SEARCH, "indexing": INDEXING,
        "frames": record,
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
