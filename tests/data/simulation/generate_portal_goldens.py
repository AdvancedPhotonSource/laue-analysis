#!/usr/bin/env python3
"""Regenerate the pre-port JZT simulation baselines.

This is a provenance tool, not a routine test helper.  Run it against the
reviewed sibling ``laue-portal`` checkout before changing the vendored backend::

    python tests/data/simulation/generate_portal_goldens.py ../laue-portal
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PORTAL_COMMIT = "477e4be"
JZT_ENTRY_COMMIT = "083cb6f"
ENERGY_RANGE_KEV = (6.0, 30.0)
DEPTH_UM = 0.0
DETECTOR = {
    "detector_id": "PHASE1-SYNTHETIC",
    "nx": 2048,
    "ny": 2048,
    "size_x_um": 409_600.0,
    "size_y_um": 409_600.0,
    "translation_um": [0.0, 0.0, 300_000.0],
    "rotation_vector_rad": [0.0, 0.0, 0.0],
}


def _git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def _metadata(case: str, crystal, reciprocal_rows: np.ndarray) -> dict:
    crystal_source = {
        "Ni": "tests/config/Ni.xml",
        "CdTe": "tests/config/CdTe.xml",
        "Si": "synthetic portal test fixture",
    }[case]
    return {
        "case": case,
        "baseline_kind": "raw pre-port portal/JZT output",
        "portal_commit": PORTAL_COMMIT,
        "jzt_entry_commit": JZT_ENTRY_COMMIT,
        "jzt_snapshot": "laue_portal/analysis/JZTLaueSim at portal commit",
        "crystal": {
            "name": crystal.name,
            "space_group": crystal.space_group,
            "source": crystal_source,
            "cell_nm_deg": [
                crystal.cell.a,
                crystal.cell.b,
                crystal.cell.c,
                crystal.cell.alpha,
                crystal.cell.beta,
                crystal.cell.gamma,
            ],
            "atoms": [
                {
                    "symbol": atom.symbol,
                    "label": atom.label,
                    "position": atom.position,
                    "occupancy": atom.occupancy,
                }
                for atom in crystal.atoms
            ],
        },
        "reciprocal_rows_1_per_nm": reciprocal_rows.tolist(),
        "detector": DETECTOR,
        "depth_um": DEPTH_UM,
        "energy_range_kev": ENERGY_RANGE_KEV,
        "energy_boundary_behavior": "JZT strict low < energy < high",
        "occupancy_behavior": "portal adapter did not pass occupancy",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "laueanalysis": importlib.metadata.version("laueanalysis"),
    }


def _baseline_rotation() -> np.ndarray:
    """Map [111] to a reproducible, on-detector diffraction direction."""
    source_u = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    source_v = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    source_w = np.cross(source_u, source_v)
    target_u = np.asarray([np.sqrt(1.0 - 0.18**2), 0.0, -0.18])
    target_v = np.asarray([0.0, 1.0, 0.0])
    target_w = np.cross(target_u, target_v)
    return np.vstack([source_u, source_v, source_w]).T @ np.vstack(
        [target_u, target_v, target_w]
    )


def _simulate(case: str, crystal, portal_modules: dict, output_dir: Path) -> None:
    cell = crystal.cell
    if not np.allclose([cell.a, cell.b, cell.c], cell.a):
        raise ValueError("Phase 1 material baselines currently require a cubic cell")
    reciprocal_rows = _baseline_rotation() * (2.0 * np.pi / cell.a)

    crystal_dict = {
        "structure_desc": crystal.name,
        "space_group": crystal.space_group,
        "lattice_params": [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        "atoms": [
            {
                "label": atom.label,
                "symbol": atom.symbol,
                "Zatom": atom.symbol,
                "xyz": atom.position,
                "occupancy": atom.occupancy,
            }
            for atom in crystal.atoms
        ],
    }
    detector = portal_modules["DetectorGeometry"](
        detector_id=DETECTOR["detector_id"],
        Nx=DETECTOR["nx"],
        Ny=DETECTOR["ny"],
        sizeX=DETECTOR["size_x_um"] / 1000.0,
        sizeY=DETECTOR["size_y_um"] / 1000.0,
        R=np.asarray(DETECTOR["rotation_vector_rad"]),
        P=np.asarray(DETECTOR["translation_um"]) / 1000.0,
    )
    lattice = portal_modules["build_lattice"](
        crystal_dict,
        portal_modules["LatticeBase"],
        portal_modules["Lattice"],
    )
    adapter = portal_modules["DetectorAdapter"](detector, depth=DEPTH_UM)
    pattern = portal_modules["LauePattern"].LauePattern(
        lattice,
        detector=adapter,
        recip=np.matrix(reciprocal_rows.T),
    )
    spots = pattern.calc(ELO=ENERGY_RANGE_KEV[0], EHI=ENERGY_RANGE_KEV[1], Nmax=100_000)

    hkl = np.asarray(
        [[int(spot.hkl.item(0, axis)) for axis in range(3)] for spot in spots],
        dtype=np.int64,
    ).reshape((-1, 3))
    detector_xy = np.asarray([spot.pixel for spot in spots], dtype=np.float64).reshape((-1, 2))
    energy_kev = np.asarray([spot.keV for spot in spots], dtype=np.float64)
    relative_intensity = np.asarray([spot.EwPo for spot in spots], dtype=np.float64)
    q = hkl @ reciprocal_rows
    metadata = _metadata(case, crystal, reciprocal_rows)
    metadata["accepted_candidate_count"] = len(spots)
    metadata["jzt_hkl_limits"] = [pattern.hmax, pattern.kmax, pattern.lmax]

    np.savez_compressed(
        output_dir / f"portal_jzt_{case.lower()}.npz",
        hkl=hkl,
        q=q,
        detector_xy=detector_xy,
        energy_kev=energy_kev,
        relative_intensity=relative_intensity,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"{case}: captured {len(spots)} spots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("portal", type=Path, help="Reviewed laue-portal checkout")
    args = parser.parse_args()
    portal = args.portal.resolve()
    revision = _git_revision(portal)
    if revision != PORTAL_COMMIT:
        raise SystemExit(f"Expected portal {PORTAL_COMMIT}, found {revision}")

    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(portal))
    sys.path.insert(0, str(repository / "src"))
    from laueanalysis.indexing import Atom, Cell, Crystal, load_crystal
    from laue_portal.analysis.geometry import DetectorGeometry
    from laue_portal.analysis.jzt_simulation import (
        _PortalDetectorAdapter,
        _build_lattice,
        _import_jzt_modules,
    )

    modules = _import_jzt_modules()
    portal_modules = {
        **modules,
        "DetectorGeometry": DetectorGeometry,
        "DetectorAdapter": _PortalDetectorAdapter,
        "build_lattice": _build_lattice,
    }
    cases = {
        "Ni": load_crystal(repository / "tests/config/Ni.xml"),
        "CdTe": load_crystal(repository / "tests/config/CdTe.xml"),
        "Si": Crystal(
            "Si",
            227,
            Cell(0.5431, 0.5431, 0.5431),
            (Atom("Si", (0.0, 0.0, 0.0), label="Si001"),),
        ),
    }
    output_dir = Path(__file__).resolve().parent
    for case, crystal in cases.items():
        _simulate(case, crystal, portal_modules, output_dir)


if __name__ == "__main__":
    main()
