#!/usr/bin/env python3
"""Regenerate reviewed Phase 2 normalized simulation fixtures.

Run from repository root after deliberately reviewing scientific changes::

    python tests/data/simulation/generate_normalized_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _load_case(case: str, repository: Path, fixture_dir: Path):
    from lauelab.indexing import Atom, Cell, Crystal, DetectorGeometry, load_crystal

    with np.load(fixture_dir / f"portal_jzt_{case}.npz") as raw:
        metadata = json.loads(raw["metadata_json"].item())
    detector_data = metadata["detector"]
    detector = DetectorGeometry(
        nx=detector_data["nx"],
        ny=detector_data["ny"],
        size_x=detector_data["size_x_um"],
        size_y=detector_data["size_y_um"],
        detector_id=detector_data["detector_id"],
        translation=np.asarray(detector_data["translation_um"]),
        rotation_vector=np.asarray(detector_data["rotation_vector_rad"]),
        rotation=np.eye(3),
    )
    if case == "si":
        crystal = Crystal(
            "Si",
            227,
            Cell(0.5431, 0.5431, 0.5431),
            (Atom("Si", (0.0, 0.0, 0.0), label="Si001"),),
        )
    else:
        crystal = load_crystal(repository / metadata["crystal"]["source"])
    reciprocal = np.asarray(metadata["reciprocal_rows_1_per_nm"])
    return crystal, reciprocal, detector, metadata


def main() -> None:
    fixture_dir = Path(__file__).resolve().parent
    repository = fixture_dir.parents[2]
    sys.path.insert(0, str(repository / "src"))
    from lauelab.analysis import simulate_reflections

    for case in ("ni", "cdte", "si"):
        crystal, reciprocal, detector, source_metadata = _load_case(
            case, repository, fixture_dir
        )
        result = simulate_reflections(crystal, reciprocal, detector)
        metadata = {
            "case": case,
            "contract": "Phase 2 normalized SimulationResult",
            "source_fixture": f"portal_jzt_{case}.npz",
            "source_portal_commit": source_metadata["portal_commit"],
            "normalization": [
                "crystal occupancy propagated",
                "inclusive energy bounds enforced",
                "full-detector positions recomputed with DetectorGeometry",
                "strongest positive-harmonic representative retained",
                "rows ordered by descending intensity, energy, and HKL tie-breakers",
            ],
            "comparison_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "spot_count": len(result.hkl),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        }
        np.savez_compressed(
            fixture_dir / f"normalized_{case}.npz",
            hkl=result.hkl,
            q=result.q,
            detector_xy=result.detector_xy,
            energy_kev=result.energy_kev,
            relative_intensity=result.relative_intensity,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        print(f"{case}: captured {len(result.hkl)} normalized spots")


if __name__ == "__main__":
    main()
