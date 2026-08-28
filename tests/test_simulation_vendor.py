"""Containment and pre-normalization parity tests for the private JZT snapshot."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest


GOLDEN_DIR = Path(__file__).parent / "data" / "simulation"


class _BaselineDetector:
    """Minimal face-on detector matching the recorded portal fixture."""

    def __init__(self, metadata: dict):
        detector = metadata["detector"]
        self.name = detector["detector_id"]
        self.Nx = detector["nx"]
        self.Ny = detector["ny"]
        self.dx = detector["size_x_um"] / self.Nx / 1_000_000.0
        self.dy = detector["size_y_um"] / self.Ny / 1_000_000.0
        self.distance = detector["translation_um"][2] / 1_000_000.0
        self.XYZcenter = self.pixel2XYZ((self.Nx - 1) / 2.0, self.Ny - 1.0)

    def pixel2XYZ(self, px, py):
        return np.matrix(
            [
                (float(px) - (self.Nx - 1) / 2.0) * self.dx,
                (float(py) - (self.Ny - 1) / 2.0) * self.dy,
                self.distance,
            ]
        )

    def XYZ2pixel(self, xyz):
        outgoing = np.asarray(xyz, dtype=float).reshape(3)
        if outgoing[2] <= 0:
            return None
        scale = self.distance / outgoing[2]
        px = outgoing[0] * scale / self.dx + (self.Nx - 1) / 2.0
        py = outgoing[1] * scale / self.dy + (self.Ny - 1) / 2.0
        if not (0 <= px <= self.Nx - 1 and 0 <= py <= self.Ny - 1):
            return None
        return float(px), float(py)


def _simulate_raw(golden):
    from laueanalysis.analysis._vendor.jzt import Lattice, LatticeBase, LauePattern

    metadata = json.loads(golden["metadata_json"].item())
    crystal = metadata["crystal"]
    atoms = tuple(
        LatticeBase.atomXtal(
            label=atom["label"] or atom["symbol"],
            Zatom=atom["symbol"],
            xyz=atom["position"],
        )
        for atom in crystal["atoms"]
    )
    lattice = Lattice.Lattice3D(
        crystal["space_group"],
        crystal["cell_nm_deg"],
        desc=crystal["name"],
        atoms=atoms,
    )
    reciprocal = np.asarray(metadata["reciprocal_rows_1_per_nm"], dtype=float)
    pattern = LauePattern.LauePattern(
        lattice,
        detector=_BaselineDetector(metadata),
        recip=np.matrix(reciprocal.T),
    )
    spots = pattern.calc(
        ELO=metadata["energy_range_kev"][0],
        EHI=metadata["energy_range_kev"][1],
        Nmax=100_000,
    )
    hkl = np.asarray(
        [[int(spot.hkl.item(0, axis)) for axis in range(3)] for spot in spots],
        dtype=np.int64,
    ).reshape((-1, 3))
    return {
        "hkl": hkl,
        "q": hkl @ reciprocal,
        "detector_xy": np.asarray([spot.pixel for spot in spots]).reshape((-1, 2)),
        "energy_kev": np.asarray([spot.keV for spot in spots]),
        "relative_intensity": np.asarray([spot.EwPo for spot in spots]),
    }


@pytest.mark.parametrize("case", ["ni", "cdte", "si"])
def test_private_jzt_matches_pre_portal_baseline(case, tmp_path):
    actual_path = tmp_path / f"actual-{case}.npz"
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, str(actual_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.stderr == ""
    with np.load(GOLDEN_DIR / f"portal_jzt_{case}.npz") as golden:
        with np.load(actual_path) as actual:
            np.testing.assert_array_equal(actual["hkl"], golden["hkl"])
            for field in ("q", "detector_xy", "energy_kev", "relative_intensity"):
                np.testing.assert_allclose(actual[field], golden[field], rtol=1e-12, atol=1e-12)


def test_normal_imports_do_not_load_vendor_or_emit_warnings():
    code = textwrap.dedent(
        """
        import json
        import sys
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            import laueanalysis
            import laueanalysis.analysis
        print(json.dumps({
            'vendor_loaded': any(
                name.startswith('laueanalysis.analysis._vendor') for name in sys.modules
            ),
            'visualization_loaded': any(
                name.startswith('laueanalysis.visualization') for name in sys.modules
            ),
            'plotly_loaded': any(name.startswith('plotly') for name in sys.modules),
            'warnings': [str(item.message) for item in caught],
        }))
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert result == {
        "vendor_loaded": False,
        "visualization_loaded": False,
        "plotly_loaded": False,
        "warnings": [],
    }


def test_atomic_data_is_a_package_resource():
    from importlib import resources

    resource = resources.files("laueanalysis.analysis._vendor.jzt").joinpath("elementData.xml")
    assert resource.is_file()
    assert resource.stat().st_size > 100_000


def _write_subprocess_result(case: str, output: Path) -> None:
    with np.load(GOLDEN_DIR / f"portal_jzt_{case}.npz") as golden:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            actual = _simulate_raw(golden)
    np.savez(output, **actual)


if __name__ == "__main__":
    _write_subprocess_result(sys.argv[1], Path(sys.argv[2]))
