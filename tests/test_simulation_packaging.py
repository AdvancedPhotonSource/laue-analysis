"""Installed-wheel smoke coverage for the private simulation resource."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import zipfile


def test_installed_wheel_contains_and_runs_private_jzt(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    shutil.copytree(
        repository,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist", "envs", "sandbox"
        ),
    )
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("laueanalysis-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    prefix = "laueanalysis/analysis/_vendor/jzt/"
    assert any(member.endswith(prefix + "elementData.xml") for member in members)
    assert any(member.endswith(prefix + "README.md") for member in members)
    assert not any(member.endswith(prefix + "LauePattern_allspots.py") for member in members)

    target = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    code = textwrap.dedent(
        f"""
        import json
        import sys
        import warnings
        sys.path.insert(0, {str(target)!r})
        original_path = list(sys.path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            import laueanalysis
            import laueanalysis.analysis
        normal_import_loaded_vendor = any(
            name.startswith('laueanalysis.analysis._vendor') for name in sys.modules
        )
        from laueanalysis.analysis import simulate_reflections
        from laueanalysis.indexing import Atom, Cell, Crystal, DetectorGeometry
        import numpy as np
        crystal = Crystal(
            'Ni', 225, Cell(0.35238, 0.35238, 0.35238),
            (Atom('Ni', (0, 0, 0), label='Ni001'),),
        )
        detector = DetectorGeometry(
            64, 64, 12_800, 12_800, 'WHEEL-SMOKE',
            np.asarray([0, 0, 30_000]), np.zeros(3), np.eye(3),
        )
        reciprocal = np.eye(3) * (2 * np.pi / 0.35238)
        with warnings.catch_warnings(record=True) as simulation_warnings:
            warnings.simplefilter('always')
            result = simulate_reflections(crystal, reciprocal, detector)
        print(json.dumps({{
            'normal_import_loaded_vendor': normal_import_loaded_vendor,
            'normal_import_warnings': [str(item.message) for item in caught],
            'simulation_warnings': [str(item.message) for item in simulation_warnings],
            'simulation_schema': [
                list(result.hkl.shape), list(result.q.shape),
                list(result.detector_xy.shape), list(result.energy_kev.shape),
                list(result.relative_intensity.shape),
            ],
            'vendor_loaded_after_simulation': any(
                name.startswith('laueanalysis.analysis._vendor.jzt') for name in sys.modules
            ),
            'visualization_loaded_after_simulation': any(
                name.startswith('laueanalysis.visualization') for name in sys.modules
            ),
            'sys_path_unchanged': sys.path == original_path,
        }}))
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(process.stdout) == {
        "normal_import_loaded_vendor": False,
        "normal_import_warnings": [],
        "simulation_warnings": [],
        "simulation_schema": [[0, 3], [0, 3], [0, 2], [0], [0]],
        "vendor_loaded_after_simulation": True,
        "visualization_loaded_after_simulation": False,
        "sys_path_unchanged": True,
    }
