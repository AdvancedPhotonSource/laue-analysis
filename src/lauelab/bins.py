from importlib import resources
import os
from typing import Any, Dict

from lauelab.reconstruct.reconstruct import find_executable, find_gpu_executable


def get_packaged_executable_path(program_name: str) -> str:
    """Return the path to a packaged indexing executable."""
    try:
        executable = resources.files("lauelab.indexing.bin") / program_name
        if executable.is_file():
            return str(executable)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    raise FileNotFoundError(
        f"Could not locate executable {program_name!r} in "
        "lauelab.indexing.bin package"
    )


def _is_executable(path: str) -> bool:
    """Return True if path exists and is executable."""
    return os.path.exists(path) and os.access(path, os.X_OK)


def check_avail_bins() -> Dict[str, Any]:
    """
    Lightweight availability check for LaueAnalysis binaries.

    Returns:
        {
          "avail": {
            "indexing": {"euler": bool, "peaksearch": bool, "pix2qs": bool},
            "reconstruct": {"cpu": bool, "gpu": bool}
          },
          "paths": {
            "indexing": {"euler": str | None, "peaksearch": str | None, "pix2qs": str | None},
            "reconstruct": {"cpu": str | None, "gpu": str | None}
          }
        }
    """
    result: Dict[str, Any] = {
        "avail": {
            "indexing": {"euler": False, "peaksearch": False, "pix2qs": False},
            "reconstruct": {"cpu": False, "gpu": False},
        },
        "paths": {
            "indexing": {"euler": None, "peaksearch": None, "pix2qs": None},
            "reconstruct": {"cpu": None, "gpu": None},
        },
    }

    # Indexing binaries
    for prog in ["euler", "peaksearch", "pix2qs"]:
        try:
            path = get_packaged_executable_path(prog)
            result["paths"]["indexing"][prog] = path
            result["avail"]["indexing"][prog] = _is_executable(path)
        except FileNotFoundError:
            result["paths"]["indexing"][prog] = None
            result["avail"]["indexing"][prog] = False

    # Reconstruction CPU
    try:
        cpu_path = find_executable()
        result["paths"]["reconstruct"]["cpu"] = cpu_path
        result["avail"]["reconstruct"]["cpu"] = _is_executable(cpu_path)
    except FileNotFoundError:
        result["paths"]["reconstruct"]["cpu"] = None
        result["avail"]["reconstruct"]["cpu"] = False

    # Reconstruction GPU (always checked)
    try:
        gpu_path = find_gpu_executable()
        result["paths"]["reconstruct"]["gpu"] = gpu_path
        result["avail"]["reconstruct"]["gpu"] = _is_executable(gpu_path)
    except FileNotFoundError:
        result["paths"]["reconstruct"]["gpu"] = None
        result["avail"]["reconstruct"]["gpu"] = False

    return result
