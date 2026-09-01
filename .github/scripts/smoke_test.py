"""Smoke test for an installed lauelab wheel.

Run with ``python -I`` from a directory outside the repository so that only
the installed package is importable.  Checks that the native library loads,
the Python subpackages import, every executable runs, and that no installed
native file carries an absolute RPATH/RUNPATH entry.
"""

from __future__ import annotations

import subprocess
import sys
from importlib import resources


def main() -> int:
    from cffi import FFI

    import lauelab
    import lauelab.analysis  # noqa: F401
    import lauelab.visualization  # noqa: F401
    from lauelab.indexing._liblaue import get_library
    from lauelab.reconstruct import find_executable

    print("package:", lauelab.__file__)
    if "site-packages" not in lauelab.__file__:
        print("ERROR: lauelab was not imported from an installed location")
        return 1

    lib = get_library()
    print("liblaue version:", FFI().string(lib.laue_version()).decode())

    bin_dir = resources.files("lauelab.indexing.bin")
    executables = [str(bin_dir / n) for n in ("peaksearch", "pix2qs", "euler")] + [find_executable()]
    failures = 0
    for exe in executables:
        run = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        ok = bool(run.stdout.strip() or run.stderr.strip())
        print(f"{exe.rsplit('/', 1)[-1]}: exit {run.returncode}, usage text: {ok}")
        failures += not ok
    native = [str(bin_dir / "liblaue.so")] + executables
    for f in native:
        dyn = subprocess.run(["readelf", "-d", f], capture_output=True, text=True).stdout
        entries = [line.split("[", 1)[1].rstrip("]") for line in dyn.splitlines() if "RPATH" in line or "RUNPATH" in line]
        for entry in entries:
            for part in entry.split(":"):
                if part.startswith("/"):
                    print(f"ERROR: absolute RPATH entry in {f}: {part}")
                    failures += 1
        print(f"{f.rsplit('/', 1)[-1]}: rpath entries {entries}")
    print("smoke test:", "FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
