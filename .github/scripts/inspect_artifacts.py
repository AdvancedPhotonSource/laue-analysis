"""Check sdist and wheel contents against the packaging policy.

Usage: python .github/scripts/inspect_artifacts.py dist/
Exits non-zero on any violation.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

# CLI programs must carry the exec bit. liblaue.so is dlopen()ed, which needs
# read only — CMake on Debian-family systems installs shared libraries 0644
# (CMAKE_INSTALL_SO_NO_EXE), so the wheel legitimately ships it either way.
NATIVE_EXECUTABLES = {"peaksearch", "pix2qs", "euler", "reconstructN_cpu"}
NATIVE_FILES = NATIVE_EXECUTABLES | {"liblaue.so"}
GENERATED = re.compile(r"\.(so|o|a|whl|pyc)$|/bin/(peaksearch|pix2qs|euler|reconstructN_[a-z]+)$|egg-info|/_build/|__pycache__")
NATIVE_SOURCE = re.compile(r"\.(c|h|cu)$|/indexing/src/|/reconstruct/source/|makefile", re.IGNORECASE)


def check_sdist(path: Path) -> list[str]:
    names = [m.name.split("/", 1)[1] for m in tarfile.open(path).getmembers() if m.isfile()]
    problems = []
    for required in ("CMakeLists.txt", "pyproject.toml", "LICENSE", "README.md"):
        if required not in names:
            problems.append(f"sdist is missing {required}")
    if not any(n.endswith(".c") for n in names):
        problems.append("sdist has no C sources")
    if not any(n.endswith(".cu") for n in names):
        problems.append("sdist has no CUDA source")
    for n in names:
        if GENERATED.search(n):
            problems.append(f"sdist contains generated artifact: {n}")
        if n.startswith(("build/", "dist/", "baselines/", "sandbox/")):
            problems.append(f"sdist contains excluded tree entry: {n}")
    print(f"sdist {path.name}: {len(names)} files")
    return problems


def check_wheel(path: Path) -> list[str]:
    z = zipfile.ZipFile(path)
    names = z.namelist()
    problems = []
    present = {n.rsplit("/", 1)[-1] for n in names if n.rsplit("/", 1)[-1] in NATIVE_FILES}
    for missing in NATIVE_FILES - present:
        problems.append(f"wheel is missing native file {missing}")
    for n in names:
        if NATIVE_SOURCE.search(n):
            problems.append(f"wheel contains native source/build file: {n}")
        base = n.rsplit("/", 1)[-1]
        if base in NATIVE_FILES:
            mode = (z.getinfo(n).external_attr >> 16) & 0o777
            if base in NATIVE_EXECUTABLES and not mode & 0o111:
                problems.append(f"wheel entry is not executable: {n} (mode {oct(mode)})")
            if not mode & 0o444:
                problems.append(f"wheel entry is not readable: {n} (mode {oct(mode)})")
    if not any("_vendor/jzt/elementData.xml" in n for n in names):
        problems.append("wheel is missing the JZT element data")
    if not path.name.endswith("py3-none-linux_x86_64.whl"):
        problems.append(f"unexpected wheel tag: {path.name}")
    print(f"wheel {path.name}: {len(names)} files, native: {sorted(present)}")
    return problems


def main(argv: list[str]) -> int:
    dist = Path(argv[1] if len(argv) > 1 else "dist")
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    problems = []
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist in {dist}, found {len(sdists)}")
    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel in {dist}, found {len(wheels)}")
    for sd in sdists:
        problems += check_sdist(sd)
    for wh in wheels:
        problems += check_wheel(wh)
    for p in problems:
        print("ERROR:", p)
    print("artifact check:", "FAILED" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
