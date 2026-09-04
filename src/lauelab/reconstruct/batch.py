# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Process-pool helpers for wire-scan reconstruction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from .reconstructor import Reconstructor, physical_core_count

_worker = None
_output_dir = None


def _init_worker(geometry, detector, output_dir, threads, options):
    global _worker, _output_dir
    _worker = Reconstructor(geometry, detector, num_threads=threads, **options)
    _output_dir = Path(output_dir)


def _reconstruct_one(path):
    path = Path(path)
    return _worker.reconstruct(path, _output_dir / f"{path.stem}_")


def reconstruct_points(paths, output_dir, *, geometry, detector, workers=None,
                       threads_per_worker=16, **reconstructor_kwargs):
    """Reconstruct point files in spawn-based worker processes.

    Each worker creates one reusable :class:`Reconstructor`. Spawn avoids
    forking a process after the OpenMP runtime has initialized.
    """
    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be positive")
    if workers is None:
        workers = max(1, physical_core_count() // threads_per_worker)
    if workers < 1:
        raise ValueError("workers must be positive")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(geometry.path if hasattr(geometry, "path") else geometry), detector,
                  str(output_dir), threads_per_worker, reconstructor_kwargs),
    ) as pool:
        return list(pool.map(_reconstruct_one, paths))
