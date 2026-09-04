# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Process-pool helpers for wire-scan reconstruction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from lauelab.indexing.errors import LaueError

from .reconstruct import ReconstructionResult
from .reconstructor import Reconstructor, physical_core_count

_worker = None
_output_dir = None


def _init_worker(geometry, detector, output_dir, threads, options):
    global _worker, _output_dir
    _worker = Reconstructor(geometry, detector, num_threads=threads, **options)
    _output_dir = Path(output_dir)


def _reconstruct_one(path):
    path = Path(path)
    try:
        return _worker.reconstruct(path, _output_dir / f"{path.stem}_")
    except (LaueError, MemoryError, OSError) as error:
        return ReconstructionResult(
            False, [], "", error=str(error), command="liblaue", return_code=-1
        )


def reconstruct_points(paths, output_dir, *, geometry, detector, workers=None,
                       threads_per_worker=16, **reconstructor_kwargs):
    """Reconstruct point files in spawn-based worker processes.

    Each worker creates one reusable :class:`Reconstructor`. Spawn avoids
    forking a process after the OpenMP runtime has initialized.

    Parameters
    ----------
    paths
        Sequence of input 34-ID-E multi-image HDF5 point paths. Results preserve
        this order.
    output_dir : pathlib.Path or str
        Directory for reconstructed files. Each point uses ``<stem>_`` as its
        output filename prefix. The directory is created after options are
        validated.
    geometry : Geometry, pathlib.Path, or str
        Parsed geometry or path to a geometry XML file.
    detector : int
        Active physical detector slot in ``geometry``.
    workers : int or None
        Positive worker-process count. The default is ``None``, which uses the
        physical-core estimate divided by ``threads_per_worker``, with at least
        one worker.
    threads_per_worker : int
        Positive OpenMP thread count used by each worker. The default is ``16``.
    **reconstructor_kwargs
        Keyword arguments for :class:`Reconstructor`, including the required
        ``depth_range``. Do not pass ``num_threads``; use
        ``threads_per_worker`` instead.

    Returns
    -------
    list of ReconstructionResult
        One result per input path, in input order. An expected input, memory, or
        I/O failure sets ``success=False`` for that point without stopping the
        remaining points.

    Raises
    ------
    ValueError
        If a worker or thread count is invalid, or ``num_threads`` is passed.
    InputError
        If the shared reconstructor configuration is invalid.
    """
    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir)
    if "num_threads" in reconstructor_kwargs:
        raise ValueError("use threads_per_worker instead of num_threads")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be positive")
    if workers is None:
        workers = max(1, physical_core_count() // threads_per_worker)
    if workers < 1:
        raise ValueError("workers must be positive")
    Reconstructor(geometry, detector, num_threads=threads_per_worker,
                  **reconstructor_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(geometry.path if hasattr(geometry, "path") else geometry), detector,
                  str(output_dir), threads_per_worker, reconstructor_kwargs),
    ) as pool:
        return list(pool.map(_reconstruct_one, paths))
