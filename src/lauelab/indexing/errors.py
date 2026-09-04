# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Public exceptions for in-process indexing."""


class LaueError(Exception):
    """Base class for errors reported by the in-process indexing API.

    Notes
    -----
    Native allocation failures are reported as the built-in :class:`MemoryError`
    rather than as a ``LaueError`` subclass.
    """


class InputError(LaueError, ValueError):
    """Invalid geometry selection, parameters, frame data, or metadata.

    This exception is both a :class:`LaueError` and a :class:`ValueError`, so
    callers may catch it either as a package-specific error or as invalid input.
    """


class IndexingError(LaueError, RuntimeError):
    """Numerical or internal failure in a native indexing stage.

    This exception is both a :class:`LaueError` and a :class:`RuntimeError`.
    Its message identifies the failed stage and includes the native diagnostic.
    """


class ReconstructionError(LaueError, RuntimeError):
    """Runtime failure in the in-process reconstruction pipeline."""
