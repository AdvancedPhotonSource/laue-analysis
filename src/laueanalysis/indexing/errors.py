"""Public exceptions for in-process indexing."""


class LaueError(Exception):
    """Base error raised by the in-process indexing API."""


class InputError(LaueError, ValueError):
    """Invalid geometry, crystal, parameter, or frame input."""


class IndexingError(LaueError, RuntimeError):
    """A native indexing stage failed."""
