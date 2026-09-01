"""
Laue indexing submodule.

This submodule contains the core indexing functionality including:
- Functional indexing interface (index.py)
- Data classes for configuration and data structures
- XML writer utilities
- Compiled C binaries for peak search, q-space conversion, and indexing
"""

from ._liblaue import DetectorGeometry, Geometry, load_geometry
from .crystal import Atom, Cell, Crystal, load_crystal
from .errors import IndexingError, InputError, LaueError
from .index import IndexingResult, index, lauego
from .indexer import (
    FrameMetadata, FrameResult, Indexer, IndexParams, Pattern, PeakParams, index_frame,
)
from .xmlWriter import XMLWriter

__all__ = [
    'index',
    'lauego',
    'IndexingResult',
    'index_frame',
    'Indexer',
    'Cell',
    'Atom',
    'Crystal',
    'load_crystal',
    'DetectorGeometry',
    'Geometry',
    'load_geometry',
    'LaueError',
    'InputError',
    'IndexingError',
    'PeakParams',
    'IndexParams',
    'Pattern',
    'FrameMetadata',
    'FrameResult',
    'XMLWriter'
]
