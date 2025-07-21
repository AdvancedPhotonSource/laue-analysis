"""
Laue indexing submodule.

This submodule contains the core indexing functionality including:
- Functional indexing interface (index.py)
- MPI runner for distributed processing
- Data classes for configuration and data structures
- XML writer utilities
- Compiled C binaries for peak search, q-space conversion, and indexing
"""

from .index import index, IndexingResult
from .xmlWriter import XMLWriter

__all__ = [
    'index',
    'IndexingResult', 
    'XMLWriter'
]
