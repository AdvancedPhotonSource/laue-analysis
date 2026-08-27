"""Surface frames and stereographic pole projection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np


@dataclass(frozen=True)
class SurfaceFrame:
    """Right-handed orthonormal sample-surface coordinate frame."""

    tilt: np.ndarray
    roll: np.ndarray
    normal: np.ndarray
    name: str | None = None

    def __post_init__(self):
        vectors = []
        for field_name in ("tilt", "roll", "normal"):
            vector = np.array(getattr(self, field_name), dtype=float, copy=True)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{field_name} must be a finite 3-vector")
            norm = np.linalg.norm(vector)
            if norm < 1e-12:
                raise ValueError(f"{field_name} must be nonzero")
            vectors.append(vector / norm)
        tilt, roll, normal = vectors
        if not np.allclose(np.asarray(vectors) @ np.asarray(vectors).T, np.eye(3), atol=1e-3):
            raise ValueError("surface vectors must be orthogonal")
        if np.dot(np.cross(tilt, roll), normal) < 0.999:
            raise ValueError("surface frame must be right-handed: tilt x roll = normal")
        for field_name, vector in zip(("tilt", "roll", "normal"), vectors, strict=True):
            vector.setflags(write=False)
            object.__setattr__(self, field_name, vector)

    @classmethod
    def from_vectors(cls, *, tilt, roll, normal, name=None):
        """Construct and normalize a frame from three vectors."""
        return cls(tilt=tilt, roll=roll, normal=normal, name=name)

    @classmethod
    def aps_34ide(cls, name="normal"):
        """Return a named APS 34-ID-E surface frame."""
        ir2 = 1.0 / np.sqrt(2.0)
        frames = {
            "normal": ([1, 0, 0], [0, -ir2, -ir2], [0, ir2, -ir2]),
            "F": ([1, 0, 0], [0, -ir2, -ir2], [0, ir2, -ir2]),
            "X": ([0, -ir2, -ir2], [0, ir2, -ir2], [1, 0, 0]),
            "H": ([1, 0, 0], [0, ir2, -ir2], [0, ir2, ir2]),
            "Y": ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
            "Z": ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        }
        if name not in frames:
            raise ValueError(f"unknown APS 34-ID-E surface {name!r}; choose from {tuple(frames)}")
        tilt, roll, normal = frames[name]
        return cls(tilt, roll, normal, f"aps_34ide:{name}")


def cubic_hkl_family(hkl):
    """Return normalized symmetry-equivalent directions for a cubic HKL."""
    values = tuple(abs(int(value)) for value in hkl)
    if len(values) != 3 or values == (0, 0, 0):
        raise ValueError("hkl must contain three integers and cannot be (0, 0, 0)")
    candidates = {
        tuple(sign * value for sign, value in zip(signs, perm, strict=True))
        for perm in permutations(values)
        for signs in product((1, -1), repeat=3)
    }
    return np.asarray(sorted(candidates), dtype=float) / np.linalg.norm(values)


def pole_figure_points(recip_lattices, hkl_family, *, surface=None):
    """Project reciprocal-lattice pole directions onto a surface frame."""
    lattices = np.asarray(recip_lattices, dtype=float)
    family = np.asarray(hkl_family, dtype=float)
    if lattices.ndim != 3 or lattices.shape[1:] != (3, 3):
        raise ValueError("recip_lattices must have shape (n, 3, 3)")
    if family.ndim != 2 or family.shape[1:] != (3,):
        raise ValueError("hkl_family must have shape (m, 3)")
    surface = SurfaceFrame.aps_34ide() if surface is None else surface
    if len(lattices) == 0 or len(family) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    vectors = np.matmul(
        lattices.transpose(0, 2, 1)[:, None, :, :],
        family[None, :, :, None],
    )[..., 0]
    norms = np.linalg.norm(vectors, axis=2)
    nonzero = ~(norms < 1e-12)
    vectors = np.divide(
        vectors,
        norms[..., None],
        out=np.full_like(vectors, np.nan),
        where=nonzero[..., None],
    )
    dot_normal = vectors @ surface.normal
    keep = nonzero & ~(dot_normal < 0)
    radius = np.divide(
        np.sqrt(1.0 - np.clip(dot_normal, 0, 1) ** 2),
        1.0 + dot_normal,
        out=np.zeros_like(dot_normal),
        where=(1.0 + dot_normal) > 1e-12,
    )
    angle = np.arctan2(vectors @ surface.roll, vectors @ surface.tilt)
    points = np.column_stack([(radius * np.cos(angle))[keep], (radius * np.sin(angle))[keep]])
    indices = np.broadcast_to(np.arange(len(lattices))[:, None], keep.shape)[keep]
    return points, indices


def pole_color_radius(center, angle_deg):
    """Convert an angular color radius to a stereographic radius."""
    x, y = np.asarray(center, dtype=float)
    radius = np.hypot(x, y)
    if radius < 1e-12:
        return float(np.tan(np.radians(angle_deg) / 2.0))
    phi = 2.0 * np.arctan(1.0 / radius)
    delta = np.radians(angle_deg)
    phi = phi - delta if delta < phi else phi + delta
    return float(abs(radius - np.sin(phi) / (1.0 - np.cos(phi))))
