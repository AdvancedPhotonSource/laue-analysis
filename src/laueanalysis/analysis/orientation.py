"""Backend-independent crystal orientation calculations."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def lattice_params_to_reciprocal(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Return a reciprocal lattice whose rows are ``a*``, ``b*``, and ``c*``."""
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_g = np.sin(gamma)
    volume = np.sqrt(
        1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    )
    direct = np.array([
        [a, 0.0, 0.0],
        [b * cos_g, b * sin_g, 0.0],
        [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, c * volume / sin_g],
    ])
    return 2.0 * np.pi * np.linalg.inv(direct).T


def recip_to_orientation(recip_lattice, reference_recip):
    """Return the orientation matrix for measured and reference lattices."""
    measured = np.asarray(recip_lattice, dtype=float)
    reference = np.asarray(reference_recip, dtype=float)
    if measured.shape != (3, 3) or reference.shape != (3, 3):
        raise ValueError("reciprocal lattices must have shape (3, 3)")
    return measured.T @ np.linalg.inv(reference.T)


def orientation_to_rodrigues(rotation):
    """Convert a 3 by 3 rotation matrix to a Rodrigues vector."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * np.tan(angle / 2.0)


def crystal_direction(rotation, lab_direction):
    """Express a lab-frame direction in crystal coordinates."""
    rotation = np.asarray(rotation, dtype=float)
    direction = np.asarray(lab_direction, dtype=float)
    if rotation.shape != (3, 3) or direction.shape != (3,):
        raise ValueError("rotation and lab_direction must have shapes (3, 3) and (3,)")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("lab_direction must be a finite nonzero vector")
    result = np.linalg.solve(rotation, direction / norm)
    return result / np.linalg.norm(result)


def _rotation_matrix(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = np.radians(angle_deg)
    c, s, c1 = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
        [x * y * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
        [x * z * c1 - y * s, y * z * c1 + x * s, c + z * z * c1],
    ])


def _cubic_symmetry_ops():
    operations = [np.eye(3)]
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        operations.extend(_rotation_matrix(axis, angle) for angle in (90, 180, 270))
    for axis in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]):
        operations.extend(_rotation_matrix(axis, angle) for angle in (120, 240))
    for axis in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]):
        operations.append(_rotation_matrix(axis, 180))
    return np.asarray(operations)


def _hexagonal_symmetry_ops():
    operations = [_rotation_matrix([0, 0, 1], angle) for angle in range(0, 360, 60)]
    for angle in range(0, 180, 30):
        operations.append(_rotation_matrix([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0], 180))
    return np.asarray(operations)


CUBIC_SYMMETRY = _cubic_symmetry_ops()
HEXAGONAL_SYMMETRY = _hexagonal_symmetry_ops()


def symmetry_operations(symmetry: str | int) -> np.ndarray:
    """Return proper rotations for a supported symmetry name or space group."""
    if isinstance(symmetry, (int, np.integer)):
        if 195 <= symmetry <= 230:
            return CUBIC_SYMMETRY.copy()
        if 168 <= symmetry <= 194:
            return HEXAGONAL_SYMMETRY.copy()
    elif symmetry == "cubic":
        return CUBIC_SYMMETRY.copy()
    elif symmetry == "hexagonal":
        return HEXAGONAL_SYMMETRY.copy()
    raise ValueError("symmetry must be 'cubic', 'hexagonal', or a supported space group")


def symmetry_reduce_orientation(rotation, *, operations=None):
    """Return the symmetry-equivalent orientation nearest to identity."""
    rotation = np.asarray(rotation, dtype=float)
    operations = np.eye(3)[None, ...] if operations is None else np.asarray(operations, dtype=float)
    candidates = rotation @ np.swapaxes(operations, 1, 2)
    return candidates[np.argmax(np.trace(candidates, axis1=1, axis2=2))]


def misorientation_matrix(rotation_a, rotation_b, *, operations=None):
    """Return the minimum-angle misorientation from ``rotation_b`` to ``rotation_a``."""
    a = np.asarray(rotation_a, dtype=float)
    b_inv = np.linalg.inv(np.asarray(rotation_b, dtype=float))
    operations = np.eye(3)[None, ...] if operations is None else np.asarray(operations, dtype=float)
    candidates = a @ (np.swapaxes(operations, 1, 2) @ b_inv)
    return candidates[np.argmax(np.trace(candidates, axis1=1, axis2=2))]


def misorientation_angle(rotation_a, rotation_b, *, operations=None):
    """Return the minimum misorientation angle in degrees."""
    relative = misorientation_matrix(rotation_a, rotation_b, operations=operations)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def misorientation_from_reference(rotations, reference_index, *, operations=None):
    """Return Rodrigues vectors and angles relative to one orientation."""
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (n, 3, 3)")
    if not 0 <= reference_index < len(rotations):
        raise IndexError("reference_index is out of range")
    reference = rotations[reference_index]
    reduced = np.asarray([
        misorientation_matrix(rotation, reference, operations=operations)
        for rotation in rotations
    ])
    vectors = np.asarray([orientation_to_rodrigues(rotation) for rotation in reduced])
    angles = 2.0 * np.degrees(np.arctan(np.linalg.norm(vectors, axis=1)))
    return vectors, angles


def pairwise_misorientation(rotations, *, indices=None, operations=None):
    """Return pairs and corresponding misorientation angles."""
    rotations = np.asarray(rotations, dtype=float)
    selected = range(len(rotations)) if indices is None else tuple(indices)
    pairs = tuple(combinations(selected, 2))
    angles = np.asarray([
        misorientation_angle(rotations[i], rotations[j], operations=operations)
        for i, j in pairs
    ])
    return pairs, angles
