"""Typed scientific API for complete on-detector Laue simulations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from numbers import Real
from typing import Iterator
import warnings

import numpy as np

from laueanalysis.indexing import Crystal, DetectorGeometry


_CANDIDATE_LIMIT = 100_000
_ORDER_SIGNIFICANT_DIGITS = 12
_HC_KEV_NM = 1.2398419739


def _integer_array(value, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise TypeError(f"{name} must be an integer array") from error
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must have an integer dtype")
    if array.dtype.kind == "u" and array.size and np.max(array) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains values outside the supported integer range")
    return np.array(array, dtype=np.int64, copy=True)


def _float_array(value, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise TypeError(f"{name} must be a real numeric array") from error
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype")
    return np.array(array, dtype=np.float64, copy=True)


def _direction_key(hkl: np.ndarray) -> tuple[int, int, int]:
    values = tuple(int(value) for value in hkl)
    divisor = int(np.gcd.reduce(np.abs(values)))
    if divisor == 0:
        raise ValueError("Miller indices must not contain the zero reflection")
    return tuple(value // divisor for value in values)


def _stable_order_value(value: float) -> float:
    """Discard nonphysical last-bit noise from a floating-point sort key."""
    return float(f"{value:.{_ORDER_SIGNIFICANT_DIGITS}g}")


@dataclass(frozen=True)
class SimulationResult:
    """Immutable, aligned output from one detector reflection simulation.

    Parameters
    ----------
    hkl
        Integer Miller indices with shape ``(n, 3)``.
    q
        Reciprocal vectors in ``1/nm`` with shape ``(n, 3)``.
    detector_xy
        Zero-based, unbinned full-detector ``(x, y)`` pixels with shape
        ``(n, 2)``.
    energy_kev
        Photon energies in keV with shape ``(n,)``.
    relative_intensity
        Uncalibrated JZT-derived relative intensities with shape ``(n,)``.

    Notes
    -----
    All arrays are copied, normalized to 64-bit dtypes, and made read-only.
    Rows retain the deterministic order produced by :func:`simulate_reflections`.
    """

    hkl: np.ndarray
    q: np.ndarray
    detector_xy: np.ndarray
    energy_kev: np.ndarray
    relative_intensity: np.ndarray

    def __post_init__(self) -> None:
        hkl = _integer_array(self.hkl, name="hkl")
        if hkl.ndim != 2 or hkl.shape[1:] != (3,):
            raise ValueError(f"hkl must have shape (n, 3); received {hkl.shape}")
        count = len(hkl)
        specifications = {
            "q": (count, 3),
            "detector_xy": (count, 2),
            "energy_kev": (count,),
            "relative_intensity": (count,),
        }
        arrays = {"hkl": hkl}
        for name, shape in specifications.items():
            array = _float_array(getattr(self, name), name=name)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; received {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            arrays[name] = array
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def missing_from(self, indexed_hkl: np.ndarray) -> "SimulationResult":
        """Return simulated directions absent from indexed Miller indices.

        Positive scalar harmonics share a direction. Opposite signed Miller
        indices remain distinct Friedel directions.

        Parameters
        ----------
        indexed_hkl
            Integer Miller indices with shape ``(m, 3)``. A one-dimensional
            empty integer array is also accepted.

        Returns
        -------
        SimulationResult
            A new immutable result. Its rows retain their original order.

        Raises
        ------
        TypeError
            If ``indexed_hkl`` does not have an integer dtype.
        ValueError
            If ``indexed_hkl`` has the wrong shape or contains ``(0, 0, 0)``.
        """
        try:
            indexed = np.asarray(indexed_hkl)
        except Exception as error:
            raise TypeError("indexed_hkl must be an integer array") from error
        if indexed.shape == (0,):
            indexed = np.empty((0, 3), dtype=np.int64)
        else:
            indexed = _integer_array(indexed, name="indexed_hkl")
        if indexed.ndim != 2 or indexed.shape[1:] != (3,):
            raise ValueError(
                f"indexed_hkl must have shape (m, 3); received {indexed.shape}"
            )
        indexed_directions = {_direction_key(row) for row in indexed}
        selected = np.asarray(
            [_direction_key(row) not in indexed_directions for row in self.hkl],
            dtype=bool,
        )
        return SimulationResult(
            hkl=self.hkl[selected],
            q=self.q[selected],
            detector_xy=self.detector_xy[selected],
            energy_kev=self.energy_kev[selected],
            relative_intensity=self.relative_intensity[selected],
        )


class _JZTDetectorAdapter:
    """Private adapter from maintained detector geometry to JZT's protocol."""

    def __init__(self, detector: DetectorGeometry, depth: float):
        self._detector = detector
        self._depth = depth
        self.name = detector.detector_id
        self.Nx = detector.nx
        self.Ny = detector.ny
        self.dx = detector.size_x / detector.nx / 1_000_000.0
        self.dy = detector.size_y / detector.ny / 1_000_000.0
        self.XYZcenter = self.pixel2XYZ((detector.nx - 1) / 2.0, detector.ny - 1.0)

    def pixel2XYZ(self, px, py):
        pixels = np.asarray([float(px), float(py)], dtype=np.float64)
        return self._detector.pixel_to_lab(pixels).reshape(1, 3) / 1_000_000.0

    def XYZ2pixel(self, xyz):
        try:
            outgoing = np.asarray(xyz, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        q = outgoing - np.asarray([0.0, 0.0, 1.0])
        xy = self._detector.q_to_pixel(q, depth=self._depth, on_detector=True)
        if not np.isfinite(xy).all():
            return None
        return float(xy[0]), float(xy[1])


@dataclass(frozen=True)
class _BackendOutput:
    candidates: tuple[object, ...]
    limit_reached: bool


@contextmanager
def _vendor_warning_boundary() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"the matrix subclass is not the recommended way to represent matrices.*",
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Conversion of an array with ndim > 0 to a scalar is deprecated.*",
            category=DeprecationWarning,
            module=r"laueanalysis\.analysis\._vendor\.jzt(?:\.|$)",
        )
        yield


def _load_jzt_modules():
    return (
        import_module("laueanalysis.analysis._vendor.jzt.LatticeBase"),
        import_module("laueanalysis.analysis._vendor.jzt.Lattice"),
        import_module("laueanalysis.analysis._vendor.jzt.LauePattern"),
    )


def _cell_parameters_nm(crystal: Crystal) -> list[float]:
    cell = crystal.cell.in_angstrom
    return [
        cell.a / 10.0,
        cell.b / 10.0,
        cell.c / 10.0,
        cell.alpha,
        cell.beta,
        cell.gamma,
    ]


def _atomic_number(lattice_base, symbol: str) -> int:
    try:
        atomic_number = int(lattice_base.atomGeneral.baseAtom(symbol).Z)
    except Exception as error:
        raise ValueError(f"unknown chemical element symbol {symbol!r}") from error
    if not 1 <= atomic_number <= lattice_base.Zmax:
        raise ValueError(f"unknown chemical element symbol {symbol!r}")
    return atomic_number


def _maximum_bragg_angle(detector: DetectorGeometry, depth: float) -> float:
    source = np.asarray([0.0, 0.0, depth])
    corners = np.asarray([
        [0.0, 0.0],
        [detector.nx - 1.0, 0.0],
        [0.0, detector.ny - 1.0],
        [detector.nx - 1.0, detector.ny - 1.0],
    ])
    points = [*detector.pixel_to_lab(corners)]

    for first, second in ((0, 1), (2, 3), (0, 2), (1, 3)):
        origin = points[first] - source
        direction = points[second] - points[first]
        denominator = direction[2] * np.dot(origin, direction) - origin[2] * np.dot(
            direction, direction
        )
        if denominator:
            position = (
                origin[2] * np.dot(origin, direction)
                - direction[2] * np.dot(origin, origin)
            ) / denominator
            if 0.0 < position < 1.0:
                points.append(source + origin + position * direction)

    rays = np.asarray(points) - source
    cosines = rays[:, 2] / np.linalg.norm(rays, axis=1)
    maximum_two_theta = np.max(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return maximum_two_theta / 2.0


def _hkl_bound(
    reciprocal: np.ndarray,
    detector: DetectorGeometry,
    depth: float,
    energy_high_kev: float,
) -> tuple[int, int, int]:
    q_max = (
        4.0
        * np.pi
        * np.sin(_maximum_bragg_angle(detector, depth))
        * energy_high_kev
        / _HC_KEV_NM
    )
    bounds = np.ceil(
        q_max * np.linalg.norm(np.linalg.inv(reciprocal), axis=0)
    ).astype(np.int64)
    return tuple(int(value) for value in bounds)


def _execute_jzt(
    modules,
    crystal: Crystal,
    reciprocal: np.ndarray,
    detector: DetectorGeometry,
    depth: float,
    energy_range_kev: tuple[float, float],
) -> _BackendOutput:
    lattice_base, lattice_module, pattern_module = modules
    used_labels: set[str] = set()
    atoms = []
    for index, atom in enumerate(crystal.atoms):
        base_label = atom.label or f"{atom.symbol}{index + 1}"
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)
        atoms.append(
            lattice_base.atomXtal(
                label=label,
                Zatom=_atomic_number(lattice_base, atom.symbol),
                xyz=atom.position,
                occ=atom.occupancy,
            )
        )
    lattice = lattice_module.Lattice3D(
        crystal.space_group,
        _cell_parameters_nm(crystal),
        desc=crystal.name,
        atoms=tuple(atoms),
    )
    adapter = _JZTDetectorAdapter(detector, depth)
    pattern = pattern_module.LauePattern(
        lattice,
        detector=adapter,
        recip=reciprocal.T,
    )
    low, high = energy_range_kev
    pattern.calc(
        ELO=np.nextafter(low, -np.inf),
        EHI=np.nextafter(high, np.inf),
        hklMax=_hkl_bound(reciprocal, detector, depth, high),
        Nmax=_CANDIDATE_LIMIT,
    )
    return _BackendOutput(
        candidates=tuple(pattern._all_spots),
        limit_reached=bool(pattern._candidate_limit_reached),
    )


def _empty_result() -> SimulationResult:
    return SimulationResult(
        hkl=np.empty((0, 3), dtype=np.int64),
        q=np.empty((0, 3), dtype=np.float64),
        detector_xy=np.empty((0, 2), dtype=np.float64),
        energy_kev=np.empty(0, dtype=np.float64),
        relative_intensity=np.empty(0, dtype=np.float64),
    )


def _normalize_candidates(
    candidates: tuple[object, ...],
    reciprocal: np.ndarray,
    detector: DetectorGeometry,
    depth: float,
    energy_range_kev: tuple[float, float],
) -> SimulationResult:
    low, high = energy_range_kev
    rows: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for spot in candidates:
        try:
            raw_hkl = np.asarray(spot.hkl).reshape(-1)
            if raw_hkl.shape != (3,) or raw_hkl.dtype.kind not in "iuf":
                raise ValueError("invalid Miller indices")
            rounded = np.rint(raw_hkl)
            if not np.isfinite(raw_hkl).all() or not np.array_equal(raw_hkl, rounded):
                raise ValueError("non-integral Miller indices")
            hkl = rounded.astype(np.int64)
            _direction_key(hkl)
            energy = float(spot.keV)
            intensity = float(spot.EwPo)
        except Exception as error:
            raise RuntimeError("Reflection simulation backend returned an invalid candidate") from error
        if (
            not np.isfinite(energy)
            or not np.isfinite(intensity)
            or energy <= 0
            or intensity <= 0
        ):
            raise RuntimeError("Reflection simulation backend returned invalid numerical output")
        if not low <= energy <= high:
            continue
        q = hkl @ reciprocal
        if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise RuntimeError("Reflection simulation produced an invalid reciprocal vector")
        rows.append((hkl, q, energy, intensity))
    if not rows:
        return _empty_result()

    q_values = np.asarray([row[1] for row in rows], dtype=np.float64)
    try:
        detector_xy = detector.q_to_pixel(q_values, depth=depth, on_detector=True)
    except Exception as error:
        raise RuntimeError("Failed to project simulated reflections onto the detector") from error
    finite = np.isfinite(detector_xy).all(axis=1)

    grouped: dict[
        tuple[int, int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    ] = {}
    for row, xy, keep in zip(rows, detector_xy, finite, strict=True):
        if not keep:
            continue
        hkl, q, energy, intensity = row
        key = _direction_key(hkl)
        candidate = (hkl, q, xy, energy, intensity)
        previous = grouped.get(key)
        if previous is None or (
            -_stable_order_value(intensity),
            _stable_order_value(energy),
            *hkl.tolist(),
        ) < (
            -_stable_order_value(previous[4]),
            _stable_order_value(previous[3]),
            *previous[0].tolist(),
        ):
            grouped[key] = candidate
    if not grouped:
        return _empty_result()

    ordered = sorted(
        grouped.values(),
        key=lambda row: (
            -_stable_order_value(row[4]),
            _stable_order_value(row[3]),
            *row[0].tolist(),
        ),
    )
    return SimulationResult(
        hkl=np.asarray([row[0] for row in ordered], dtype=np.int64),
        q=np.asarray([row[1] for row in ordered], dtype=np.float64),
        detector_xy=np.asarray([row[2] for row in ordered], dtype=np.float64),
        energy_kev=np.asarray([row[3] for row in ordered], dtype=np.float64),
        relative_intensity=np.asarray([row[4] for row in ordered], dtype=np.float64),
    )


def _validate_crystal(crystal: Crystal) -> None:
    if not crystal.atoms:
        raise ValueError("crystal must contain at least one atom site")
    cell = crystal.cell
    cell_values = np.asarray(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma], dtype=np.float64
    )
    if not np.isfinite(cell_values).all():
        raise ValueError("crystal cell parameters must be finite")
    for atom in crystal.atoms:
        if not isinstance(atom.symbol, str) or not atom.symbol.strip():
            raise ValueError("crystal atom symbols must be nonempty strings")
        position = np.asarray(atom.position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("crystal atom positions must be finite three-vectors")
        if not np.isfinite(atom.occupancy):
            raise ValueError("crystal atom occupancies must be finite")


def _validate_detector(detector: DetectorGeometry) -> None:
    if (
        isinstance(detector.nx, bool)
        or isinstance(detector.ny, bool)
        or not isinstance(detector.nx, (int, np.integer))
        or not isinstance(detector.ny, (int, np.integer))
        or detector.nx <= 0
        or detector.ny <= 0
    ):
        raise ValueError("detector dimensions must be positive integers")
    if (
        not np.isfinite(detector.size_x)
        or not np.isfinite(detector.size_y)
        or detector.size_x <= 0
        or detector.size_y <= 0
    ):
        raise ValueError("detector physical dimensions must be finite and positive")


def _real_scalar(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _energy_interval(value) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError("energy_range_kev must contain two real scalars")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError("energy_range_kev must contain two real scalars") from error
    if len(values) != 2:
        raise ValueError("energy_range_kev must contain exactly two values")
    low = _real_scalar(values[0], name="energy_range_kev lower bound")
    high = _real_scalar(values[1], name="energy_range_kev upper bound")
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("energy_range_kev must contain positive, increasing values")
    return low, high


def simulate_reflections(
    crystal: Crystal,
    reciprocal: np.ndarray,
    detector: DetectorGeometry,
    *,
    energy_range_kev: tuple[float, float] = (6.0, 30.0),
    depth: float = 0.0,
) -> SimulationResult:
    """Simulate a complete, deterministic on-detector reflection pattern.

    ``reciprocal`` contains reciprocal basis vectors as rows in ``1/nm``.
    Energy bounds are inclusive, and ``depth`` is measured in micrometres.
    One strongest-intensity representative is retained per signed primitive
    harmonic direction. Equal intensities prefer lower energy and then
    lexicographically smaller HKL; final rows use the same tie-breakers after
    descending intensity. Floating sort keys are compared to 12 significant
    digits so numerical noise in symmetry-equivalent calculations reaches the
    integer HKL tie-breaker.

    Parameters
    ----------
    crystal
        Package crystal description. The crystal must contain at least one
        atom site. Atom identity, fractional position, and occupancy enter the
        structure-factor calculation.
    reciprocal
        Finite, nonsingular reciprocal matrix with shape ``(3, 3)``. Basis
        vectors occupy rows in ``1/nm``. For Miller index row ``hkl``, the
        returned reciprocal vector is ``hkl @ reciprocal``.
    detector
        Metadata for one physical detector slot. Simulation uses its unbinned
        dimensions, physical size, translation, and rotation.
    energy_range_kev
        Two finite, positive, increasing energy bounds in keV. Both bounds are
        inclusive. The default is ``(6.0, 30.0)``.
    depth
        Finite sample depth in micrometres. The default is ``0.0``.

    Returns
    -------
    SimulationResult
        Complete direction-distinct reflections that intersect the selected
        detector within the requested energy interval. A valid simulation with
        no accepted reflections returns aligned empty arrays.

    Raises
    ------
    TypeError
        If a package-owned object or numeric input has an unsupported type.
    ValueError
        If a scientific input is malformed, non-finite, or otherwise invalid.
    RuntimeError
        If the private simulation backend cannot load, execute, return valid
        numerical output, or complete within its private candidate limit.
    """
    if not isinstance(crystal, Crystal):
        raise TypeError("crystal must be a laueanalysis.indexing.Crystal")
    if not isinstance(detector, DetectorGeometry):
        raise TypeError("detector must be a laueanalysis.indexing.DetectorGeometry")
    _validate_crystal(crystal)
    _validate_detector(detector)

    try:
        reciprocal_input = np.asarray(reciprocal)
    except Exception as error:
        raise TypeError("reciprocal must be a real numeric array") from error
    if reciprocal_input.dtype.kind not in "iuf":
        raise TypeError("reciprocal must be a real numeric array")
    reciprocal_array = np.array(reciprocal_input, dtype=np.float64, copy=True)
    if reciprocal_array.shape != (3, 3):
        raise ValueError(
            f"reciprocal must have shape (3, 3); received {reciprocal_array.shape}"
        )
    if not np.isfinite(reciprocal_array).all():
        raise ValueError("reciprocal must contain only finite values")
    try:
        rank = np.linalg.matrix_rank(reciprocal_array)
    except np.linalg.LinAlgError as error:
        raise ValueError("reciprocal must define a valid nonsingular basis") from error
    if rank != 3:
        raise ValueError("reciprocal must be nonsingular")
    energy_interval = _energy_interval(energy_range_kev)
    depth_value = _real_scalar(depth, name="depth")

    try:
        with _vendor_warning_boundary():
            modules = _load_jzt_modules()
    except Exception as error:
        raise RuntimeError("Failed to load the reflection simulation backend") from error
    try:
        with _vendor_warning_boundary():
            output = _execute_jzt(
                modules,
                crystal,
                reciprocal_array,
                detector,
                depth_value,
                energy_interval,
            )
    except Exception as error:
        raise RuntimeError("Failed to execute the reflection simulation backend") from error
    if output.limit_reached:
        raise RuntimeError(
            f"Reflection simulation reached its {_CANDIDATE_LIMIT:,}-candidate safety limit"
        )
    return _normalize_candidates(
        output.candidates,
        reciprocal_array,
        detector,
        depth_value,
        energy_interval,
    )
