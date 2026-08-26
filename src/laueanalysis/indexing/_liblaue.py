"""ABI-mode bindings for the in-process Laue indexing library."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from os import fspath
from pathlib import Path
from typing import Optional

import numpy as np
from cffi import FFI

ffi = FFI()
ffi.cdef(
    """
    typedef struct laue_geometry laue_geometry;
    typedef struct laue_crystal laue_crystal;
    enum {
        LAUE_OK = 0,
        LAUE_INVALID_ARGUMENT = 1,
        LAUE_OUT_OF_MEMORY = 2,
        LAUE_NUMERICAL_ERROR = 3,
        LAUE_INTERNAL_ERROR = 4
    };
    typedef struct {
        char name[60];
        double x, y, z, occupancy;
    } laue_atom;
    typedef struct {
        int nx, ny;
        double size_x, size_y;
        char detector_id[256];
    } laue_detector_info;
    typedef struct {
        int boxsize;
        double max_rfactor;
        int min_size;
        int min_separation;
        double threshold;
        double threshold_ratio;
        int peak_shape;
        int max_peaks;
        int smooth;
        const unsigned char *mask;
        int detect_binning;
    } laue_peak_params;
    typedef struct {
        double kev_max_calc, kev_max_test, angle_tolerance_deg, cone_deg;
        int hkl_prefer[3];
        int max_data;
    } laue_index_params;
    typedef struct {
        double fit_x, fit_y, intens, integral, hwhm_x, hwhm_y, tilt, chisq, background;
        double qhat[3];
    } laue_peak;
    typedef struct {
        double euler_deg[3], rotation[3][3], recip[3][3];
        double goodness, rms_error_deg;
        int n_indexed;
        int *hkl, *pk_index;
        double *err_deg, *energy_kev, *pred_intens;
    } laue_pattern;
    typedef struct {
        int nx, ny, startx, starty, groupx, groupy;
        double depth;
        double threshold_used, total_sum, sum_above_threshold;
        long num_above_threshold;
        int n_peaks;
        laue_peak *peaks;
        int n_patterns, n_indexed;
        laue_pattern *patterns;
        int status;
        char message[256];
    } laue_frame_result;
    laue_geometry *laue_geometry_from_file(const char *, char *, size_t);
    void laue_geometry_free(laue_geometry *);
    laue_crystal *laue_crystal_from_file(const char *, char *, size_t);
    laue_crystal *laue_crystal_create(const char *, int, double, double, double,
                                       double, double, double, const laue_atom *, size_t,
                                       char *, size_t);
    void laue_crystal_free(laue_crystal *);
    int laue_geometry_detector_count(const laue_geometry *);
    int laue_geometry_find_detector(const laue_geometry *, const char *);
    int laue_geometry_detector_info(const laue_geometry *, int, laue_detector_info *, char *, size_t);
    int laue_find_peaks(const unsigned short *, int, int, const laue_peak_params *, laue_frame_result *);
    int laue_pixels_to_q(const laue_geometry *, int, laue_frame_result *);
    int laue_index(const laue_crystal *, const laue_index_params *, laue_frame_result *);
    void laue_frame_result_free(laue_frame_result *);
    const char *laue_version(void);
    """
)


def _load_library():
    library = resources.files("laueanalysis.indexing.bin") / "liblaue.so"
    try:
        return ffi.dlopen(fspath(library))
    except OSError as error:
        raise ImportError(
            "liblaue.so is unavailable; rebuild laueanalysis to use the in-process indexer"
        ) from error


_lib = None


def get_library():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


class NativeCrystal:
    """Compiled crystal structure retained for repeated indexing calls."""

    def __init__(self, handle, library):
        self._library = library
        self._handle = ffi.gc(handle, library.laue_crystal_free)

    @classmethod
    def from_file(cls, path: str | Path):
        error = ffi.new("char[256]")
        library = get_library()
        handle = library.laue_crystal_from_file(fspath(path).encode(), error, 256)
        if handle == ffi.NULL:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(f"Failed to load crystal {path}: {message}")
        return cls(handle, library)

    @classmethod
    def create(cls, crystal):
        library = get_library()
        error = ffi.new("char[256]")
        atoms = ffi.new("laue_atom[]", len(crystal.atoms))
        for index, atom in enumerate(crystal.atoms):
            atoms[index].name = atom.symbol.encode()
            atoms[index].x, atoms[index].y, atoms[index].z = atom.position
            atoms[index].occupancy = atom.occupancy
        cell = crystal.cell.in_angstrom
        handle = library.laue_crystal_create(
            crystal.name.encode(), crystal.space_group,
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,
            atoms, len(crystal.atoms), error, 256,
        )
        if handle == ffi.NULL:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(f"Failed to initialize crystal: {message}")
        return cls(handle, library)


@dataclass(frozen=True)
class DetectorGeometry:
    """Validated metadata for one detector in a geometry file.

    Parameters
    ----------
    nx, ny
        Detector dimensions in unbinned pixels.
    size_x, size_y
        Detector dimensions in micrometres.
    detector_id
        Detector identifier stored in the geometry file.

    Notes
    -----
    Instances are returned by :meth:`Geometry.detector`. Detector slots may be
    sparse, so a detector's physical slot cannot be inferred from
    :attr:`Geometry.detector_count`.
    """

    nx: int
    ny: int
    size_x: float
    size_y: float
    detector_id: str


class Geometry:
    """Parsed detector geometry retained for repeated in-process conversion.

    Parameters
    ----------
    path
        Path to a Laue geometry XML file. Only detector geometry is loaded by
        this public interface.

    Raises
    ------
    ValueError
        If the detector declarations are malformed, incomplete, duplicated, or
        contain invalid dimensions or physical parameters.
    ImportError
        If the native ``liblaue`` library is unavailable.

    Notes
    -----
    Detector indices are physical slots from the geometry file, not ordinal
    positions among active detectors. Slots may therefore be sparse.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        error = ffi.new("char[256]")
        library = get_library()
        handle = library.laue_geometry_from_file(fspath(path).encode(), error, 256)
        if handle == ffi.NULL:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(f"Failed to load geometry {path}: {message}")
        self._library = library
        self._handle = ffi.gc(handle, library.laue_geometry_free)

    @property
    def detector_count(self) -> int:
        """Number of active detectors in the geometry."""
        return self._library.laue_geometry_detector_count(self._handle)

    def find_detector(self, detector_id: str) -> int:
        """Find the physical slot for a detector identifier.

        Parameters
        ----------
        detector_id
            Exact detector identifier from the geometry file.

        Returns
        -------
        int
            Physical detector slot, or ``-1`` if the identifier is absent.
        """
        return self._library.laue_geometry_find_detector(self._handle, detector_id.encode())

    def detector(self, detector_index: int = 0) -> DetectorGeometry:
        """Return metadata for an active detector slot.

        Parameters
        ----------
        detector_index
            Physical detector slot from the geometry file.

        Returns
        -------
        DetectorGeometry
            Validated detector dimensions and identity.

        Raises
        ------
        ValueError
            If the slot is outside the supported range or is inactive.
        """
        info = ffi.new("laue_detector_info *")
        error = ffi.new("char[256]")
        status = self._library.laue_geometry_detector_info(
            self._handle, detector_index, info, error, 256
        )
        if status:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(message)
        return DetectorGeometry(
            info.nx,
            info.ny,
            info.size_x,
            info.size_y,
            ffi.string(info.detector_id).decode(errors="replace"),
        )

    def pixels_to_q(
        self,
        peaks: np.ndarray,
        *,
        detector_index: int = 0,
        start: tuple[int, int] = (0, 0),
        group: tuple[int, int] = (1, 1),
        depth: Optional[float] = None,
    ) -> np.ndarray:
        """Convert peak coordinates to unit scattering vectors.

        Parameters
        ----------
        peaks
            Array-like peak coordinates with shape ``(n, 2)``. Columns are
            zero-based ``(x, y)`` coordinates in the supplied frame.
        detector_index
            Physical detector slot from the geometry file.
        start
            Zero-based detector ``(x, y)`` origin of the frame.
        group
            Positive detector-pixel grouping factors as ``(x, y)``.
        depth
            Optional finite sample depth in micrometres passed to the geometry
            calculation.

        Returns
        -------
        numpy.ndarray
            Unit scattering vectors with shape ``(n, 3)`` and ``float64``
            dtype.

        Raises
        ------
        ValueError
            If coordinates, region parameters, depth, or detector selection
            are invalid.
        RuntimeError
            If native pixel-to-q conversion fails.

        Notes
        -----
        Grouped coordinates are mapped to the center of the corresponding
        detector-pixel group before bounds validation and conversion.
        """
        pixels = np.asarray(peaks, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("peaks must have shape (n, 2)")
        if len(start) != 2 or len(group) != 2:
            raise ValueError("start and group must each contain two integers")
        if not all(isinstance(value, (int, np.integer)) for value in (*start, *group)):
            raise ValueError("start and group values must be integers")
        if min(start) < 0 or min(group) < 1:
            raise ValueError("start values must be nonnegative and group values positive")
        if depth is not None and not np.isfinite(depth):
            raise ValueError("depth must be finite when provided")
        if not np.isfinite(pixels).all():
            raise ValueError("peak coordinates must be finite")
        detector = self.detector(detector_index)
        if len(pixels):
            detector_pixels = np.asarray(start) + pixels * np.asarray(group) + (np.asarray(group) - 1) / 2
            if (detector_pixels < 0).any() or (detector_pixels[:, 0] > detector.nx - 1).any() or (
                detector_pixels[:, 1] > detector.ny - 1
            ).any():
                raise ValueError(
                    f"peak coordinates fall outside detector bounds {detector.nx}x{detector.ny}"
                )

        c_peaks = ffi.new("laue_peak[]", len(pixels))
        for index, (x, y) in enumerate(pixels):
            c_peaks[index].fit_x = x
            c_peaks[index].fit_y = y

        result = ffi.new("laue_frame_result *")
        result.startx, result.starty = start
        result.groupx, result.groupy = group
        result.depth = np.nan if depth is None else depth
        result.n_peaks = len(pixels)
        result.peaks = c_peaks
        status = self._library.laue_pixels_to_q(self._handle, detector_index, result)
        if status:
            raise RuntimeError(ffi.string(result.message).decode(errors="replace"))

        return np.asarray(
            [[c_peaks[i].qhat[j] for j in range(3)] for i in range(len(pixels))],
            dtype=np.float64,
        ).reshape((-1, 3))


def load_geometry(path: str | Path) -> Geometry:
    """Load detector geometry for reuse.

    Parameters
    ----------
    path
        Path to a Laue geometry XML file.

    Returns
    -------
    Geometry
        Parsed detector geometry backed by native state.

    Raises
    ------
    ValueError
        If the detector geometry is invalid.
    ImportError
        If the native ``liblaue`` library is unavailable.
    """
    return Geometry(path)


def version() -> str:
    return ffi.string(get_library().laue_version()).decode()
