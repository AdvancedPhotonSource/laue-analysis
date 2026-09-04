# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""In-process wire-scan reconstruction driver."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Literal

import h5py
import numpy as np

from lauelab._native import ffi, get_library
from lauelab.indexing import Geometry
from lauelab.indexing.errors import InputError, ReconstructionError

from ._reader import ImageGeometry, cutoff_mask, normalization_plane, read_scan_info
from ._writer import (PIXEL_DTYPES, create_outputs, normalization_rescale,
                      pixel_type, write_stripe, write_summary)
from .reconstruct import ReconstructionResult

_LOG = logging.getLogger(__name__)
_EDGE = {"leading": 1, "trailing": 0, "both": -1}
_POSITIONER = {"none": 0, "pm500": 1, "alio": 2}


@dataclass(frozen=True)
class StripeTiming:
    """Elapsed I/O and native compute time for one row stripe.

    Attributes
    ----------
    row_start : int
        Zero-based first image row in the stripe.
    row_stop : int
        Exclusive image-row stop index.
    read_seconds : float
        Input read time in seconds.
    compute_seconds : float
        Native reconstruction time in seconds.
    write_seconds : float
        Output write time in seconds, or 0 when no files are written.
    """

    row_start: int
    row_stop: int
    read_seconds: float
    compute_seconds: float
    write_seconds: float


def physical_core_count() -> int:
    """Estimate physical cores, halving logical CPUs only when SMT is detected."""
    logical = os.cpu_count() or 1
    siblings = Path("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
    try:
        values = siblings.read_text().strip().split(",")
        if len(values) > 1 or any("-" in value for value in values):
            return max(1, logical // 2)
    except OSError:
        pass
    return logical


def _raise_native(library, status: int, stage: str, message: str) -> None:
    detail = f"{stage} failed: {message}"
    if status == library.LAUE_INVALID_ARGUMENT:
        raise InputError(detail)
    if status == library.LAUE_OUT_OF_MEMORY:
        raise MemoryError(detail)
    raise ReconstructionError(detail)


class Reconstructor:
    """Reusable in-process wire-scan reconstructor.

    Parameters
    ----------
    geometry : Geometry or pathlib.Path
        Parsed geometry or path to a geometry XML file. The geometry must
        contain a complete wire section.
    detector : int
        Active detector slot in ``geometry``. This is a physical geometry slot,
        not an ordinal position among active detectors.
    depth_range : tuple of float
        Inclusive ``(start, end)`` sample depths in µm along the incident beam,
        relative to the Si origin in ``geometry``. Values must be finite and
        nondecreasing. Equal endpoints request one depth.
    resolution : float
        Distance between reconstructed depths in µm. The default is ``1.0``.
        The value must be positive and finite.
    wire_edge : str
        Wire edge or edges used for reconstruction: ``"leading"``,
        ``"trailing"``, or ``"both"``. The default is ``"leading"``.
        ``"both"`` defaults file output to pixel-type code 1 when
        ``output_pixel_type`` is omitted.
    percent_brightest : float
        Percentage of the brightest intensity-map pixels retained by the
        reconstruction mask. The default is ``100.0``. The value must be
        greater than 0 and at most 100.
    normalization : str or None
        HDF5 vector below ``entry1`` used to scale file-input frames. The default
        is ``None``. ``"mA"``
        values are divided by 102 and ``"cnt3"`` values by 88100; other tags
        have no fixed divisor. A missing or short vector raises
        :class:`~lauelab.indexing.InputError`. This parameter does not apply to
        :meth:`reconstruct_array`; pass its ``scale`` argument instead.
    norm_exponent : float or None
        Exponent normalization applied from the intensity map. The default is
        ``None``. Values must be greater than 0 and at most 5. This normalization
        applies to file and array input.
    norm_threshold : float or None
        Positive intensity threshold for exponent normalization. The default is
        ``None``; in that case, the threshold is the mean plus five standard
        deviations of the lowest half of the intensity-map pixels.
    cosmic_filter : bool
        Apply the executable-compatible cosmic-ray filter before reconstruction.
        The default is ``False``.
    output_pixel_type : int or None
        Output-file pixel type. The default is ``None``. Code 0 is
        ``numpy.float32``, 1 is ``numpy.int32``, 2
        is ``numpy.int16``, 3 is ``numpy.uint16``, 5 is ``numpy.float64``, 6 is
        ``numpy.int8``, and 7 is ``numpy.uint8``. File input defaults to the
        input type when it has a corresponding code, otherwise to code 5;
        ``wire_edge="both"`` defaults to code 1. This parameter does not alter
        arrays returned by :meth:`reconstruct_array`.
    num_threads : int or None
        Positive OpenMP thread count for each reconstruction call. The default
        is ``None``, which estimates physical cores from Linux SMT topology and
        otherwise uses the logical CPU count.
    rows_per_stripe : int or None
        Positive number of image rows processed per stripe. The default is
        ``None``, which uses at most 256 rows and may use fewer to satisfy
        ``memory_limit_mb``.
    memory_limit_mb : int
        Positive stripe-buffer limit in MiB. The default is ``8192``. It does
        not include retained result images or HDF5 library buffers.

    Notes
    -----
    Invalid arguments and failures before stripe processing raise an exception.
    After processing starts, an expected reconstruction or I/O failure returns
    a :class:`ReconstructionResult` with ``success=False`` and partial progress.
    """

    def __init__(
        self,
        geometry: Geometry | str | os.PathLike,
        detector: int,
        *,
        depth_range: tuple[float, float],
        resolution: float = 1.0,
        wire_edge: Literal["leading", "trailing", "both"] = "leading",
        percent_brightest: float = 100.0,
        normalization: str | None = None,
        norm_exponent: float | None = None,
        norm_threshold: float | None = None,
        cosmic_filter: bool = False,
        output_pixel_type: int | None = None,
        num_threads: int | None = None,
        rows_per_stripe: int | None = None,
        memory_limit_mb: int = 8192,
    ) -> None:
        self.geometry = geometry if isinstance(geometry, Geometry) else Geometry(geometry)
        self.geometry_path = self.geometry.path
        if self.geometry.wire is None:
            raise InputError("geometry has no complete wire section")
        try:
            self.detector_geometry = self.geometry.detector(detector)
        except (TypeError, ValueError, OverflowError) as error:
            raise InputError(f"detector {detector!r} is not an active detector slot") from error
        if len(depth_range) != 2 or not np.isfinite(depth_range).all() or depth_range[0] > depth_range[1]:
            raise InputError("depth_range must contain finite nondecreasing values")
        if not np.isfinite(resolution) or resolution <= 0:
            raise InputError("resolution must be positive and finite")
        if wire_edge not in _EDGE:
            raise InputError("wire_edge must be 'leading', 'trailing', or 'both'")
        if not 0 < percent_brightest <= 100:
            raise InputError("percent_brightest must be greater than 0 and at most 100")
        if output_pixel_type is not None and output_pixel_type not in PIXEL_DTYPES:
            raise InputError("output_pixel_type must be one of 0, 1, 2, 3, 5, 6, or 7")
        if num_threads is not None and num_threads < 1:
            raise InputError("num_threads must be positive")
        if rows_per_stripe is not None and rows_per_stripe < 1:
            raise InputError("rows_per_stripe must be positive")
        if memory_limit_mb < 1:
            raise InputError("memory_limit_mb must be positive")
        self.detector = detector
        self.depth_range = tuple(float(value) for value in depth_range)
        self.resolution = float(resolution)
        self.wire_edge = wire_edge
        self.percent_brightest = float(percent_brightest)
        self.normalization = normalization
        self.norm_exponent = norm_exponent
        self.norm_threshold = norm_threshold
        self.cosmic_filter = bool(cosmic_filter)
        self.output_pixel_type = output_pixel_type
        self.num_threads = physical_core_count() if num_threads is None else num_threads
        self.rows_per_stripe = rows_per_stripe
        self.memory_limit_mb = memory_limit_mb

    def _create_handle(self, image_geometry: ImageGeometry):
        rows, cols = image_geometry.shape
        params = ffi.new("laue_recon_params *")
        params.depth_start_um, params.depth_end_um = self.depth_range
        params.resolution_um = self.resolution
        params.wire_edge = _EDGE[self.wire_edge]
        params.cosmic_filter = self.cosmic_filter
        params.nx_full = image_geometry.nx_full
        params.ny_full = image_geometry.ny_full
        params.start_i, params.start_j = image_geometry.start
        params.bin_i, params.bin_j = image_geometry.group
        params.n_rows_total, params.n_cols = rows, cols
        error = ffi.new("char[256]")
        library = get_library()
        handle = library.laue_recon_create(self.geometry._handle, self.detector, params, error, 256)
        if handle == ffi.NULL:
            raise InputError(f"reconstruction setup failed: {ffi.string(error).decode(errors='replace')}")
        return ffi.gc(handle, library.laue_recon_free)

    def _stripe_rows(self, n_images: int, n_depths: int, rows: int,
                     cols: int, input_itemsize: int) -> int:
        if self.rows_per_stripe is not None:
            return min(rows, self.rows_per_stripe)
        bytes_per_row = 2 * n_images * cols * input_itemsize + 2 * n_depths * cols * 8
        limit_bytes = self.memory_limit_mb * 2**20
        return min(rows, 256, max(1, limit_bytes // bytes_per_row))

    def reconstruct(self, path, output_base=None, *, return_images=False) -> ReconstructionResult:
        """Reconstruct one HDF5 point, optionally writing per-depth files.

        Parameters
        ----------
        path : pathlib.Path or str
            Input 34-ID-E multi-image HDF5 file.
        output_base : pathlib.Path, str, or None
            Output filename prefix. The default is ``None``. When supplied, one
            HDF5 file is written per depth plus ``<output_base>summary.txt``.
            Existing output files are replaced.
        return_images : bool
            Retain the unscaled reconstructed images in memory. The default is
            ``False``. This is independent of writing output files.

        Returns
        -------
        ReconstructionResult
            Reconstruction status, output paths, depth coordinates, intensity
            totals, timings, and optional images.

        Raises
        ------
        InputError
            If the input path, file metadata, geometry, or options are invalid,
            or setup fails before stripe processing begins.
        MemoryError
            If allocation fails before stripe processing begins.
        """
        path = Path(path)
        if not path.is_file():
            raise InputError(f"input file does not exist: {path}")
        with h5py.File(path, "r") as source:
            info = read_scan_info(source, self.normalization)
            data = source["entry1/data/data"]
            stripe_data = data if info.dtype == np.dtype(np.uint16) else data.astype("f8")
            return self._run(
                lambda row0, row1: np.ascontiguousarray(
                    stripe_data[1:-1, row0:row1, :]
                ),
                info.shape, info.dtype, info.wire_xyz,
                intensity_map=info.intensity_map,
                positioner=info.positioner,
                image_geometry=info.image_geometry,
                scale=info.scale,
                output_base=output_base,
                source=source,
                return_images=return_images,
                scan_number=info.scan_number,
                sample_position=info.sample_position,
                energy_kev=info.energy_kev,
            )

    def reconstruct_array(self, images, wire_xyz, *, intensity_map=None,
                          positioner="none", image_geometry: ImageGeometry | None = None,
                          scale=None) -> ReconstructionResult:
        """Reconstruct aligned in-memory images and raw wire positions.

        Parameters
        ----------
        images : numpy.ndarray
            Numeric array with shape ``(N, rows, columns)``. ``numpy.uint16``
            input remains ``uint16``; every other numeric dtype is converted to
            contiguous ``numpy.float64`` storage.
        wire_xyz : numpy.ndarray
            Raw wire positions with shape ``(N + 1, 3)`` in the acquisition
            coordinate system. No file-format bookkeeping offset is applied.
        intensity_map : numpy.ndarray or None
            Intensity map with shape ``(rows, columns)`` used for the bright-pixel
            mask and exponent normalization. The default is ``None``, which uses
            the first image.
        positioner : str
            Historical correction applied to ``wire_xyz``: ``"none"``,
            ``"pm500"``, or ``"alio"``. The default is ``"none"``.
        image_geometry : ImageGeometry or None
            Full-detector dimensions and ROI mapping. The default is ``None``,
            which describes an
            unbinned, zero-based full frame whose detector size is exactly
            ``(columns, rows)``. Binned images and detector ROIs require an
            explicit geometry; ``start`` and ``group`` use unbinned pixels.
        scale : numpy.ndarray or None
            Per-image dimensionless scale factors with shape ``(N,)``. The
            default is ``None``. This is the array-path equivalent of the
            constructor's HDF5 ``normalization`` vector.

        Returns
        -------
        ReconstructionResult
            A result whose ``images`` field is an unscaled ``numpy.float64``
            array. Array reconstruction does not write files.

        Notes
        -----
        The constructor's ``normalization`` and ``output_pixel_type`` parameters
        do not apply on this path. ``norm_exponent`` still applies through
        ``intensity_map``.

        Raises
        ------
        InputError
            If an array shape, dtype, positioner, geometry, or scale is invalid,
            or setup fails before stripe processing begins.
        MemoryError
            If allocation fails before stripe processing begins.
        """
        array = np.asarray(images)
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
            raise InputError("images must be a 3D numeric array")
        input_dtype = array.dtype
        array = np.ascontiguousarray(
            array, dtype=np.uint16 if input_dtype == np.dtype(np.uint16) else np.float64
        )
        rows, cols = array.shape[1:]
        if image_geometry is None:
            image_geometry = ImageGeometry(cols, rows, n_rows=rows, n_cols=cols)
        if image_geometry.shape != (rows, cols):
            raise InputError("image_geometry shape does not match images")
        if intensity_map is None:
            intensity_map = array[0]
        intensity_map = np.asarray(intensity_map, dtype=np.float64)
        if intensity_map.shape != (rows, cols):
            raise InputError("intensity_map shape does not match images")
        wires = np.ascontiguousarray(wire_xyz, dtype=np.float64)
        if wires.shape != (len(array) + 1, 3):
            raise InputError("wire_xyz must have shape (N + 1, 3)")
        if positioner not in _POSITIONER:
            raise InputError("positioner must be 'none', 'pm500', or 'alio'")
        if scale is not None:
            scale = np.ascontiguousarray(scale, dtype=np.float64)
            if scale.shape != (len(array),):
                raise InputError("scale must have shape (N,)")
        return self._run(
            lambda row0, row1: np.ascontiguousarray(array[:, row0:row1, :]),
            array.shape, input_dtype, wires, intensity_map=intensity_map,
            positioner=positioner, image_geometry=image_geometry, scale=scale,
            output_base=None, source=None, return_images=True, scan_number=None,
            sample_position=None, energy_kev=None,
        )

    def _run(self, read_stripe, shape, dtype, wire_xyz, *, intensity_map,
             positioner, image_geometry, scale, output_base, source,
             return_images, scan_number, sample_position, energy_kev) -> ReconstructionResult:
        library = get_library()
        handle = self._create_handle(image_geometry)
        status = library.laue_recon_set_wire_positions(
            handle, ffi.from_buffer("double[]", wire_xyz), len(wire_xyz), _POSITIONER[positioner]
        )
        if status:
            _raise_native(
                library, status, "wire-position setup",
                ffi.string(library.laue_recon_last_error(handle)).decode(),
            )
        n_images, rows, cols = shape
        n_depths = library.laue_recon_n_depths(handle)
        depth_um = np.asarray([library.laue_recon_depth_um(handle, i) for i in range(n_depths)])
        input_itemsize = 2 if np.dtype(dtype) == np.dtype(np.uint16) else 8
        stripe_rows = self._stripe_rows(n_images, n_depths, rows, cols, input_itemsize)
        mask = cutoff_mask(intensity_map, self.percent_brightest)
        plane, threshold = normalization_plane(intensity_map, self.norm_exponent, self.norm_threshold)
        output_type = self.output_pixel_type
        if output_type is None:
            if self.wire_edge == "both":
                output_type = 1
            else:
                try:
                    output_type = pixel_type(dtype)
                except ValueError:
                    output_type = 5
        rescale = normalization_rescale(output_type) if self.norm_exponent is not None else 1.0
        all_images = np.zeros((n_depths, rows, cols)) if return_images else None
        totals = np.zeros(n_depths)
        timings = []
        handles = []
        output_files = []
        last_completed = None
        io_thread = None
        pipeline_started = False
        try:
            if output_base is not None:
                if source is None:
                    raise InputError("output_base is only supported for HDF5 input")
                handles, output_files = create_outputs(
                    source, output_base, depth_um, (rows, cols), PIXEL_DTYPES[output_type],
                    cosmic_filter=self.cosmic_filter, norm_exponent=self.norm_exponent,
                    norm_threshold=threshold, norm_rescale=rescale,
                )
            ranges = [(row0, min(rows, row0 + stripe_rows)) for row0 in range(0, rows, stripe_rows)]
            read_started = perf_counter()
            stripe = read_stripe(*ranges[0])
            read_seconds = perf_counter() - read_started
            first_row0, first_row1 = ranges[0]
            first_output = np.zeros((n_depths, first_row1 - first_row0, cols))
            work = Queue(maxsize=1)
            ready = Queue(maxsize=1)

            def io_worker():
                while True:
                    item = work.get()
                    if item is None:
                        return
                    previous, row0, row1 = item
                    try:
                        write_seconds = 0.0
                        completed = None
                        if previous is not None and handles:
                            write_started = perf_counter()
                            if rescale != 1.0:
                                np.multiply(previous[2], rescale, out=previous[2])
                            write_stripe(handles, previous[1], previous[2])
                            write_seconds = perf_counter() - write_started
                            completed = previous[0]
                        read_started = perf_counter()
                        next_stripe = read_stripe(row0, row1) if row0 is not None else None
                        ready.put((next_stripe, perf_counter() - read_started,
                                   write_seconds, completed, None))
                    except Exception as error:
                        ready.put((None, 0.0, 0.0, None, error))

            io_thread = Thread(target=io_worker, name="lauelab-reconstruction-io")
            io_thread.start()
            pending_output = None
            started = perf_counter()
            for stripe_index, (row0, row1) in enumerate(ranges):
                next_range = ranges[stripe_index + 1] if stripe_index + 1 < len(ranges) else (None, None)
                work.put((pending_output, *next_range))
                out = first_output if stripe_index == 0 else np.zeros((n_depths, row1 - row0, cols))
                elapsed = ffi.new("double *")
                kind = library.LAUE_PIXEL_U16 if stripe.dtype == np.uint16 else library.LAUE_PIXEL_F64
                pipeline_started = True
                status = library.laue_recon_stripe(
                    handle, ffi.from_buffer(stripe), kind, n_images, row0, row1 - row0,
                    ffi.NULL if scale is None else ffi.from_buffer("double[]", scale),
                    ffi.NULL if plane is None else ffi.from_buffer("double[]", plane[row0:row1]),
                    ffi.from_buffer("unsigned char[]", mask[row0:row1]),
                    ffi.from_buffer("double[]", out), self.num_threads, elapsed,
                )
                stripe, next_read_seconds, write_seconds, completed, io_error = ready.get()
                if completed is not None:
                    last_completed = completed
                if io_error is not None:
                    raise io_error
                if timings:
                    timings[-1] = StripeTiming(
                        timings[-1].row_start, timings[-1].row_stop,
                        timings[-1].read_seconds, timings[-1].compute_seconds,
                        write_seconds,
                    )
                if status:
                    _raise_native(
                        library, status, "reconstruction",
                        ffi.string(library.laue_recon_last_error(handle)).decode(),
                    )
                totals += out.sum(axis=(1, 2))
                if all_images is not None:
                    all_images[:, row0:row1] = out
                pending_output = (stripe_index, row0, out)
                timings.append(StripeTiming(row0, row1, read_seconds, elapsed[0], 0.0))
                read_seconds = next_read_seconds
                if not handles:
                    last_completed = stripe_index
                _LOG.debug("reconstructed rows %d:%d in %.6f s", row0, row1, elapsed[0])
            work.put((pending_output, None, None))
            _, _, write_seconds, completed, io_error = ready.get()
            if completed is not None:
                last_completed = completed
            if io_error is not None:
                raise io_error
            if timings:
                timings[-1] = StripeTiming(
                    timings[-1].row_start, timings[-1].row_stop,
                    timings[-1].read_seconds, timings[-1].compute_seconds,
                    write_seconds,
                )
            elapsed_total = perf_counter() - started
            if output_base is not None:
                write_summary(
                    f"{output_base}summary.txt", input_path=str(source.filename),
                    output_base=str(output_base), geometry_path=str(self.geometry_path),
                    detector=self.detector, depth_um=depth_um, resolution=self.resolution,
                    wire_edge=_EDGE[self.wire_edge], output_type=output_type,
                    percent_brightest=self.percent_brightest,
                    memory_limit_mb=self.memory_limit_mb,
                    cosmic_filter=self.cosmic_filter, normalization=self.normalization,
                    norm_exponent=self.norm_exponent, norm_threshold=threshold,
                    norm_rescale=rescale, scan_number=scan_number,
                    sample_position=sample_position, energy_kev=energy_kev,
                    image_geometry=image_geometry, rows_per_stripe=stripe_rows,
                    elapsed=elapsed_total, depth_intensity=totals,
                )
                output_files.append(f"{output_base}summary.txt")
            return ReconstructionResult(
                True, output_files, "", command="liblaue", images=all_images,
                depth_um=depth_um, depth_intensity=totals, timings=timings,
                last_completed_stripe=last_completed,
            )
        except Exception as error:
            if not pipeline_started:
                raise
            return ReconstructionResult(
                False, output_files, "", error=str(error), command="liblaue",
                return_code=-1, images=all_images, depth_um=depth_um,
                depth_intensity=totals, timings=timings,
                last_completed_stripe=last_completed,
            )
        finally:
            if io_thread is not None:
                work.put(None)
                io_thread.join()
            for output in handles:
                output.close()
