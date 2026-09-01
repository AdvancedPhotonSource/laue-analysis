"""Shared frame input and detector ROI conversions."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import numpy as np


def roi_to_detector_pixels(points, start, group):
    """Convert ROI pixel centers to full-detector pixel coordinates."""
    points = np.asarray(points)
    start = np.asarray(start)
    group = np.asarray(group)
    return start + points * group + (group - 1) / 2.0


def detector_to_roi_pixels(points, start, group):
    """Convert full-detector pixel coordinates to ROI pixel centers."""
    points = np.asarray(points)
    start = np.asarray(start)
    group = np.asarray(group)
    return (points - start - (group - 1) / 2.0) / group


def roi_inclusive_end(image_shape, start, group):
    """Return the inclusive full-detector ``(x, y)`` endpoint of an ROI."""
    size = np.asarray(image_shape)[::-1]
    return tuple(np.asarray(start) + size * np.asarray(group) - 1)


def read_h5_frame(path: str | Path):
    """Read the supported HDF5 image, metadata, and optional ROI settings."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to read an HDF5 frame") from error

    def scalar(source, name):
        if name not in source:
            return None
        value = source[name][()]
        if isinstance(value, np.ndarray):
            value = value.flat[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value.item() if isinstance(value, np.generic) else value

    def integer(source, name):
        value = scalar(source, name)
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"HDF5 field {name!r} must be an integer")
        converted = int(value)
        if not np.isfinite(value) or value != converted:
            raise ValueError(f"HDF5 field {name!r} must be an integer")
        return converted

    with h5py.File(path, "r") as source:
        image = source["entry1/data/data"][...]
        shutter = scalar(source, "entry1/microDiffraction/CCDshutter")
        position = tuple(
            scalar(source, name) for name in (
                "entry1/sample/sampleX",
                "entry1/sample/sampleY",
                "entry1/sample/sampleZ",
            )
        )
        file_time = source.attrs.get("file_time")
        if isinstance(file_time, bytes):
            file_time = file_time.decode("utf-8")
        metadata = {
            "title": scalar(source, "entry1/title"),
            "sample_name": scalar(source, "entry1/sample/name"),
            "user_name": scalar(source, "entry1/user/name"),
            "beamline": scalar(source, "Facility/facility_beamline"),
            "scan_number": scalar(source, "entry1/scanNum"),
            "date_exposed": file_time,
            "beam_bad": scalar(source, "entry1/microDiffraction/BeamBad"),
            "ccd_shutter": (
                ("out" if shutter else "in") if shutter is not None else None
            ),
            "light_on": scalar(source, "entry1/microDiffraction/LightOn"),
            "mono_mode": scalar(source, "entry1/microDiffraction/MonoMode"),
            "sample_position": (
                position if all(value is not None for value in position) else None
            ),
            "energy_kev": scalar(source, "entry1/sample/incident_energy"),
            "hutch_temperature": scalar(source, "entry1/microDiffraction/HutchTemperature"),
            "sample_distance": scalar(source, "entry1/sample/distance"),
            "detector_id": scalar(source, "entry1/detector/ID"),
            "exposure_seconds": scalar(source, "entry1/detector/exposure"),
        }
        metadata = {name: value for name, value in metadata.items() if value is not None}

        processing = {}
        start = (
            integer(source, "entry1/detector/startx"),
            integer(source, "entry1/detector/starty"),
        )
        group = (
            integer(source, "entry1/detector/binx"),
            integer(source, "entry1/detector/biny"),
        )
        if all(value is not None for value in start):
            processing["start"] = start
        if all(value is not None for value in group):
            processing["group"] = group

    return image, metadata, processing
