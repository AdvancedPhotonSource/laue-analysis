# Geometry

Detector indices are physical slots from the geometry file. They are not
ordinal positions among active detectors, and slots may be sparse.

## Load Geometry

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autofunction:: load_geometry
```

## Parsed Geometry

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: Geometry
   :members: detector_count, wire, find_detector, detector, pixels_to_q

.. autoclass:: DetectorGeometry
   :members: pixel_to_lab, q_to_pixel

.. autoclass:: WireGeometry
```

`Geometry.detector()` returns an immutable metadata record containing the
selected detector's pixel dimensions, physical dimensions, pose, and
identifier. Physical dimensions and translation use micrometres. The
axis-angle rotation vector uses radians.

`Geometry.wire` returns immutable wire metadata when the geometry file contains
a complete wire section, or `None` otherwise. Diameter, `F`, and origin use
micrometres; the rotation magnitude uses degrees.
