# Geometry

Detector indices are physical slots from the geometry file. They are not
ordinal positions among active detectors, and slots may be sparse.

## Load Geometry

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autofunction:: load_geometry
```

## Parsed Geometry

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autoclass:: Geometry
   :members: detector_count, find_detector, detector, pixels_to_q

.. autoclass:: DetectorGeometry
   :members: pixel_to_lab, q_to_pixel
```

`Geometry.detector()` returns an immutable metadata record containing the
selected detector's pixel dimensions, physical dimensions, pose, and
identifier. Physical dimensions and translation use micrometres. The
axis-angle rotation vector uses radians.
