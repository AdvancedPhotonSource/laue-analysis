import numpy as np
import pytest

from laueanalysis.analysis import (
    CUBIC_SYMMETRY,
    HEXAGONAL_SYMMETRY,
    SurfaceFrame,
    closest_pole_colors,
    cubic_hkl_family,
    cubic_ipf_colors,
    cubic_ipf_key,
    hsv_key,
    hsv_position_colors,
    misorientation_angle,
    orientation_to_rodrigues,
    pole_color_radius,
    pole_figure_points,
    rodrigues_colors,
    symmetry_operations,
)


def test_aps_surface_frames_are_right_handed_and_read_only():
    for name in ("normal", "F", "X", "H", "Y", "Z"):
        frame = SurfaceFrame.aps_34ide(name)
        np.testing.assert_allclose(np.cross(frame.tilt, frame.roll), frame.normal)
        np.testing.assert_allclose(
            np.asarray([frame.tilt, frame.roll, frame.normal])
            @ np.asarray([frame.tilt, frame.roll, frame.normal]).T,
            np.eye(3),
            atol=1e-15,
        )
        assert not frame.normal.flags.writeable
    with pytest.raises(ValueError, match="unknown"):
        SurfaceFrame.aps_34ide("missing")


def test_surface_frame_normalizes_and_validates_handedness():
    frame = SurfaceFrame.from_vectors(tilt=[2, 0, 0], roll=[0, 3, 0], normal=[0, 0, 4])
    np.testing.assert_allclose(frame.normal, [0, 0, 1])
    with pytest.raises(ValueError, match="right-handed"):
        SurfaceFrame.from_vectors(tilt=[1, 0, 0], roll=[0, 1, 0], normal=[0, 0, -1])


@pytest.mark.parametrize("hkl,count", [((1, 0, 0), 6), ((1, 1, 0), 12), ((1, 1, 1), 8), ((2, 1, 0), 24), ((3, 2, 1), 48)])
def test_cubic_hkl_family_counts(hkl, count):
    family = cubic_hkl_family(hkl)
    assert family.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(family, axis=1), 1)


def test_pole_figure_points_are_in_unit_circle_and_keep_pattern_ids():
    lattices = np.tile(np.eye(3), (3, 1, 1))
    points, indices = pole_figure_points(lattices, cubic_hkl_family((1, 0, 0)))
    assert points.shape[1] == 2
    assert set(indices) == {0, 1, 2}
    assert np.all(np.linalg.norm(points, axis=1) <= 1 + 1e-12)


def test_symmetry_and_misorientation_primitives():
    assert CUBIC_SYMMETRY.shape == (24, 3, 3)
    assert HEXAGONAL_SYMMETRY.shape == (12, 3, 3)
    np.testing.assert_allclose(symmetry_operations(225), CUBIC_SYMMETRY)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(orientation_to_rodrigues(rotation), [0, 0, 1], atol=1e-12)
    assert misorientation_angle(rotation, rotation, operations=CUBIC_SYMMETRY) == pytest.approx(0)
    with pytest.raises(ValueError, match="supported"):
        symmetry_operations(1)


def test_color_primitives_have_expected_shapes_and_ranges():
    directions = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    ipf = cubic_ipf_colors(directions)
    rods = rodrigues_colors(np.array([[0, 0, 0], [1, 0, 0]], dtype=float))
    hsv = hsv_position_colors([0, 1], [0, 0])
    assert ipf.shape == (3, 3)
    assert rods.shape == hsv.shape == (2, 3)
    assert np.all((ipf >= 0) & (ipf <= 1))
    np.testing.assert_allclose(hsv[0], [1, 1, 1])
    np.testing.assert_allclose(hsv[1], [1, 0, 0])


def test_closest_pole_colors_and_keys():
    colors = closest_pole_colors(
        np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]),
        np.array([0, 0, 1]),
        2,
    )
    np.testing.assert_allclose(colors[0], [1, 1, 1])
    assert cubic_ipf_key(16).shape == (16, 16, 4)
    assert hsv_key(16).shape == (16, 16, 4)
    assert pole_color_radius((0, 0), 45) == pytest.approx(np.tan(np.radians(22.5)))
