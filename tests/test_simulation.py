"""Scientific and model tests for the public reflection simulation API."""

from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import lauelab.analysis.simulation as simulation
from lauelab.analysis import (
    SimulationResult,
    lattice_params_to_reciprocal,
    simulate_reflections,
)
from lauelab.indexing import (
    Atom,
    Cell,
    Crystal,
    DetectorGeometry,
    load_crystal,
    load_geometry,
)


DATA = Path(__file__).parent / "data" / "simulation"


def _detector(*, distance=300_000.0, size=409_600.0, pixels=2048):
    return DetectorGeometry(
        nx=pixels,
        ny=pixels,
        size_x=size,
        size_y=size,
        detector_id="SYNTHETIC",
        translation=np.asarray([0.0, 0.0, distance]),
        rotation_vector=np.zeros(3),
        rotation=np.eye(3),
    )


def _case(case: str):
    with np.load(DATA / f"portal_jzt_{case}.npz") as raw:
        metadata = json.loads(raw["metadata_json"].item())
    detector_data = metadata["detector"]
    detector = DetectorGeometry(
        nx=detector_data["nx"],
        ny=detector_data["ny"],
        size_x=detector_data["size_x_um"],
        size_y=detector_data["size_y_um"],
        detector_id=detector_data["detector_id"],
        translation=np.asarray(detector_data["translation_um"]),
        rotation_vector=np.asarray(detector_data["rotation_vector_rad"]),
        rotation=np.eye(3),
    )
    if case == "si":
        crystal = Crystal(
            "Si",
            227,
            Cell(0.5431, 0.5431, 0.5431),
            (Atom("Si", (0.0, 0.0, 0.0), label="Si001"),),
        )
    else:
        crystal = load_crystal(Path(__file__).parents[1] / metadata["crystal"]["source"])
    return crystal, np.asarray(metadata["reciprocal_rows_1_per_nm"]), detector


def _result(hkl):
    hkl = np.asarray(hkl, dtype=np.int64).reshape((-1, 3))
    count = len(hkl)
    return SimulationResult(
        hkl=hkl,
        q=np.arange(count * 3, dtype=float).reshape((count, 3)),
        detector_xy=np.arange(count * 2, dtype=float).reshape((count, 2)),
        energy_kev=np.arange(count, dtype=float) + 6.0,
        relative_intensity=np.arange(count, dtype=float) + 1.0,
    )


def _spot(hkl, energy, intensity):
    return SimpleNamespace(
        hkl=np.asarray(hkl, dtype=np.int64).reshape((1, 3)),
        keV=energy,
        EwPo=intensity,
    )


def _mock_candidates(monkeypatch, candidates, *, limit_reached=False):
    monkeypatch.setattr(simulation, "_load_jzt_modules", lambda: object())
    monkeypatch.setattr(
        simulation,
        "_execute_jzt",
        lambda *args, **kwargs: simulation._BackendOutput(tuple(candidates), limit_reached),
    )


def _hkl_order(hkl):
    return np.lexsort((hkl[:, 2], hkl[:, 1], hkl[:, 0]))


def _simple_inputs():
    crystal = Crystal("Al", 1, Cell(0.5, 0.5, 0.5), (Atom("Al", (0, 0, 0)),))
    return crystal, np.eye(3), _detector(size=400_000.0, distance=100_000.0, pixels=100)


def test_simulation_result_owns_validates_and_freezes_arrays():
    hkl = np.asarray([[1, 2, 3]], dtype=np.int32)
    q = np.asarray([[1, 2, 3]], dtype=np.float32)
    result = SimulationResult(hkl, q, [[4, 5]], [6], [7])
    hkl[0, 0] = 99
    q[0, 0] = 99

    assert result.hkl.dtype == np.int64
    assert result.q.dtype == np.float64
    assert result.hkl[0, 0] == 1
    assert result.q[0, 0] == 1
    for field in fields(result):
        assert not getattr(result, field.name).flags.writeable
        with pytest.raises(ValueError):
            getattr(result, field.name).flat[0] = 0


def test_simulation_result_preserves_complete_empty_schema():
    result = simulation._empty_result()
    assert result.hkl.shape == (0, 3)
    assert result.q.shape == (0, 3)
    assert result.detector_xy.shape == (0, 2)
    assert result.energy_kev.shape == (0,)
    assert result.relative_intensity.shape == (0,)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("hkl", np.zeros((1, 2), dtype=int), ValueError),
        ("hkl", np.zeros((1, 3), dtype=float), TypeError),
        ("q", np.zeros((2, 3)), ValueError),
        ("detector_xy", np.zeros((1, 3)), ValueError),
        ("energy_kev", np.zeros(2), ValueError),
        ("relative_intensity", [np.nan], ValueError),
        ("q", np.asarray([[1 + 2j, 0, 0]]), TypeError),
    ],
)
def test_simulation_result_rejects_invalid_fields(field, value, error):
    values = {
        "hkl": np.ones((1, 3), dtype=int),
        "q": np.ones((1, 3)),
        "detector_xy": np.ones((1, 2)),
        "energy_kev": np.ones(1),
        "relative_intensity": np.ones(1),
    }
    values[field] = value
    with pytest.raises(error):
        SimulationResult(**values)


def test_missing_from_uses_signed_primitive_harmonic_directions():
    result = _result([[1, 1, 1], [-1, -1, -1], [2, 0, -1], [0, 1, -1]])
    missing = result.missing_from([[2, 2, 2], [4, 4, 4], [2, 0, -1]])

    np.testing.assert_array_equal(missing.hkl, [[-1, -1, -1], [0, 1, -1]])
    np.testing.assert_array_equal(missing.energy_kev, [7.0, 9.0])
    assert not missing.hkl.flags.writeable
    np.testing.assert_array_equal(result.missing_from([]).hkl, result.hkl)


def test_missing_from_validates_integer_shape_and_nonzero_direction():
    result = _result([[1, 1, 1]])
    with pytest.raises(TypeError, match="integer dtype"):
        result.missing_from([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        result.missing_from([1, 1, 1])
    with pytest.raises(ValueError, match="zero reflection"):
        result.missing_from([[0, 0, 0]])


def test_harmonic_grouping_keeps_strongest_and_sorts_deterministically(monkeypatch):
    candidates = [
        _spot([2, 0, -1], 10.0, 5.0),
        _spot([4, 0, -2], 20.0, 8.0),
        _spot([0, 2, -1], 12.0, 8.0),
        _spot([0, 4, -2], 9.0, 8.0),
        _spot([-2, 0, -1], 9.0, 8.0),
    ]
    _mock_candidates(monkeypatch, candidates)
    crystal, reciprocal, detector = _simple_inputs()

    first = simulate_reflections(crystal, reciprocal, detector, energy_range_kev=(6, 30))
    second = simulate_reflections(crystal, reciprocal, detector, energy_range_kev=np.asarray([6, 30]))

    np.testing.assert_array_equal(
        first.hkl,
        [[-2, 0, -1], [0, 4, -2], [4, 0, -2]],
    )
    np.testing.assert_array_equal(first.hkl, second.hkl)
    np.testing.assert_array_equal(first.relative_intensity, [8, 8, 8])
    np.testing.assert_array_equal(first.energy_kev, [9, 9, 20])


@pytest.mark.parametrize("perturbed_index", [0, 1])
def test_sort_uses_hkl_for_last_bit_float_differences(monkeypatch, perturbed_index):
    hkls = [[-1, 5, -3], [5, -1, -3]]
    candidates = [_spot(hkl, 20.0, 8.0) for hkl in hkls]
    candidates[perturbed_index] = _spot(
        hkls[perturbed_index],
        np.nextafter(20.0, np.inf),
        np.nextafter(8.0, np.inf),
    )
    _mock_candidates(monkeypatch, reversed(candidates))

    result = simulate_reflections(
        *_simple_inputs(),
        energy_range_kev=(6.0, 30.0),
    )

    np.testing.assert_array_equal(result.hkl, hkls)


def test_harmonic_representative_uses_stable_float_order(monkeypatch):
    candidates = [
        _spot([4, 0, -2], np.nextafter(20.0, np.inf), np.nextafter(8.0, np.inf)),
        _spot([2, 0, -1], 20.0, 8.0),
    ]
    _mock_candidates(monkeypatch, candidates)

    result = simulate_reflections(
        *_simple_inputs(),
        energy_range_kev=(6.0, 30.0),
    )

    np.testing.assert_array_equal(result.hkl, [[2, 0, -1]])


def test_energy_bounds_are_inclusive_and_off_detector_points_are_removed(monkeypatch):
    candidates = [
        _spot([2, 0, -1], 6.0, 3.0),
        _spot([0, 2, -1], 30.0, 2.0),
        _spot([1, 0, -1], 10.0, 100.0),
        _spot([-2, 0, -1], np.nextafter(6.0, 0.0), 50.0),
    ]
    _mock_candidates(monkeypatch, candidates)
    crystal, reciprocal, detector = _simple_inputs()
    result = simulate_reflections(crystal, reciprocal, detector)

    np.testing.assert_array_equal(result.hkl, [[2, 0, -1], [0, 2, -1]])
    np.testing.assert_array_equal(result.energy_kev, [6.0, 30.0])
    np.testing.assert_allclose(
        result.detector_xy,
        detector.q_to_pixel(result.q, depth=0.0, on_detector=True),
    )


def test_depth_uses_maintained_detector_projection(monkeypatch):
    _mock_candidates(monkeypatch, [_spot([2, 0, -1], 10.0, 3.0)])
    crystal, reciprocal, detector = _simple_inputs()
    result = simulate_reflections(crystal, reciprocal, detector, depth=25_000)
    expected = detector.q_to_pixel(result.q, depth=25_000, on_detector=True)
    np.testing.assert_allclose(result.detector_xy, expected)


def test_valid_simulation_with_no_candidates_returns_empty_result(monkeypatch):
    _mock_candidates(monkeypatch, [])
    result = simulate_reflections(*_simple_inputs())
    assert result.hkl.shape == (0, 3)


def test_backend_receives_fitted_row_basis_cell_units_and_occupancy(monkeypatch):
    recorded = {}

    class LatticeBase:
        Zmax = 109
        atomGeneral = SimpleNamespace(
            baseAtom=lambda symbol: SimpleNamespace(Z={"Ni": 28}[symbol])
        )

        @staticmethod
        def atomXtal(**kwargs):
            recorded.setdefault("atoms", []).append(kwargs)
            return kwargs

    class Lattice:
        @staticmethod
        def Lattice3D(*args, **kwargs):
            recorded["lattice"] = (args, kwargs)
            return object()

    class Pattern:
        def __init__(self, lattice, detector, recip):
            recorded["reciprocal"] = np.asarray(recip).copy()
            self._all_spots = []
            self._candidate_limit_reached = False

        def calc(self, **kwargs):
            recorded["calc"] = kwargs

    monkeypatch.setattr(
        simulation,
        "_load_jzt_modules",
        lambda: (LatticeBase, Lattice, SimpleNamespace(LauePattern=Pattern)),
    )
    crystal = Crystal(
        "mixed",
        1,
        Cell(5, 6, 7, 80, 90, 100, unit="angstrom"),
        (
            Atom("Ni", (0, 0, 0), occupancy=0.25, label="site1"),
            Atom("Ni", (0.5, 0.5, 0.5), occupancy=0.75, label="metal-B"),
        ),
    )
    reciprocal = np.asarray([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
    result = simulate_reflections(crystal, reciprocal, _detector())

    assert result.hkl.shape == (0, 3)
    assert [atom["occ"] for atom in recorded["atoms"]] == [0.25, 0.75]
    assert [atom["Zatom"] for atom in recorded["atoms"]] == [28, 28]
    assert [atom["label"] for atom in recorded["atoms"]] == ["site1", "metal-B"]
    np.testing.assert_allclose(recorded["lattice"][0][1], [0.5, 0.6, 0.7, 80, 90, 100])
    np.testing.assert_array_equal(recorded["reciprocal"], reciprocal.T)
    assert recorded["calc"]["ELO"] < 6.0
    assert recorded["calc"]["EHI"] > 30.0
    expected_q_max = (
        4
        * np.pi
        * np.sin(simulation._maximum_bragg_angle(_detector(), 0.0))
        * 30.0
        / simulation._HC_KEV_NM
    )
    expected_hkl_max = tuple(
        np.ceil(
            expected_q_max * np.linalg.norm(np.linalg.inv(reciprocal), axis=0)
        ).astype(int)
    )
    assert recorded["calc"]["hklMax"] == expected_hkl_max
    np.testing.assert_array_equal(reciprocal, [[1, 2, 3], [0, 4, 5], [0, 0, 6]])


def test_real_backend_uses_symbol_independently_of_site_label():
    reciprocal = np.eye(3) * (2 * np.pi / 0.5) @ _baseline_rotation()
    conventional = Crystal(
        "Ni conventional label",
        1,
        Cell(0.5, 0.5, 0.5),
        (Atom("Ni", (0, 0, 0), label="Ni1"),),
    )
    arbitrary = Crystal(
        "Ni arbitrary label",
        1,
        Cell(0.5, 0.5, 0.5),
        (Atom("Ni", (0, 0, 0), label="site1"),),
    )

    expected = simulate_reflections(
        conventional, reciprocal, _detector(), energy_range_kev=(6, 15)
    )
    actual = simulate_reflections(
        arbitrary, reciprocal, _detector(), energy_range_kev=(6, 15)
    )

    np.testing.assert_array_equal(actual.hkl, expected.hkl)
    np.testing.assert_allclose(actual.energy_kev, expected.energy_kev)
    np.testing.assert_allclose(actual.relative_intensity, expected.relative_intensity)


def test_real_backend_includes_high_angle_corner_reflection():
    detector = load_geometry(
        Path(__file__).parent / "data/geo/geoN_2022-03-29_14-15-05.xml"
    ).detector(0)
    target_pixel = np.asarray([20.0, 20.0])
    outgoing = detector.pixel_to_lab(target_pixel)
    outgoing /= np.linalg.norm(outgoing)
    target_q = outgoing - np.asarray([0.0, 0.0, 1.0])
    target_q /= np.linalg.norm(target_q)
    source_hkl = np.asarray([18.0, 1.0, 0.0])
    source_hkl /= np.linalg.norm(source_hkl)
    cross = np.cross(source_hkl, target_q)
    cosine = np.dot(source_hkl, target_q)
    cross_matrix = np.asarray([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    rotation = (
        np.eye(3)
        + cross_matrix
        + cross_matrix @ cross_matrix * ((1.0 - cosine) / np.dot(cross, cross))
    ).T
    reciprocal = (2 * np.pi / 0.5) * rotation
    crystal = Crystal(
        "Al high angle",
        1,
        Cell(0.5, 0.5, 0.5),
        (Atom("Al", (0, 0, 0)),),
    )

    result = simulate_reflections(
        crystal, reciprocal, detector, energy_range_kev=(6, 30)
    )
    row = np.flatnonzero(np.all(result.hkl == (18, 1, 0), axis=1))

    assert row.size == 1
    np.testing.assert_allclose(result.detector_xy[row[0]], target_pixel, atol=1e-9)
    assert result.energy_kev[row[0]] == pytest.approx(27.7406465)


def test_real_backend_honors_partial_occupancy():
    reciprocal = np.eye(3) * (2 * np.pi / 0.5) @ _baseline_rotation()
    full = Crystal(
        "Al full",
        1,
        Cell(0.5, 0.5, 0.5),
        (Atom("Al", (0, 0, 0), occupancy=1.0),),
    )
    partial = Crystal(
        "Al partial",
        1,
        Cell(0.5, 0.5, 0.5),
        (Atom("Al", (0, 0, 0), occupancy=0.5),),
    )
    complete = simulate_reflections(
        full, reciprocal, _detector(), energy_range_kev=(6, 15)
    )
    occupied = simulate_reflections(
        partial, reciprocal, _detector(), energy_range_kev=(6, 15)
    )
    np.testing.assert_array_equal(occupied.hkl, complete.hkl)
    np.testing.assert_allclose(
        occupied.relative_intensity,
        complete.relative_intensity * 0.5,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "reciprocal",
    [
        np.eye(2),
        np.full((3, 3), np.nan),
        np.zeros((3, 3)),
    ],
)
def test_invalid_reciprocal_values_fail_before_backend(monkeypatch, reciprocal):
    monkeypatch.setattr(simulation, "_load_jzt_modules", pytest.fail)
    crystal, _, detector = _simple_inputs()
    with pytest.raises(ValueError):
        simulate_reflections(crystal, reciprocal, detector)


def test_nonnumeric_reciprocal_is_a_type_error():
    crystal, _, detector = _simple_inputs()
    with pytest.raises(TypeError, match="real numeric"):
        simulate_reflections(crystal, "not a matrix", detector)


@pytest.mark.parametrize(
    "energy_range",
    [None, "6,30", (6,), (6, 30, 40), (-1, 30), (30, 6), (6, 6), (6, np.inf)],
)
def test_invalid_energy_ranges_are_rejected(energy_range):
    crystal, reciprocal, detector = _simple_inputs()
    error = TypeError if energy_range is None or isinstance(energy_range, str) else ValueError
    with pytest.raises(error):
        simulate_reflections(crystal, reciprocal, detector, energy_range_kev=energy_range)


@pytest.mark.parametrize("depth", [np.nan, np.inf])
def test_nonfinite_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="depth"):
        simulate_reflections(*_simple_inputs(), depth=depth)


def test_nonnumeric_depth_is_rejected():
    with pytest.raises(TypeError, match="depth"):
        simulate_reflections(*_simple_inputs(), depth="0")


def test_package_owned_types_and_scientific_metadata_are_required():
    crystal, reciprocal, detector = _simple_inputs()
    with pytest.raises(TypeError, match="Crystal"):
        simulate_reflections({}, reciprocal, detector)
    with pytest.raises(TypeError, match="DetectorGeometry"):
        simulate_reflections(crystal, reciprocal, {})
    with pytest.raises(ValueError, match="atom site"):
        simulate_reflections(Crystal("empty", 1, Cell(1, 1, 1)), reciprocal, detector)
    bad_atom = Atom("Al", (np.nan, 0, 0))
    with pytest.raises(ValueError, match="positions"):
        simulate_reflections(
            Crystal("bad", 1, Cell(1, 1, 1), (bad_atom,)), reciprocal, detector
        )


def test_invalid_detector_dimensions_are_rejected():
    crystal, reciprocal, detector = _simple_inputs()
    invalid = DetectorGeometry(
        0,
        detector.ny,
        detector.size_x,
        detector.size_y,
        detector.detector_id,
        detector.translation,
        detector.rotation_vector,
        detector.rotation,
    )
    with pytest.raises(ValueError, match="dimensions"):
        simulate_reflections(crystal, reciprocal, invalid)


def test_backend_import_execution_numerical_and_limit_failures_are_runtime_errors(monkeypatch):
    crystal, reciprocal, detector = _simple_inputs()
    monkeypatch.setattr(
        simulation,
        "_load_jzt_modules",
        lambda: (_ for _ in ()).throw(ImportError("missing resource")),
    )
    with pytest.raises(RuntimeError, match="load") as imported:
        simulate_reflections(crystal, reciprocal, detector)
    assert isinstance(imported.value.__cause__, ImportError)

    monkeypatch.setattr(simulation, "_load_jzt_modules", lambda: object())
    monkeypatch.setattr(
        simulation,
        "_execute_jzt",
        lambda *args: (_ for _ in ()).throw(ValueError("backend broke")),
    )
    with pytest.raises(RuntimeError, match="execute") as executed:
        simulate_reflections(crystal, reciprocal, detector)
    assert isinstance(executed.value.__cause__, ValueError)

    _mock_candidates(monkeypatch, [_spot([2, 0, -1], 10, np.nan)])
    with pytest.raises(RuntimeError, match="numerical"):
        simulate_reflections(crystal, reciprocal, detector)

    _mock_candidates(monkeypatch, [_spot([2, 0, -1], -1, 1)])
    with pytest.raises(RuntimeError, match="numerical"):
        simulate_reflections(crystal, reciprocal, detector)

    _mock_candidates(monkeypatch, [_spot([2, 0, -1], 10, 1)], limit_reached=True)
    with pytest.raises(RuntimeError, match="safety limit"):
        simulate_reflections(crystal, reciprocal, detector)


def test_unexpected_backend_warnings_are_not_suppressed(monkeypatch):
    crystal, reciprocal, detector = _simple_inputs()
    monkeypatch.setattr(simulation, "_load_jzt_modules", lambda: object())

    def execute(*args):
        warnings.warn("unexpected backend warning", UserWarning)
        return simulation._BackendOutput((), False)

    monkeypatch.setattr(simulation, "_execute_jzt", execute)
    with pytest.warns(UserWarning, match="unexpected backend warning"):
        simulate_reflections(crystal, reciprocal, detector)


def test_public_surface_has_no_backend_or_fallback_controls():
    assert set(SimulationResult.__dataclass_fields__) == {
        "hkl",
        "q",
        "detector_xy",
        "energy_kev",
        "relative_intensity",
    }
    assert list(inspect.signature(simulate_reflections).parameters) == [
        "crystal",
        "reciprocal",
        "detector",
        "energy_range_kev",
        "depth",
    ]
    import lauelab.analysis as analysis

    assert len(analysis.__all__) == 24
    assert "SimulationResult" in analysis.__all__
    assert "simulate_reflections" in analysis.__all__
    assert not any("backend" in name.lower() or "fallback" in name.lower() for name in analysis.__all__)


@pytest.mark.parametrize("case", ["ni", "cdte", "si"])
def test_real_simulation_is_finite_on_detector_warning_free_and_deterministic(case):
    crystal, reciprocal, detector = _case(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = simulate_reflections(crystal, reciprocal, detector)
    second = simulate_reflections(crystal, reciprocal, detector)

    assert len(first.hkl) > 0
    assert caught == []
    np.testing.assert_array_equal(first.hkl, second.hkl)
    np.testing.assert_allclose(first.q, second.q, rtol=0, atol=0)
    assert np.isfinite(first.q).all()
    assert np.isfinite(first.detector_xy).all()
    assert np.isfinite(first.energy_kev).all()
    assert np.isfinite(first.relative_intensity).all()
    assert np.all((first.detector_xy[:, 0] >= 0) & (first.detector_xy[:, 0] <= detector.nx - 1))
    assert np.all((first.detector_xy[:, 1] >= 0) & (first.detector_xy[:, 1] <= detector.ny - 1))
    np.testing.assert_allclose(first.q, first.hkl @ reciprocal, rtol=1e-14, atol=1e-14)
    with np.load(DATA / f"normalized_{case}.npz") as golden:
        actual_order = _hkl_order(first.hkl)
        golden_order = _hkl_order(golden["hkl"])
        np.testing.assert_array_equal(first.hkl[actual_order], golden["hkl"][golden_order])
        for name in ("q", "detector_xy", "energy_kev", "relative_intensity"):
            np.testing.assert_allclose(
                getattr(first, name)[actual_order],
                golden[name][golden_order],
                rtol=1e-12,
                atol=1e-12,
            )


def test_known_fcc_and_diamond_systematic_absences():
    ni = simulate_reflections(*_case("ni"))
    parity = np.mod(ni.hkl, 2)
    assert np.all(np.all(parity == 0, axis=1) | np.all(parity == 1, axis=1))

    si = simulate_reflections(*_case("si"))
    parity = np.mod(si.hkl, 2)
    all_odd = np.all(parity == 1, axis=1)
    all_even = np.all(parity == 0, axis=1)
    assert np.all(all_odd | (all_even & (np.mod(np.sum(si.hkl, axis=1), 4) == 0)))


def _baseline_rotation():
    source_u = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    source_v = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    source_w = np.cross(source_u, source_v)
    target_u = np.asarray([np.sqrt(1.0 - 0.18**2), 0.0, -0.18])
    target_v = np.asarray([0.0, 1.0, 0.0])
    target_w = np.cross(target_u, target_v)
    return np.vstack([source_u, source_v, source_w]).T @ np.vstack(
        [target_u, target_v, target_w]
    )


@pytest.mark.parametrize(
    ("space_group", "cell"),
    [
        pytest.param(1, Cell(0.5, 0.6, 0.7, 80, 90, 100), id="triclinic-constructed"),
        pytest.param(3, Cell(0.5, 0.6, 0.7, 90, 100, 90), id="monoclinic-constructed"),
        pytest.param(16, Cell(0.5, 0.6, 0.7), id="orthorhombic-constructed"),
        pytest.param(75, Cell(0.5, 0.5, 0.7), id="tetragonal-constructed"),
        pytest.param(143, Cell(0.5, 0.5, 0.8, 90, 90, 120), id="trigonal-constructed"),
        pytest.param(168, Cell(0.5, 0.5, 0.8, 90, 90, 120), id="hexagonal-constructed"),
        pytest.param(195, Cell(0.5, 0.5, 0.5), id="cubic-constructed"),
    ],
)
def test_constructed_crystal_system_code_paths(space_group, cell):
    crystal = Crystal(
        f"constructed SG {space_group}",
        space_group,
        cell,
        (Atom("Al", (0.123, 0.234, 0.345)),),
    )
    reciprocal = lattice_params_to_reciprocal(
        cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma
    ) @ _baseline_rotation()
    result = simulate_reflections(
        crystal,
        reciprocal,
        _detector(),
        energy_range_kev=(1.0, 40.0),
    )
    assert isinstance(result, SimulationResult)
    assert result.hkl.shape == result.q.shape
    assert result.detector_xy.shape == (len(result.hkl), 2)
    assert result.energy_kev.shape == result.relative_intensity.shape == (len(result.hkl),)
    assert np.isfinite(result.q).all()
    assert np.isfinite(result.detector_xy).all()
