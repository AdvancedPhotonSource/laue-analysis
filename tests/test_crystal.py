from dataclasses import FrozenInstanceError

import pytest

from laueanalysis.indexing import Atom, Cell, Crystal, load_crystal


@pytest.mark.parametrize(
    ("unit", "scale"),
    [("angstrom", 1.0), ("A", 1.0), ("nm", 10.0), ("micron", 1e4)],
)
def test_cell_converts_supported_units_to_angstrom(unit, scale):
    converted = Cell(1, 2, 3, 80, 90, 100, unit).in_angstrom

    assert converted == Cell(scale, 2 * scale, 3 * scale, 80, 90, 100, "angstrom")


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: Cell(0, 1, 1), "lengths"),
        (lambda: Cell(1, 1, 1, alpha=180), "angles"),
        (lambda: Cell(1, 1, 1, unit="inch"), "unit"),
        (lambda: Atom("Ni", (0, 0)), "three"),
        (lambda: Atom("Ni", (0, 0, 0), occupancy=1.1), "occupancy"),
        (lambda: Crystal("Ni", 0, Cell(1, 1, 1)), "space_group"),
        (lambda: Crystal("Ni", 231, Cell(1, 1, 1)), "space_group"),
    ],
)
def test_crystal_models_reject_invalid_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_crystal_models_are_frozen_and_normalize_atoms():
    atom = Atom("Ni", (0, 0, 0))
    crystal = Crystal("Ni", 225, Cell(0.352, 0.352, 0.352), [atom])

    assert crystal.atoms == (atom,)
    assert crystal.crystal_system == "cubic"
    with pytest.raises(FrozenInstanceError):
        crystal.space_group = 229


@pytest.mark.parametrize(
    ("space_group", "system"),
    [
        (1, "triclinic"),
        (3, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (195, "cubic"),
    ],
)
def test_crystal_system_boundaries(space_group, system):
    assert Crystal("test", space_group, Cell(1, 1, 1)).crystal_system == system


def test_load_crystal_supports_nested_space_group_and_atom_defaults(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text(
        """<crystal>
  <chemical_name_common>Sample</chemical_name_common>
  <space_group><IT_number>225</IT_number></space_group>
  <cell><a unit="angstrom">3</a><b>4</b><c>5</c>
    <alpha>80</alpha><beta>90</beta><gamma>100</gamma></cell>
  <atom_site><label>Ni12</label><fract_xyz>0 0.25 0.5</fract_xyz></atom_site>
</crystal>"""
    )

    crystal = load_crystal(path)

    assert crystal.name == "Sample"
    assert crystal.cell == Cell(3, 4, 5, 80, 90, 100, "angstrom")
    assert crystal.atoms == (Atom("Ni", (0, 0.25, 0.5), label="Ni12"),)
    assert crystal.source == str(path)


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        ("<crystal/>", "no cell"),
        ("<crystal><cell/></crystal>", "no cell length a"),
        ("<crystal><cell><a>1</a></cell></crystal>", "no space group"),
    ],
)
def test_load_crystal_reports_missing_required_sections(tmp_path, xml, message):
    path = tmp_path / "invalid.xml"
    path.write_text(xml)

    with pytest.raises(ValueError, match=message):
        load_crystal(path)
