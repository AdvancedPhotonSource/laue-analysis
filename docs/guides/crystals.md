# Crystals

Orientation indexing needs a crystal description containing a unit cell and an International Tables space-group number. You can load the established XML format or construct the public data model in Python.

## Load a crystal

{func}`~lauelab.indexing.load_crystal` reads the supported crystal XML structure:

```python
from lauelab.indexing import load_crystal

crystal = load_crystal("Ni.xml")
print(crystal.name)
print(crystal.space_group)
print(crystal.cell)
```

The loader requires:

- A `cell` element with `a`, `b`, `c`, `alpha`, `beta`, and `gamma`
- A space-group number in `space_group_IT_number` or `space_group/IT_number`

It reads atom sites from `atom_site` elements. Sites without `fract` or `fract_xyz` coordinates are ignored. Occupancy defaults to `1`, and the crystal name defaults to the file stem.

## Construct a crystal

Use {class}`~lauelab.indexing.Cell`, {class}`~lauelab.indexing.Atom`, and {class}`~lauelab.indexing.Crystal` for generated or application-owned descriptions:

```python
from lauelab.indexing import Atom, Cell, Crystal

nickel = Crystal(
    name="Ni",
    space_group=225,
    cell=Cell(
        a=0.35238,
        b=0.35238,
        c=0.35238,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        unit="nm",
    ),
    atoms=(
        Atom("Ni", position=(0.0, 0.0, 0.0), label="Ni001"),
    ),
)
```

`Crystal.crystal_system` derives the crystal-system name from the space-group number.

## Units and coordinates

Cell angles use degrees. Cell lengths support `"angstrom"`, `"A"`, `"nm"`, and `"micron"`. The native indexing model converts lengths to angstroms.

```python
cell_angstrom = nickel.cell.in_angstrom
assert abs(cell_angstrom.a - 3.5238) < 1e-12
assert cell_angstrom.unit == "angstrom"
```

Atom positions are three fractional coordinates in the unit cell. Occupancy is a value from `0` through `1`, inclusive. The current public model validates the tuple length and occupancy range. It does not normalize fractional values to the interval from zero to one.

The space group is an International Tables number from 1 through 230. Before reciprocal-lattice calculations, the native model coerces the supplied cell to the constraints of that space group's crystal system. For example, cubic groups use `a` for all three lengths and 90 deg for all angles; tetragonal groups use `a` for `b`; hexagonal groups use `a` for `b` and 120 deg for gamma. Monoclinic, orthorhombic, and trigonal cells are adjusted by the corresponding symmetry constraints. This coercion is silent and also forces the ideal angles of each system, so validate generated lattice parameters before indexing. The per-system rules are listed in the [results guide](results.md).

`Crystal.setting` is retained as provenance and is written to indexing XML. It is not passed to the native model; the native rhombohedral-versus-hexagonal axis choice is inferred from the cell angles.

## Modify an immutable description

Crystal objects are frozen dataclasses. Use {func}`dataclasses.replace` to derive a modified value:

```python
from dataclasses import replace

body_centered = replace(nickel, space_group=229)

assert nickel.space_group == 225
assert body_centered.space_group == 229
```

Replace nested values separately when changing a cell:

```python
expanded_cell = replace(nickel.cell, a=0.353, b=0.353, c=0.353)
expanded_nickel = replace(nickel, cell=expanded_cell)
```

Constructing an `Indexer` from the modified crystal validates and transfers it to the native indexing model.

## Validation errors

Construction raises `ValueError` when:

- A cell length is zero or negative.
- A cell angle is not strictly between 0 deg and 180 deg.
- The cell unit is unsupported.
- An atom position does not contain three values.
- Occupancy is outside the interval from zero through one.
- The space-group number is outside 1 through 230.

Loading can also raise `OSError` for an unreadable file or `xml.etree.ElementTree.ParseError` for malformed XML. Missing required XML fields raise `ValueError`.

## Use the crystal for indexing

Pass either a `Crystal` or its XML path to `Indexer`:

```python
from lauelab.indexing import Indexer

from_model = Indexer("geometry.xml", nickel)
from_xml = Indexer("geometry.xml", "Ni.xml")
```

Omit the crystal only when you want peak search and pixel-to-q conversion without orientation indexing.
