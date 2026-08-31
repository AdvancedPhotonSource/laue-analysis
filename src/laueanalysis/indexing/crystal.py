"""Small public crystal descriptions for in-process indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


_LENGTH_TO_ANGSTROM = {"angstrom": 1.0, "a": 1.0, "nm": 10.0, "micron": 1e4}


@dataclass(frozen=True)
class Cell:
    """Crystallographic unit-cell parameters.

    Parameters
    ----------
    a, b, c
        Positive unit-cell lengths in ``unit``.
    alpha, beta, gamma
        Unit-cell angles in degrees. Each angle must be strictly between zero
        and 180 degrees.
    unit
        Length unit for ``a``, ``b``, and ``c``. Supported values are
        ``"angstrom"``, ``"A"``, ``"nm"``, and ``"micron"``.

    Raises
    ------
    ValueError
        If a length or angle is invalid, or the unit is unsupported.

    Notes
    -----
    Instances are immutable. Use :func:`dataclasses.replace` to derive a cell
    with changed parameters.
    """

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    unit: str = "nm"

    def __post_init__(self):
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("cell lengths must be positive")
        if not all(0 < angle < 180 for angle in (self.alpha, self.beta, self.gamma)):
            raise ValueError("cell angles must be between 0 and 180 degrees")
        if self.unit.lower() not in _LENGTH_TO_ANGSTROM:
            raise ValueError(f"unsupported cell unit: {self.unit}")

    @property
    def in_angstrom(self) -> "Cell":
        """Copy the cell with lengths converted to angstroms.

        Returns
        -------
        Cell
            A new cell with ``unit="angstrom"``. Angles are unchanged.
        """
        scale = _LENGTH_TO_ANGSTROM[self.unit.lower()]
        return Cell(
            self.a * scale, self.b * scale, self.c * scale,
            self.alpha, self.beta, self.gamma, "angstrom",
        )


@dataclass(frozen=True)
class Atom:
    """Atom site in a crystal description.

    Parameters
    ----------
    symbol
        Chemical element symbol passed to the native crystal model.
    position
        Three fractional coordinates within the unit cell.
    occupancy
        Site occupancy from zero through one, inclusive.
    label
        Optional site label. If omitted, XML output uses ``symbol``.

    Raises
    ------
    ValueError
        If ``position`` does not contain three coordinates or occupancy is
        outside the supported range.

    Notes
    -----
    Instances are immutable. Use :func:`dataclasses.replace` to derive a site
    with changed values.
    """

    symbol: str
    position: tuple[float, float, float]
    occupancy: float = 1.0
    label: str | None = None

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("atom position must contain three fractional coordinates")
        if not 0 <= self.occupancy <= 1:
            raise ValueError("atom occupancy must be between 0 and 1")


@dataclass(frozen=True)
class Crystal:
    """Immutable crystal description used for orientation indexing.

    Parameters
    ----------
    name
        Human-readable crystal or material name.
    space_group
        International Tables space-group number from 1 through 230.
    cell
        Crystallographic unit cell.
    atoms
        Atom sites. Any iterable supplied at construction is stored as a tuple.
    source
        Optional source XML path retained as provenance.
    setting
        Optional explicit space-group setting, such as ``"R"`` or ``"H"``
        for trigonal crystals.

    Raises
    ------
    ValueError
        If ``space_group`` is outside the range 1 through 230.

    Notes
    -----
    Instances are immutable. Use :func:`dataclasses.replace` to derive a
    modified crystal description.
    """

    name: str
    space_group: int
    cell: Cell
    atoms: tuple[Atom, ...] = ()
    source: str | None = None
    setting: str | None = None

    def __post_init__(self):
        if not 1 <= self.space_group <= 230:
            raise ValueError("space_group must be between 1 and 230")
        setting = self.setting.upper() if self.setting is not None else None
        if setting not in {None, "H", "R"}:
            raise ValueError("setting must be 'H', 'R', or None")
        if setting is not None and self.crystal_system != "trigonal":
            raise ValueError("an H or R setting is only valid for trigonal crystals")
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "setting", setting)

    @property
    def crystal_system(self) -> str:
        """Crystal system inferred from the space-group number.

        Returns
        -------
        str
            One of ``"triclinic"``, ``"monoclinic"``, ``"orthorhombic"``,
            ``"tetragonal"``, ``"trigonal"``, ``"hexagonal"``, or ``"cubic"``.
        """
        number = self.space_group
        if number <= 2:
            return "triclinic"
        if number <= 15:
            return "monoclinic"
        if number <= 74:
            return "orthorhombic"
        if number <= 142:
            return "tetragonal"
        if number <= 167:
            return "trigonal"
        if number <= 194:
            return "hexagonal"
        return "cubic"


def load_crystal(path: str | Path) -> Crystal:
    """Load a Laue crystal XML file.

    Parameters
    ----------
    path
        Path to a crystal XML file containing a cell and space-group number.

    Returns
    -------
    Crystal
        Immutable crystal description with ``source`` set to ``path``.

    Raises
    ------
    OSError
        If the file cannot be read.
    xml.etree.ElementTree.ParseError
        If the file is not well-formed XML.
    ValueError
        If required crystal data is missing, nonnumeric, or outside the
        supported ranges.

    Notes
    -----
    Atom sites without fractional coordinates are ignored. Missing occupancy
    defaults to one, and a missing chemical name defaults to the file stem.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    cell_node = root.find("cell")
    if cell_node is None:
        raise ValueError(f"Crystal file {path} has no cell")
    a_node = cell_node.find("a")
    if a_node is None:
        raise ValueError(f"Crystal file {path} has no cell length a")
    space_group = root.findtext("space_group_IT_number") or root.findtext("space_group/IT_number")
    if space_group is None:
        raise ValueError(f"Crystal file {path} has no space group")
    setting = None
    space_group_id = root.findtext("space_group/id")
    if space_group_id and ":" in space_group_id:
        suffix = space_group_id.rsplit(":", 1)[1].upper()
        if suffix in {"H", "R"}:
            setting = suffix

    cell = Cell(
        *(float(cell_node.findtext(name)) for name in ("a", "b", "c")),
        *(float(cell_node.findtext(name)) for name in ("alpha", "beta", "gamma")),
        unit=a_node.get("unit", "nm"),
    )
    atoms = []
    for site in root.findall("atom_site"):
        label = site.findtext("label", "")
        symbol = site.findtext("symbol") or "".join(value for value in label if not value.isdigit())
        position = site.findtext("fract") or site.findtext("fract_xyz")
        if position is None:
            continue
        atoms.append(Atom(
            symbol=symbol,
            position=tuple(float(value) for value in position.split()),
            occupancy=float(site.findtext("occupancy", "1")),
            label=label or None,
        ))
    return Crystal(
        name=root.findtext("chemical_name_common", path.stem),
        space_group=int(space_group),
        cell=cell,
        atoms=tuple(atoms),
        source=str(path),
        setting=setting,
    )
