"""Small public crystal descriptions for in-process indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


_LENGTH_TO_ANGSTROM = {"angstrom": 1.0, "a": 1.0, "nm": 10.0, "micron": 1e4}


@dataclass(frozen=True)
class Cell:
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
        scale = _LENGTH_TO_ANGSTROM[self.unit.lower()]
        return Cell(
            self.a * scale, self.b * scale, self.c * scale,
            self.alpha, self.beta, self.gamma, "angstrom",
        )


@dataclass(frozen=True)
class Atom:
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
    name: str
    space_group: int
    cell: Cell
    atoms: tuple[Atom, ...] = ()
    source: str | None = None

    def __post_init__(self):
        if not 1 <= self.space_group <= 230:
            raise ValueError("space_group must be between 1 and 230")
        object.__setattr__(self, "atoms", tuple(self.atoms))

    @property
    def crystal_system(self) -> str:
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
    """Load a Laue crystal XML file into an editable immutable description."""
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
    )
