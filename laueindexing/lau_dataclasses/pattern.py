from xml.etree.ElementTree import Element
from dataclasses import dataclass, field

from laueindexing.lau_dataclasses.recipLattice import RecipLattice
from laueindexing.lau_dataclasses.hkls import HKLs


@dataclass
class Pattern:
    '''
    Example output:
    <pattern num="0" rms_error="0.0297" goodness="299.087" Nindexed="17">
        <recip_lattice unit="1/nm">
            ...
        </recip_lattice>
        <hkl_s>
            ...
        </hkl_s>
    </pattern>
    '''

    num: int = None
    rms_error: float = None
    goodness: float = None
    Nindexed: int = None
    recip_lattice: RecipLattice = field(default_factory=lambda: RecipLattice())
    hkl_s: HKLs = field(default_factory=lambda: HKLs())

    def set(self, key, val):
        floats = ['rms_error', 'goodness']
        if key.startswith('recip_lattice'):
            self.recip_lattice.fromString(val)
        elif key in self.__dict__.keys():
            if key in floats:
                val = float(val)
            self.__dict__[key] = val
        elif key.startswith('array'):
            self.Nindexed = val.split()[1]

    def getXMLElem(self) -> Element:
        elem = Element("pattern")
        elem.set("num", str(self.num))
        elem.set("rms_error", str(self.rms_error))
        elem.set("goodness", str(self.goodness))
        elem.set("Nindexed", str(self.Nindexed))
        elem.append(self.recip_lattice.getXMLElem())
        elem.append(self.hkl_s.getXMLElem())
        return elem