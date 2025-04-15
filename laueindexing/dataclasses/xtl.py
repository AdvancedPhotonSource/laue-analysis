from xml.etree.ElementTree import Element, SubElement
from dataclasses import dataclass, field

from laueindexing.dataclasses.atom import Atom


@dataclass
class Xtl:
    '''
    Example output:
    <xtl>
        <structureDesc>CdTe</structureDesc>
        <xtlFile>/clhome/EPIX34ID/dev/hannah-dev/laue-indexing/pipeline/config/CdTe.xml</xtlFile>
        <SpaceGroup>225</SpaceGroup>
        <latticeParameters unit="nm">0.648 0.648 0.648 90 90 90</latticeParameters>
        <atom n="1" symbol="Cd" label="Cd0">0 0 0</atom>
        <atom n="2" symbol="Te" label="Te4">0.25 0.25 0.25</atom>
    </xtl>
    '''

    structureDesc: str = None
    xtalFileName: str = None
    SpaceGroup: int = None
    latticeParameters: str = None
    lengthUnit: str = "nm"
    atoms: list = field(default_factory=lambda: [])

    def set(self, key, val): #TODO
        if key.startswith('latticeParameters'):
            for c in '{},':
                val = val.replace(c, '')
            val = val.strip()
            self.latticeParameters = val
        elif key in self.__dict__.keys():
            self.__dict__[key] = val
        elif key.startswith("Atom"):
            atom = Atom(n=key[-1])
            atom.fromDescription(val)
            self.atoms.append(atom)

    def getXMLElem(self) -> Element:
        elem = Element("xtl")
        SubElement(elem, 'structureDesc').text = self.structureDesc
        SubElement(elem, 'xtlFile').text = self.xtalFileName
        SubElement(elem, 'SpaceGroup').text = str(self.SpaceGroup)
        sub = Element('latticeParameters')
        sub.text = self.latticeParameters
        sub.set('unit', self.lengthUnit)
        elem.append(sub)
        for atom in self.atoms:
            elem.append(atom.getXMLElem())
        return elem