from xml.etree.ElementTree import Element, SubElement
from dataclasses import dataclass


@dataclass
class RecipLattice:
    '''
    Example output:
    <recip_lattice unit="1/nm">
        <astar>8.8720565 -2.4775674 -3.0275396</astar>
        <bstar>-2.9315057 0.7584848 -9.2113352</bstar>
        <cstar>2.5904845 9.3436654 -0.0550400</cstar>
    </recip_lattice>
    '''
    
    unit: str = "1/nm"
    astar: str = None
    bstar: str = None
    cstar: str = None

    def fromString(self, val):
        val = val.replace('{', '').replace('}', ' ') #leave a space
        val = val.split()
        self.astar = val[0].replace(',', ' ')
        self.bstar = val[1].replace(',', ' ')
        self.cstar = val[2].replace(',', ' ')

    def getXMLElem(self) -> Element:
        elem = Element("recip_lattice")
        elem.set("unit", self.unit)
        SubElement(elem, 'astar').text = self.astar
        SubElement(elem, 'bstar').text = self.bstar
        SubElement(elem, 'cstar').text = self.cstar
        return elem