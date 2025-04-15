from xml.etree.ElementTree import Element, SubElement
from laueindexing.pipeline.dataclasses import dataclass, field


@dataclass
class HKLs:
    '''
    Example output:
    <hkl_s>
        <h>0 -1 -2 1 -1 1 3 -1 1 2 2 -1 2 -4 3 -1 1</h>
        <k>2 3 8 5 7 7 7 9 9 6 10 7 8 10 7 9 9</k>
        <l>2 3 6 7 7 7 7 7 7 8 8 9 10 10 11 11 11</l>
        <PkIndex>2 18 4 0 3 6 11 12 14 1 17 10 21 32 29 36 34</PkIndex>
    </hkl_s>
    '''

    h: str = field(default_factory=lambda: [])
    k: list = field(default_factory=lambda: [])
    l: list = field(default_factory=lambda: [])
    PkIndex: list = field(default_factory=lambda: [])

    def fromString(self, val): #TODO
        for c in '[]()': #remove extra punctuation
            val = val.replace(c, '')
        vals = val.split()
        #$array0 10  14  G^ h k l intens E(keV) err(deg) PkIndex
        self.h.append(vals[4])
        self.k.append(vals[5])
        self.l.append(vals[6])
        self.PkIndex.append(vals[-1])

    def getXMLElem(self) -> Element:
        elem = Element("hkl_s")
        attrs = self.__dict__.keys()
        for attr in attrs:
            SubElement(elem, attr).text = ' '.join(getattr(self, attr))
        return elem