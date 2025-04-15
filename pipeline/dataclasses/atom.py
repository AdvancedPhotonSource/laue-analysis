from xml.etree.ElementTree import Element
from laueindexing.pipeline.dataclasses import dataclass


@dataclass
class Atom:
    '''
    Example output:
    <atom n="2" symbol="Te" label="Te4">0.25 0.25 0.25</atom>
    '''

    n: int = None
    symbol: str = None
    label: str = None
    values: str = None

    def fromDescription(self, description):
            description = description.replace('}', '').replace('{', '').split()
            self.symbol = ''.join([i for i in description[0] if not i.isdigit()]) #get rid of numbers
            self.label = description[0]
            self.values = ' '.join(description[1:-1])

    def getXMLElem(self) -> Element:
        elem = Element("atom")
        elem.set("n", str(self.n))
        elem.set("symbol", self.symbol)
        elem.set("label", self.label)
        elem.text = self.values
        return elem