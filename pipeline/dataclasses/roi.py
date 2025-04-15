from xml.etree.ElementTree import Element
from laueindexing.pipeline.dataclasses import dataclass


@dataclass
class ROI:
    '''
    Example output:
    <ROI startx="0" endx="2047" groupx="1" starty="0" endy="2047" groupy="1"> </ROI>
    '''

    startx: int = None
    endx: int = None
    groupx: int = None
    starty: int = None
    endy: int = None
    groupy: int = None

    def getXMLElem(self) -> Element:
        elem = Element("ROI")
        attrs = self.__dict__.keys()
        for attr in attrs:
            elem.set(attr, str(getattr(self, attr)))
        elem.text = ' '
        return elem