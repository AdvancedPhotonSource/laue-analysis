from xml.etree.ElementTree import Element, Comment
from dataclasses import dataclass, field

from laueanalysis.lau_dataclasses.pattern import Pattern
from laueanalysis.lau_dataclasses.xtl import Xtl


@dataclass
class Indexing:
    '''
    Example output:
    <indexing indexProgram="euler" Nindexed="39" Npeaks="39" Npatterns="3" keVmaxCalc="17.2" keVmaxTest="30.0" angleTolerance="0.1" cone="72.0" hklPrefer="1 1 1" executionTime="2.4">
        <!--Result of indexing.-->
        <pattern num="0" rms_error="0.0297" goodness="299.087" Nindexed="17">
            ...
        </pattern>
        <pattern num="1" rms_error="0.0917" goodness="216.471" Nindexed="11">
            ...
        </pattern>
        <xtl>
            ...
        </xtl>
    </indexing>
    '''

    indexProgram: str = ''
    Nindexed: int = None
    Npeaks: int = None
    NpatternsFound: int = None
    keVmaxCalc: float = None
    keVmaxTest: float = None
    angleTolerance: float = None
    cone: float = None
    hklPrefer: str = None
    executionTime: float = None
    patterns: list = field(default_factory=lambda: [])
    xtl = Xtl()

    def set(self, key, val):
        floats = ['keVmaxCalc', 'keVmaxTest', 'angleTolerance', 'cone', 'executionTime']
        if key in self.__dict__.keys():
            if key == 'hklPrefer':
                val = val.replace(',', ' ')
                for c in "{}'":
                    val = val.replace(c, '')
            if key in floats:
                val = float(val)
            self.__dict__[key] = val
        elif key[-1].isdigit() and not "Atom" in key:
            n = int(key[-1])
            if len(self.patterns) == n:
               self.patterns.append(Pattern(num=n))
            self.patterns[n].set(key[:-1], val)
        else:
            self.xtl.set(key, val)

    def getXMLElem(self) -> Element:
        elem = Element("indexing")
        attrs = ['indexProgram', 'Nindexed', 'Npeaks']
        for attr in attrs:
            elem.set(attr, str(getattr(self, attr)))
        #breaking up in the middle to preserve order, but need to rename NpatternsFound
        elem.set('Npatterns', self.NpatternsFound)
        attrs =['keVmaxCalc',
            'keVmaxTest', 'angleTolerance', 'cone', 'hklPrefer', 'executionTime']
        for attr in attrs:
            elem.set(attr, str(getattr(self, attr)))
        elem.append(Comment("Result of indexing."))
        for pattern in self.patterns:
            elem.append(pattern.getXMLElem())
        elem.append(self.xtl.getXMLElem())
        return elem