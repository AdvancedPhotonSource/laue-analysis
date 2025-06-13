from xml.etree.ElementTree import Element, SubElement
from dataclasses import dataclass
import h5py

from laueanalysis.indexing.lau_dataclasses.detector import Detector
from laueanalysis.indexing.lau_dataclasses.indexing import Indexing

@dataclass
class Step:
    '''
    Example output:
    <step xmlns="">
        <title> </title>
        <sampleName> </sampleName>
        <userName>Liu</userName>
        <beamline>34ID-E</beamline>
        <scanNum>276800</scanNum>
        <date>2023-02-17T04:31:43-06:00</date>
        <beamBad>0</beamBad>
        <CCDshutter>out</CCDshutter>
        <lightOn>0</lightOn>
        <monoMode>white slitted</monoMode>
        <Xsample>-182.0</Xsample>
        <Ysample>3469.42</Ysample>
        <Zsample>1560.23</Zsample>
        <depth>nan</depth>
        <energy unit="keV">14.5533</energy>
        <hutchTemperature>23.65</hutchTemperature>
        <sampleDistance>0.0</sampleDistance>
        <detector>
            ...
        </detector>
        <indexing ...>
            ...
        </indexing>
    </step>
    '''

    xmlns: str = '' #TODO what does this value represent?
    title: str = ''
    sampleName: str = ''
    userName: str = ''
    beamline: str = '34ID-E'
    scanNum: int = None
    dateExposed: str = ''
    beamBad: int = None
    CCDshutter: str = ''
    lightOn: int = None
    monoMode: str = ''
    Xsample: float = None
    Ysample: float = None
    Zsample: float = None
    depth: str = 'nan' #default no depth
    energy: float = None
    energyUnit: str = 'keV'
    hutchTemperature: float = None
    sampleDistance: float = None
    detector = Detector()
    indexing: Indexing = None

    def fromH5(self, filename:str):
        get = lambda f, val : f[val][0].decode('UTF-8')
        with h5py.File(filename, 'r') as f:
            self.title = get(f, 'entry1/title') or ' '
            self.sampleName = get(f, 'entry1/sample/name') or ' '
            self.detector.detectorID = get(f, 'entry1/detector/ID')
            self.detector.Nx = f['entry1/detector/Nx'][0]
            self.detector.Ny = f['entry1/detector/Ny'][0]
            CCDshutter = int(f['entry1/microDiffraction/CCDshutter'][0])
            self.CCDshutter = 'out' if CCDshutter else 'in'

    def set(self, key, val):
        floats = ['Xsample', 'Ysample', 'Zsample', 'energy', 'hutchTemperature', 'sampleDistance']
        if key in self.__dict__.keys():
            if key in floats:
                val = float(val)
            self.__dict__[key] = val
        else:
            self.detector.set(key, val)

    def getXMLElem(self) -> Element:
        elem = Element("step")
        elem.set('xmlns', self.xmlns)
        attrs = ['title', 'sampleName', 'userName', 'beamline',
            'scanNum']
        for attr in attrs:
            sub = Element(attr)
            sub.text = str(getattr(self, attr))
            elem.append(sub)

        SubElement(elem, 'date').text = self.dateExposed
        attrs = ['beamBad', 'CCDshutter', 'lightOn',
            'monoMode', 'Xsample', 'Ysample', 'Zsample', 'depth']
        for attr in attrs:
            sub = Element(attr)
            sub.text = str(getattr(self, attr))
            elem.append(sub)

        energy = SubElement(elem, 'energy')
        energy.set('unit', self.energyUnit)
        energy.text = str(self.energy)
        attrs = ['hutchTemperature', 'sampleDistance']

        for attr in attrs:
            sub = Element(attr)
            sub.text = str(getattr(self, attr))
            elem.append(sub)
        elem.append(self.detector.getXMLElem())
        elem.append(self.indexing.getXMLElem())

        return elem