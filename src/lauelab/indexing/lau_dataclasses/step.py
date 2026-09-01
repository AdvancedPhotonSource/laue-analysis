# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from xml.etree.ElementTree import Element, SubElement
from dataclasses import dataclass, field
import math

import h5py

from lauelab.indexing.lau_dataclasses.detector import Detector
from lauelab.indexing.lau_dataclasses.indexing import Indexing

@dataclass
class Step:
    '''
    Example output:
    <step original_xmlns="http://sector34.xray.aps.anl.gov/34ide:indexResult">
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
        <depth>12.5</depth>
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

    original_xmlns: str = 'http://sector34.xray.aps.anl.gov/34ide:indexResult'
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
    depth: float | None = None
    energy: float = None
    energyUnit: str = 'keV'
    hutchTemperature: float = None
    sampleDistance: float = None
    detector: Detector = field(default_factory=Detector)
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
        elem.set('original_xmlns', self.original_xmlns)

        def add(name, value):
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                SubElement(elem, name).text = str(value)

        for attr in ('title', 'sampleName', 'userName', 'beamline', 'scanNum'):
            add(attr, getattr(self, attr))

        add('date', self.dateExposed)
        for attr in ('beamBad', 'CCDshutter', 'lightOn', 'monoMode', 'Xsample', 'Ysample', 'Zsample', 'depth'):
            add(attr, getattr(self, attr))

        if self.energy is not None and not (isinstance(self.energy, float) and math.isnan(self.energy)):
            energy = SubElement(elem, 'energy')
            energy.set('unit', self.energyUnit)
            energy.text = str(self.energy)
        for attr in ('hutchTemperature', 'sampleDistance'):
            add(attr, getattr(self, attr))
        elem.append(self.detector.getXMLElem())
        elem.append(self.indexing.getXMLElem())

        return elem
