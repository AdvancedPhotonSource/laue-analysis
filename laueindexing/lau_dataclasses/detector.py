from xml.etree.ElementTree import Element, SubElement
from dataclasses import dataclass

from laueindexing.lau_dataclasses.roi import ROI
from laueindexing.lau_dataclasses.peaksXY import PeaksXY


@dataclass
class Detector:
    '''
    Example output:
    <detector>
        <inputImage>/data34b/Run2023-1/XEOL223/HAs_long/LAUE1/HAs_long_laue1_1.h5</inputImage>
        <detectorID>PE1621 723-3335</detectorID>
        <exposure unit="sec">0.25</exposure>
        <Nx>2048</Nx>
        <Ny>2048</Ny>
        <totalSum>553646000.0</totalSum>
        <sumAboveThreshold>2500790.0</sumAboveThreshold>
        <numAboveThreshold>899.0</numAboveThreshold>
        <cosmicFilter>True</cosmicFilter>
        <geoFile>/clhome/EPIX34ID/dev/hannah-dev/laue-indexing/pipeline/config/geoN_2023-02-07_19-19-10.xml</geoFile>
        <ROI startx="0" endx="2047" groupx="1" starty="0" endy="2047" groupy="1"> </ROI>
        <peaksXY peakProgram="peaksearch" minwidth="0.2825" threshold="250.0" thresholdRatio="-1" maxRfactor="0.5" maxwidth="27.0" maxCentToFit="18.0" boxsize="18" max_number="50" min_separation="40" peakShape="Lorentzian" Npeaks="39" executionTime="0.83">
            ...
        </peaksXY>
    </detector>
    '''

    inputImage: str = None
    detectorID: str = None
    exposure: float = None
    exposureUnit: str = 'sec'
    Nx: int = None
    Ny: int = None
    totalSum: float = None
    sumAboveThreshold: float = None
    numAboveThreshold: float = None
    cosmicFilter: bool = None
    geoFile: str = None
    roi = ROI()
    peaksXY = PeaksXY()

    def set(self, key, val):
        floats = ['exposure', 'totalSum', 'sumAboveThreshold', 'numAboveThreshold']
        if key in self.__dict__.keys():
            if key in floats:
                val = float(val)
            self.__dict__[key] = val
        elif key in self.roi.__dict__.keys():
            self.roi.__dict__[key] = val
        else:
            self.peaksXY.set(key, val)

    def getXMLElem(self) -> Element:
        elem = Element("detector")
        SubElement(elem, "inputImage").text = self.inputImage
        SubElement(elem, "detectorID").text = self.detectorID
        exposure = SubElement(elem, 'exposure')
        exposure.set('unit', self.exposureUnit)
        exposure.text = str(self.exposure)

        attrs = ['Nx', 'Ny', 'totalSum', 'sumAboveThreshold', 'numAboveThreshold',
            'cosmicFilter', 'geoFile']
        for attr in attrs:
            SubElement(elem, attr).text = str(getattr(self, attr))

        elem.append(self.roi.getXMLElem())
        elem.append(self.peaksXY.getXMLElem())

        return elem