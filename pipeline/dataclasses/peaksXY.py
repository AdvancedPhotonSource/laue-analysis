from xml.etree.ElementTree import Element, SubElement
from laueindexing.pipeline.dataclasses import dataclass, field


@dataclass
class PeaksXY:
    '''
    Example output:
    <peaksXY peakProgram="peaksearch" minwidth="0.2825" threshold="250.0" thresholdRatio="-1" maxRfactor="0.5" maxwidth="27.0" maxCentToFit="18.0" boxsize="18" max_number="50" min_separation="40" peakShape="Lorentzian" Npeaks="39" executionTime="0.83">
        <Xpixel>1704.115 1463.711 1094.592 1345.271 1032.963 1664.468 892.192 1500.050 1814.162 1202.683 1918.681 619.302 754.982 1275.487 317.543 838.083 1998.952 250.870 1762.528 1622.389 493.405 1355.047 752.711 1497.304 1071.988 960.364 1158.085 869.856 1867.585 1863.805 1782.721 965.961 1933.842 1907.283 1436.433 341.521 1744.426 927.703 1574.644</Xpixel>
        <Ypixel>1493.660 1737.600 925.371 619.504 28.137 95.464 1254.440 915.465 310.422 1344.027 850.537 1993.273 419.135 1670.509 1027.275 1793.053 1140.637 1308.351 243.636 1917.921 1062.587 1563.532 1489.811 1517.488 1298.635 583.345 599.347 663.881 479.368 1902.841 86.245 262.586 120.335 633.188 1278.684 255.509 863.499 1356.047 1459.388</Ypixel>
        <Intens>2483.0000 2338.0000 2174.0000 1670.0000 1393.0000 1200.0000 1069.0000 963.0000 900.0000 767.0000 742.0000 638.0000 617.0000 573.0000 565.0000 555.0000 500.0000 476.0000 467.0000 438.0000 427.0000 391.0000 390.0000 382.0000 381.0000 372.0000 368.0000 357.0000 353.0000 342.0000 332.0000 315.0000 309.0000 295.0000 285.0000 281.0000 278.0000 271.0000 265.0000</Intens>
        <Integral>30.02600 36.98400 27.73500 22.82400 16.89500 23.05900 8.94400 17.16700 26.87200 11.25400 12.14500 8.48700 6.16700 8.25800 6.65800 8.18200 9.29500 7.32400 5.55300 11.61800 3.19500 4.23300 7.47900 7.31100 5.12400 4.25300 8.72600 4.33900 6.37900 3.33900 9.62800 5.32400 11.56000 7.68600 3.73000 2.18100 2.10700 4.31800 0.98500</Integral>
        <hwhmX unit="pixel">1.107 1.004 0.987 1.181 1.198 1.417 0.932 1.556 2.603 1.457 1.186 0.982 1.160 1.259 1.087 1.148 1.854 1.192 1.231 2.206 1.519 1.158 3.623 1.607 1.593 1.225 1.601 1.576 1.755 1.037 1.723 1.673 2.508 2.168 1.097 1.402 1.331 1.665 1.248</hwhmX>
        <hwhmY unit="pixel">0.942 1.153 0.861 0.923 1.014 1.258 0.819 1.188 1.942 1.124 0.936 1.085 1.002 1.129 0.951 1.575 1.209 0.955 1.026 1.610 1.178 0.964 1.651 3.525 1.141 1.005 1.953 1.198 1.516 1.367 2.396 2.564 1.601 2.750 0.962 1.192 0.873 1.116 1.780</hwhmY>
        <tilt unit="degree">164.5492 69.2743 6.2127 2.0568 14.9765 176.4084 1.8397 150.0143 122.8955 154.6832 179.8712 166.6801 9.1637 157.6837 15.4393 51.2831 156.1021 8.1659 19.2314 163.0081 133.1425 171.5589 127.6689 50.3672 141.8025 164.0563 21.9228 145.3281 142.6694 52.6906 168.3150 21.8837 5.0060 19.8604 165.9225 156.5971 1.7353 34.1031 55.7726</tilt>
        <chisq>0.031715 0.029598 0.018705 0.015234 0.020577 0.01977 0.0060632 0.010082 0.039921 0.0060032 0.011936 0.0073307 0.0047077 0.0056353 0.0037517 0.0065082 0.010106 0.0037489 0.0088696 0.012357 0.004386 0.0042185 0.0059882 0.0098372 0.0033509 0.0031392 0.0041911 0.003022 0.0060183 0.0097026 0.0079393 0.0047969 0.0080864 0.0070334 0.0033307 0.003643 0.0044465 0.0028987 0.0044898</chisq>
        <Qx> 0.1473921  0.2116833 -0.0249273 -0.1169709 -0.2600605 -0.2706768  0.0673751 -0.0309414 -0.2177538  0.0961160 -0.0550029  0.2422832 -0.1546366  0.1884123  0.0056666  0.2056113  0.0389299  0.0730759 -0.2347791  0.2641481  0.0144263  0.1610323  0.1274954  0.1508276  0.0816110 -0.1181173 -0.1185052 -0.0944216 -0.1699722  0.2673992 -0.2774115 -0.2012380 -0.2737062 -0.1237768  0.0795782 -0.1774460 -0.0492088  0.0955249  0.1351552</Qx>
        <Qy> 0.7946154  0.7517902  0.7366568  0.7586552  0.6782382  0.7517344  0.7071843  0.7861381  0.7865677  0.7450420  0.8280727  0.6305395  0.6730260  0.7357548  0.6282377  0.6731577  0.8367101  0.6153383  0.7756670  0.7517537  0.6535932  0.7522897  0.6797822  0.7712514  0.7299217  0.7088175  0.7349371  0.7001570  0.8050966  0.7758741  0.7616813  0.6898304  0.7784773  0.8184319  0.7754941  0.6083358  0.8114495  0.7089171  0.7830557</Qy>
        <Qz>-0.5889499 -0.6245012 -0.6758072 -0.6409057 -0.6872856 -0.6013564 -0.7038118 -0.6172759 -0.5778361 -0.6600562 -0.5579160 -0.7373729 -0.7232728 -0.6505119 -0.7780009 -0.7103398 -0.5462606 -0.7848686 -0.5858494 -0.6042285 -0.7567085 -0.6388496 -0.7222472 -0.6184030 -0.6786412 -0.6954323 -0.6677005 -0.7077180 -0.5682684 -0.5714166 -0.5855634 -0.6954404 -0.5648522 -0.5611137 -0.6263196 -0.7735893 -0.5823472 -0.6987929 -0.6070888</Qz>
    </peaksXY>
    '''

    peakProgram: str = None
    minwidth: float = None
    threshold: float = None
    thresholdRatio: int = -1
    maxRfactor: float = None
    maxwidth: float = None
    maxCentToFit: float = None
    boxsize: int = None
    NpeakMax: int = None
    minSeparation: int = None
    peakShape: str = None
    Npeaks: int = None
    executionTime: float = None
    maskFile: str = None
    Xpixel: list = field(default_factory=lambda: [])
    Ypixel: list = field(default_factory=lambda: [])
    Intens: list = field(default_factory=lambda: [])
    Integral: list = field(default_factory=lambda: [])
    hwhmX: list = field(default_factory=lambda: [])
    hwhmXUnit: str = "pixel"
    hwhmY: list = field(default_factory=lambda: [])
    hwhmYUnit: str = "pixel"
    tilt: list = field(default_factory=lambda: [])
    tiltUnit: str = "degree"
    chisq: list = field(default_factory=lambda: [])
    Qx: list = field(default_factory=lambda: [])
    Qy: list = field(default_factory=lambda: [])
    Qz: list = field(default_factory=lambda: [])

    def set(self, key, val):
        if key in self.__dict__.keys():
            floats = ['minwidth', 'threshold', 'maxRfactor', 'maxwidth', 'maxCentToFit', 'executionTime']
            if key in ['Npeaks']:
                val = int(val)
            elif key in floats:
                val = float(val)
            self.__dict__[key] = val

    def addPeak(self, Xpixel, Ypixel, Intens, Integral, hwhmX, hwhmY, tilt, chisq):
        self.Xpixel.append(Xpixel)
        self.Ypixel.append(Ypixel)
        self.Intens.append(Intens)
        self.Integral.append(Integral)
        self.hwhmX.append(hwhmX)
        self.hwhmY.append(hwhmY)
        self.tilt.append(tilt)
        self.chisq.append(chisq)

    def addQVector(self, Qx, Qy, Qz):
        self.Qx.append(Qx)
        self.Qy.append(Qy)
        self.Qz.append(Qz)

    def getXMLElem(self) -> Element:
        elem = Element("peaksXY")
        attrs = ['peakProgram', 'minwidth', 'threshold', 'thresholdRatio',
            'maxRfactor', 'maxwidth', 'maxCentToFit', 'boxsize']

        for attr in attrs:
            elem.set(attr, str(getattr(self, attr)))

        elem.set('max_number', str(self.NpeakMax))
        elem.set('min_separation', str(self.minSeparation))

        attrs = ['peakShape', 'Npeaks', 'executionTime']
        for attr in attrs:
            elem.set(attr, str(getattr(self, attr)))

        if self.maskFile:
            elem.set('maskFile', self.maskFile)

        attrs = ['Xpixel', 'Ypixel', 'Intens', 'Integral']
        for attr in attrs:
            SubElement(elem, attr).text = ' '.join(getattr(self, attr))

        attrs = ['hwhmX', 'hwhmY', 'tilt']
        for attr in attrs:
            sub = Element(attr)
            sub.text = ' '.join(getattr(self, attr))
            sub.set('unit', getattr(self, attr + 'Unit'))
            elem.append(sub)

        attrs = ['chisq', 'Qx', 'Qy', 'Qz']
        for attr in attrs:
            SubElement(elem, attr).text = ' '.join(getattr(self, attr))

        return elem