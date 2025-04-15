from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree.ElementTree import ElementTree, tostring
from xml.etree import ElementTree
from xml.dom import minidom

class XMLWriter:
    def __init__(self):
        '''
        values in the tag are indicated here as 'attrs'
        values between the tag are indicated as 'texts'

        e.g.
        <peaksXY Npeaks='14' boxsize='5'>
                <Xpixel>1441.126 1217.307</Xpixel>
        </peaksXY>

        peaksXYAttrs = ['Npeaks', 'boxsize']
        peaksXYTexts = ['Xpixel']
        '''
        self.stepTexts = ['title', 'sampleName', 'beamBad', 'lightOn', 'monoMode', 'CCDshutter',
            'hutchTemperature', 'sampleDistance', 'date', 'depth', 'Xsample', 'Ysample', 'Zsample']
        self.detectorTexts = ['inputImage', 'detectorID', 'Nx', 'Ny', 'totalSum', 'sumAboveThreshold',
            'numAboveThreshold', 'cosmicFilter', 'geoFile']
        self.roiAttrs = ['startx', 'endx', 'groupx', 'starty', 'endy', 'groupy']
        self.peaksXYAttrs = ['peakProgram', 'minwidth', 'threshold', 'thresholdRatio', 'maxRfactor',
            'maxwidth', 'maxCentToFit', 'boxsize', 'max_number', 'min_separation', 'peakShape']
        self.peaksXYTexts = ['fitX', 'fitY', 'intens', 'integral', 'hwhmX', 'hwhmY', 'tilt', 'chisq', 'qX', 'qY', 'qZ']
        self.indexingAttrs = ['Nindexed', 'Npatterns', 'Npeaks', 'angleTolerance', 'cone', 'executionTime',
            'hklPrefer', 'indexProgram', 'keVmaxCalc', 'keVmaxTest']
        self.patternAttrs =['Nindexed', 'goodness', 'num', 'rms_error']
        self.recipLatticeTexts = ['astar', 'bstar', 'cstar']
        self.hklsTexts = ['h', 'k', 'l', 'PkIndex']
        self.xtlTexts = ['structureDesc', 'xtlFile', 'SpaceGroup']
        self.atomAttrs = ['Ni', 'label', 'n', 'symbol']

    def write(self, xmlSteps, xmlOutFile):
        allSteps = Element('AllSteps')
        for step in xmlSteps:
            allSteps.append(step)
        roughString = ElementTree.tostring(allSteps, 'utf-8')
        parsed = minidom.parseString(roughString)
        prettyXML = parsed.toprettyxml(indent='    ')
        with open(xmlOutFile, 'w') as f:
            f.write(prettyXML)

    def getStepElement(self, args):
        step = self._getElement('step', args, texts=self.stepTexts)
        step.set('xmlns', 'python3p8')
        energy = SubElement(step, 'energy', unit=args.energyUnit)
        energy.set('text', str(args.energy))
        step.append(self._getDetectorElement(args))
        step.append(self._getIndexingElement(args))
        return step

    def _getDetectorElement(self, args):
        detector = self._getElement('detector', args, texts=self.detectorTexts)
        exposure = SubElement(detector, 'exposure', unit=args.exposureUnit)
        exposure.set('text', str(args.exposure))
        detector.append(self._getElement('ROI', args, self.roiAttrs))
        if int(args.Npeaks) > 0:
            detector.append(self._getElement('peaksXY', args, self.peaksXYAttrs, self.peaksXYTexts))
        else:
            detector.append(self._getElement('peaksXY', args, self.peaksXYAttrs, []))
        return detector

    def _getIndexingElement(self, args):
        if int(args.Nindexed) > 0:
            indexing = self._getElement('indexing', args, self.indexingAttrs)
            for i in range(int(args.Npatterns)):
                indexing.append(self._getPatternElement(i, args))
            indexing.append(self._getXTLElement(args))
        else:
            indexing = self._getElement('indexing', args, ['Nindexed'])
        return indexing

    def _getPatternElement(self, num, args):
        '''
        N pattern elements have attrs with pattern number at the end of name
        output as
        <pattern num=0></pattern>
        <pattern num=1></pattern>
        ...
        '''
        pattern = Element('pattern')
        pattern.set('num', str(num))
        pattern.set('rms_error', getattr(args, f'rms_error{num}'))
        pattern.set('goodness', getattr(args, f'goodness{num}'))
        pattern.set('Nindexed', getattr(args, f'Nindexed'))
        recipLattice = Element('recip_lattice')
        recipLattice.set('unit', args.recipLatticeUnit)
        for text in self.recipLatticeTexts:
            elem = Element(text)
            elem.text = getattr(args, f'{text}{num}')
            recipLattice.append(elem)
        pattern.append(recipLattice)
        hkls = Element('hkl_s')
        for text in self.hklsTexts:
            elem = Element(text)
            elem.text = getattr(args, f'{text}{num}')
            hkls.append(elem)
        pattern.append(hkls)
        return pattern

    def _getXTLElement(self, args):
        xtl = self._getElement('xtl', args, texts=self.xtlTexts)
        latticeParameters = Element('latticeParameters')
        latticeParameters = SubElement(xtl, 'latticeParameters', unit=args.latticeParametersUnit)
        latticeParameters.text = args.latticeParameters
        xtl.append(self._getElement('atom', args, self.atomAttrs))
        return xtl

    def _getElement(self, tag, args, attrs=[], texts=[]):
        elem = Element(tag)
        for attr in attrs:
            elem.set(attr, str(getattr(args, attr)))
        for text in texts:
            SubElement(elem, text).text = str(getattr(args, text))
        return elem