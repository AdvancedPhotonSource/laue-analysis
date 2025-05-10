#!/usr/bin/env python3

import unittest
import os
import tempfile
from argparse import Namespace
from laueindexing.xmlWriter import XMLWriter


class TestXMLWriter(unittest.TestCase):
    """Tests for the XMLWriter class in the laue_indexing package."""

    def setUp(self):
        """Set up test fixtures."""
        self.xml_writer = XMLWriter()
        
        # Create a minimal test data object with required attributes
        self.test_args = Namespace(
            title="Test Title",
            sampleName="Ni Sample",
            beamBad="0",
            lightOn="0",
            monoMode="0",
            CCDshutter="1",
            hutchTemperature="25.0",
            sampleDistance="1.5",
            date="2025-04-14",
            depth="0",
            Xsample="0",
            Ysample="0",
            Zsample="0",
            inputImage="test.h5",
            detectorID="PCO Edge",
            Nx="2048",
            Ny="2048",
            totalSum="1000000",
            sumAboveThreshold="500000",
            numAboveThreshold="1000",
            cosmicFilter="1",
            geoFile="geo.xml",
            energyUnit="keV",
            energy="30.0",
            exposureUnit="seconds",
            exposure="1.0",
            startx="0",
            endx="2048",
            groupx="1",
            starty="0",
            endy="2048",
            groupy="1",
            peakProgram="peaksearch",
            minwidth="2",
            threshold="100",
            thresholdRatio="3.0",
            maxRfactor="0.5",
            maxwidth="20",
            maxCentToFit="5",
            boxsize="5",
            max_peaks="50",
            min_separation="10",
            peakShape="G",
            Npeaks="2",
            fitX="1000.0 1500.0",
            fitY="1000.0 1500.0",
            intens="1.0 2.0",
            integral="10.0 20.0",
            hwhmX="2.0 3.0",
            hwhmY="2.0 3.0",
            tilt="0.0 0.0",
            chisq="0.1 0.1",
            qX="0.1 0.2",
            qY="0.1 0.2",
            qZ="0.1 0.2",
            Nindexed="1",
            Npatterns="1",
            angleTolerance="0.1",
            cone="72.0",
            executionTime="1.0",
            hklPrefer="001",
            indexProgram="euler",
            keVmaxCalc="30.0",
            keVmaxTest="35.0",
            rms_error0="0.1",
            goodness0="0.9",
            recipLatticeUnit="1/Angstrom",
            astar0="1.0 0.0 0.0",
            bstar0="0.0 1.0 0.0",
            cstar0="0.0 0.0 1.0",
            h0="1 0",
            k0="0 1",
            l0="0 0",
            PkIndex0="0 1",
            structureDesc="FCC",
            xtlFile="Ni.xml",
            SpaceGroup="Fm-3m",
            latticeParametersUnit="Angstrom",
            latticeParameters="3.52 3.52 3.52 90 90 90",
            Ni="28",
            label="Ni001",
            n="1",
            symbol="Ni",
            atom="0 0"
        )
        
    def test_get_step_element(self):
        """Test that getStepElement returns an XML element."""
        step_element = self.xml_writer.getStepElement(self.test_args)
        self.assertIsNotNone(step_element)
        
        # Check if the returned element has the expected structure
        self.assertEqual(step_element.tag, 'step')
        self.assertEqual(step_element.get('xmlns'), 'python3p8')
        
        # Check if step has a detector child
        detector = step_element.find('detector')
        self.assertIsNotNone(detector)
        
        # Check if step has an indexing child
        indexing = step_element.find('indexing')
        self.assertIsNotNone(indexing)

    def test_write_xml(self):
        """Test writing XML data to a file."""
        # Get a step element to write
        step_element = self.xml_writer.getStepElement(self.test_args)
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write XML to the temporary file
            self.xml_writer.write([step_element], tmp_path)
            
            # Check if the file exists and has content
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
            
            # Read the file and check for expected content
            with open(tmp_path, 'r') as f:
                content = f.read()
                # Check for key elements in the XML
                self.assertIn('<step xmlns="python3p8">', content)
                self.assertIn('<title>Test Title</title>', content)
                self.assertIn('<sampleName>Ni Sample</sampleName>', content)
                self.assertIn('<detector>', content)
                self.assertIn('<indexing', content)
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()