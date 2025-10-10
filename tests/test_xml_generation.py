#!/usr/bin/env python3

"""Unit tests for XML generation from parsing functions."""

import pytest
import tempfile
import os
from pathlib import Path
from xml.etree import ElementTree
from xml.dom import minidom

from laueanalysis.indexing.parsers import (
    parse_peaks_file,
    parse_p2q_file,
    parse_indexing_file,
    parse_full_step_data
)
from laueanalysis.indexing.xml_utils import write_step_xml, write_combined_xml
from laueanalysis.indexing.lau_dataclasses.step import Step
from laueanalysis.indexing.lau_dataclasses.indexing import Indexing


@pytest.fixture
def sample_peaks_file():
    """Create a sample peaks file for testing."""
    content = """$title	 
$sampleName	 
$beamBad	0
$CCDshutter	out
$lightOn	0
$monoMode	white slitted
$Xsample	2502.0
$Ysample	-4029.29
$Zsample	-9262.29
$depth	40
$energy	19.9999
$hutchTemperature	24.6111
$sampleDistance	0.0
$totalSum	27237800.0
$sumAboveThreshold	170831.0
$numAboveThreshold	431.0
$Npeaks	3
$peakList	8\t3
      643.081     1428.503       1806.1321        38.63823      1.578      1.733   178.3092   0.059983
      524.532      327.581       1581.0948        35.07650      1.743      1.538    70.1596   0.060116
     1466.652      850.944       1434.4264        40.30882      1.813      1.971   171.0238   0.056656
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_p2q_file():
    """Create a sample p2q file for testing."""
    content = """$title	 
$sampleName	 
$N_Ghat+Intens	3
 0.1098640,  0.6673674, -0.7365804
-0.1680842,  0.6371837, -0.7521600
-0.0498679,  0.7812982, -0.6221627
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_index_file():
    """Create a sample indexing file for testing."""
    content = """$Nindexed	12
$NpatternsFound	1
$keVmaxCalc	17.8
$keVmaxTest	30.0
$angleTolerance	0.1
$cone	72.0
$hklPrefer	2 2 0
$executionTime	1.31
$rms_error0	0.00624
$goodness0	271.193
$Nindexed0	12
$recip_lattice0	{{-0.1924936,8.4843187,-12.9893838}{-15.4015677,-1.6708082,-0.8630868}{-1.8706858,12.8829168,8.4424995}}
[  0]   ( 0.2517352  0.7773854 -0.5764558)     (  3  -1   1)    1.0000,     8.8078,     0.01091      4
[  1]   ( 0.1558722  0.7066679 -0.6901625)     (  5  -1   1)    0.2379,    11.5256,     0.00902      6
[  2]   ( 0.1099165  0.6673192 -0.7366162)     (  7  -1   1)    0.0760,    14.8415,     0.00572      0
$structureDesc	Aluminum
$xtlFile	tests/data/crystal/Al.xtal
$SpaceGroup	225
$latticeParameters	0.40495 0.40495 0.40495 90 90 90
$atom0	0 0 0
$symbol0	Al
$label0	Al
$n0	1
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_h5_file():
    """Create a minimal mock H5 file for testing."""
    # For testing, we'll create a simple text file that mimics the structure
    # In real usage, this would be an actual HDF5 file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h5', delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_parse_peaks_file(sample_peaks_file):
    """Test parsing of peaks file."""
    step = Step()
    parse_peaks_file(sample_peaks_file, step)
    
    # Check that metadata was parsed
    assert step.Xsample == 2502.0
    assert step.Ysample == -4029.29
    assert step.Zsample == -9262.29
    assert step.depth == '40'
    assert step.energy == 19.9999
    
    # Check that peaks were parsed
    assert step.detector.peaksXY.Npeaks == 3
    assert len(step.detector.peaksXY.Xpixel) == 3
    assert step.detector.peaksXY.Xpixel[0] == '643.081'
    assert step.detector.peaksXY.Ypixel[0] == '1428.503'
    assert step.detector.peaksXY.Intens[0] == '1806.1321'


def test_parse_p2q_file(sample_p2q_file):
    """Test parsing of p2q file."""
    from laueanalysis.indexing.lau_dataclasses.peaksXY import PeaksXY
    
    peaks_xy = PeaksXY()
    parse_p2q_file(sample_p2q_file, peaks_xy)
    
    # Check that Q vectors were parsed
    assert len(peaks_xy.Qx) == 3
    assert len(peaks_xy.Qy) == 3
    assert len(peaks_xy.Qz) == 3
    
    # Check first Q vector - note the parser preserves spacing and newlines from the file
    assert peaks_xy.Qx[0] == ' 0.1098640'
    assert peaks_xy.Qy[0] == ' 0.6673674'  # Single space, not double
    assert peaks_xy.Qz[0] == '-0.7365804\n'  # Last value includes newline


def test_parse_indexing_file(sample_index_file):
    """Test parsing of indexing file."""
    indexing = parse_indexing_file(sample_index_file, n_peaks=13)
    
    # Check metadata - parser stores most values as their original types (int/float)
    assert indexing.Nindexed == '12'  # Stored as string by set() method
    assert indexing.NpatternsFound == '1'  # Stored as string by set() method
    assert indexing.keVmaxCalc == 17.8  # Converted to float by set() method
    assert indexing.keVmaxTest == 30.0  # Converted to float by set() method
    assert indexing.angleTolerance == 0.1  # Converted to float by set() method
    assert indexing.cone == 72.0  # Converted to float by set() method
    assert indexing.hklPrefer == '2 2 0'
    assert indexing.executionTime == 1.31  # Converted to float by set() method
    assert indexing.Npeaks == 13  # Set directly as int parameter
    
    # Check pattern data
    assert len(indexing.patterns) == 1
    pattern = indexing.patterns[0]
    assert pattern.num == 0
    assert pattern.rms_error == 0.00624  # Converted to float by Pattern.set() method
    assert pattern.goodness == 271.193  # Converted to float by Pattern.set() method
    
    # Check reciprocal lattice
    assert pattern.recip_lattice.astar == '-0.1924936 8.4843187 -12.9893838'
    
    # Check hkl data
    assert len(pattern.hkl_s.h) == 3
    assert pattern.hkl_s.h[0] == '3'
    assert pattern.hkl_s.k[0] == '-1'
    assert pattern.hkl_s.l[0] == '1'
    
    # Check crystal structure
    assert indexing.xtl.structureDesc == 'Aluminum'
    assert indexing.xtl.SpaceGroup == '225'


def test_xml_generation_structure():
    """Test that XML generation creates proper structure."""
    step = Step()
    step.title = 'Test Title'
    step.sampleName = 'Test Sample'
    step.beamline = '34ID-E'
    step.scanNum = 12345
    step.energy = 20.0
    step.energyUnit = 'keV'
    
    # Create indexing data with all required attributes to avoid None serialization
    step.indexing = Indexing()
    step.indexing.set('Nindexed', '0')
    step.indexing.set('indexProgram', 'euler')
    step.indexing.set('Npeaks', 0)
    step.indexing.set('NpatternsFound', '0')
    
    # Generate XML
    xml_elem = step.getXMLElem()
    
    # Convert to string for inspection
    xml_str = ElementTree.tostring(xml_elem, encoding='unicode')
    
    # Check that key elements are present
    assert '<step' in xml_str
    assert '<title>Test Title</title>' in xml_str
    assert '<sampleName>Test Sample</sampleName>' in xml_str
    assert '<beamline>34ID-E</beamline>' in xml_str
    assert '<scanNum>12345</scanNum>' in xml_str
    assert '<energy unit="keV">20.0</energy>' in xml_str
    assert '<detector>' in xml_str
    assert '<indexing' in xml_str


def test_write_step_xml():
    """Test writing a single step to XML file."""
    step = Step()
    step.title = 'Test'
    step.sampleName = 'Sample'
    step.beamline = '34ID-E'
    step.scanNum = 100
    step.energy = 15.0
    
    step.indexing = Indexing()
    step.indexing.set('Nindexed', '0')
    step.indexing.set('indexProgram', 'euler')
    step.indexing.set('Npeaks', 0)
    step.indexing.set('NpatternsFound', '0')
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        temp_xml = f.name
    
    try:
        write_step_xml(step, temp_xml)
        
        # Verify file was created
        assert os.path.exists(temp_xml)
        
        # Parse and verify structure
        tree = ElementTree.parse(temp_xml)
        root = tree.getroot()
        
        assert root.tag == 'AllSteps'
        assert len(root) == 1  # One step
        
        step_elem = root[0]
        assert step_elem.tag == 'step'
        
        # Find and check specific elements
        title_elem = step_elem.find('title')
        assert title_elem is not None
        assert title_elem.text == 'Test'
        
        beamline_elem = step_elem.find('beamline')
        assert beamline_elem is not None
        assert beamline_elem.text == '34ID-E'
        
    finally:
        if os.path.exists(temp_xml):
            os.unlink(temp_xml)


def test_write_combined_xml():
    """Test writing multiple steps to XML file."""
    steps = []
    for i in range(3):
        step = Step()
        step.title = f'Test {i}'
        step.scanNum = 100 + i
        step.indexing = Indexing()
        step.indexing.set('Nindexed', '0')
        step.indexing.set('indexProgram', 'euler')
        step.indexing.set('Npeaks', 0)
        step.indexing.set('NpatternsFound', '0')
        steps.append(step)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        temp_xml = f.name
    
    try:
        write_combined_xml(steps, temp_xml)
        
        # Verify file was created
        assert os.path.exists(temp_xml)
        
        # Parse and verify structure
        tree = ElementTree.parse(temp_xml)
        root = tree.getroot()
        
        assert root.tag == 'AllSteps'
        assert len(root) == 3  # Three steps
        
        # Check each step
        for i, step_elem in enumerate(root):
            assert step_elem.tag == 'step'
            title_elem = step_elem.find('title')
            assert title_elem.text == f'Test {i}'
            
    finally:
        if os.path.exists(temp_xml):
            os.unlink(temp_xml)


def test_xml_formatting():
    """Test that XML is properly formatted with indentation."""
    step = Step()
    step.title = 'Test'
    step.indexing = Indexing()
    step.indexing.set('Nindexed', '0')
    step.indexing.set('indexProgram', 'euler')
    step.indexing.set('Npeaks', 0)
    step.indexing.set('NpatternsFound', '0')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        temp_xml = f.name
    
    try:
        write_step_xml(step, temp_xml)
        
        # Read the file and check formatting
        with open(temp_xml, 'r') as f:
            content = f.read()
        
        # Check for XML declaration
        assert content.startswith('<?xml version="1.0" ?>')
        
        # Check for proper indentation (4 spaces)
        assert '\n    <step' in content
        assert '\n        <title>' in content
        
    finally:
        if os.path.exists(temp_xml):
            os.unlink(temp_xml)


def test_full_integration_with_real_data_structure():
    """Test full parsing and XML generation with realistic data structure."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as peaks_f:
        peaks_f.write("""$Xsample	100.0
$Ysample	200.0
$depth	50
$Npeaks	2
$peakList	8\t2
      643.081     1428.503       1806.1321        38.63823      1.578      1.733   178.3092   0.059983
      524.532      327.581       1581.0948        35.07650      1.743      1.538    70.1596   0.060116
""")
        peaks_file = peaks_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as p2q_f:
        p2q_f.write("""$N_Ghat+Intens	2
 0.1098640,  0.6673674, -0.7365804
-0.1680842,  0.6371837, -0.7521600
""")
        p2q_file = p2q_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as index_f:
        index_f.write("""$Nindexed	2
$NpatternsFound	1
$rms_error0	0.00624
$goodness0	271.193
$recip_lattice0	{{-0.1924936,8.4843187,-12.9893838}{-15.4015677,-1.6708082,-0.8630868}{-1.8706858,12.8829168,8.4424995}}
[  0]   ( 0.1098640  0.6673674 -0.7365804)     (  3  -1   1)    1.0000,     8.8078,     0.01091      0
[  1]   (-0.1680842  0.6371837 -0.7521600)     (  5  -1   1)    0.2379,    11.5256,     0.00902      1
$structureDesc	Aluminum
$SpaceGroup	225
""")
        index_file = index_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h5', delete=False) as h5_f:
        h5_file = h5_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as xml_f:
        xml_file = xml_f.name
    
    try:
        # Parse all files
        step = Step()
        parse_peaks_file(peaks_file, step)
        parse_p2q_file(p2q_file, step.detector.peaksXY)
        step.indexing = parse_indexing_file(index_file, n_peaks=2)
        step.indexing.set('indexProgram', 'euler')
        
        # Set additional metadata
        step.detector.set('geoFile', 'test_geo.xml')
        step.detector.set('cosmicFilter', 'False')
        step.detector.peaksXY.set('peakProgram', 'peaksearch')
        
        # Generate XML
        write_step_xml(step, xml_file)
        
        # Verify XML structure
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        
        assert root.tag == 'AllSteps'
        step_elem = root[0]
        
        # Check detector section
        detector = step_elem.find('detector')
        assert detector is not None
        
        peaks_xy = detector.find('peaksXY')
        assert peaks_xy is not None
        assert peaks_xy.get('Npeaks') == '2'
        
        # Check Q vectors
        qx = peaks_xy.find('Qx')
        assert qx is not None
        assert ' 0.1098640' in qx.text
        
        # Check indexing section
        indexing = step_elem.find('indexing')
        assert indexing is not None
        assert indexing.get('Nindexed') == '2'
        
        pattern = indexing.find('pattern')
        assert pattern is not None
        
    finally:
        # Cleanup
        for f in [peaks_file, p2q_file, index_file, h5_file, xml_file]:
            if os.path.exists(f):
                os.unlink(f)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
