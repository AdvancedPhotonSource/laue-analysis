"""XML generation utilities for Laue indexing results."""

from typing import List
from xml.etree import ElementTree
from xml.dom import minidom
from pathlib import Path

from laueanalysis.indexing.lau_dataclasses.step import Step


def write_step_xml(step: Step, xml_file: str) -> None:
    """
    Write a single step to an XML file.
    
    Args:
        step: Step object containing all indexing data.
        xml_file: Path to output XML file.
    """
    root = ElementTree.Element('AllSteps')
    root.append(step.getXMLElem())
    
    rough_string = ElementTree.tostring(root, short_empty_elements=False)
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent='    ')
    
    with open(xml_file, 'w') as f:
        f.write(pretty_xml)


def write_combined_xml(steps: List[Step], xml_file: str) -> None:
    """
    Write multiple steps to a combined XML file.
    
    Args:
        steps: List of Step objects to write.
        xml_file: Path to output XML file.
    """
    root = ElementTree.Element('AllSteps')
    for step in steps:
        root.append(step.getXMLElem())
    
    rough_string = ElementTree.tostring(root, short_empty_elements=False)
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent='    ')
    
    with open(xml_file, 'w') as f:
        f.write(pretty_xml)


def get_default_xml_filename(output_dir: str, input_image: str = '', prefix: str = '') -> str:
    """
    Get the default XML output filename based on the input image.
    
    XML files are stored in the 'xml' subdirectory of the output directory,
    following the same pattern as other intermediate files (peaks, p2q, index).
    
    Args:
        output_dir: Output directory path.
        input_image: Path to input image file (used to derive unique filename).
        prefix: Optional prefix for the filename.
        
    Returns:
        Full path to the XML file in the xml subdirectory.
    """
    xml_dir = Path(output_dir) / 'xml'
    
    # Create xml directory if it doesn't exist
    xml_dir.mkdir(parents=True, exist_ok=True)
    
    if input_image:
        # Derive filename from input image, similar to peaks/p2q/index files
        input_base = Path(input_image).stem
        filename = f'{prefix}indexed_{input_base}.xml' if prefix else f'indexed_{input_base}.xml'
    else:
        # Fallback for combined/batch operations
        filename = f'{prefix}indexed.xml' if prefix else 'indexed.xml'
    
    return str(xml_dir / filename)
