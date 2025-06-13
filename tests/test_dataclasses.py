"""Smoke tests to verify that all modules can be imported correctly."""

import laueanalysis.indexing.lau_dataclasses


def test_dataclasses_module():
    """Test that the base dataclasses module can be imported."""
    assert hasattr(laueanalysis.indexing.lau_dataclasses, "__file__")


def test_atom_import():
    """Test that the Atom dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.atom import Atom
    assert hasattr(Atom, 'fromDescription')


def test_detector_import():
    """Test that the Detector dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.detector import Detector
    assert hasattr(Detector, 'getXMLElem')


def test_hkls_import():
    """Test that the HKLs dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.hkls import HKLs
    assert hasattr(HKLs, 'fromString')


def test_indexing_import():
    """Test that the Indexing dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.indexing import Indexing
    assert hasattr(Indexing, 'set')


def test_pattern_import():
    """Test that the Pattern dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.pattern import Pattern
    assert hasattr(Pattern, 'getXMLElem')


def test_peaksxy_import():
    """Test that the PeaksXY dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.peaksXY import PeaksXY
    assert hasattr(PeaksXY, 'addPeak')


def test_recip_lattice_import():
    """Test that the RecipLattice dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.recipLattice import RecipLattice
    assert hasattr(RecipLattice, 'fromString')


def test_roi_import():
    """Test that the ROI dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.roi import ROI
    assert hasattr(ROI, 'getXMLElem')


def test_step_import():
    """Test that the Step dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.step import Step
    assert hasattr(Step, 'fromH5')


def test_xtl_import():
    """Test that the Xtl dataclass can be imported."""
    from laueanalysis.indexing.lau_dataclasses.xtl import Xtl
    assert hasattr(Xtl, 'set')


def test_main_class_import():
    """Test that the main PyLaueGo class can be imported."""
    from laueanalysis.indexing.pyLaueGo import PyLaueGo
    assert hasattr(PyLaueGo, 'index')
    assert hasattr(PyLaueGo, 'p2q')