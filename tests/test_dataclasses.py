import unittest

class TestImports(unittest.TestCase):
    """Smoke tests to verify that all modules can be imported correctly."""

    def test_dataclasses_module(self):
        """Test that the base dataclasses module can be imported."""
        import laueindexing.lau_dataclasses
        self.assertTrue(hasattr(laueindexing.lau_dataclasses, "__file__"))
    
    def test_atom_import(self):
        """Test that the Atom dataclass can be imported."""
        from laueindexing.lau_dataclasses.atom import Atom
        self.assertTrue(hasattr(Atom, 'fromDescription'))
    
    def test_detector_import(self):
        """Test that the Detector dataclass can be imported."""
        from laueindexing.lau_dataclasses.detector import Detector
        self.assertTrue(hasattr(Detector, 'getXMLElem'))
    
    def test_hkls_import(self):
        """Test that the HKLs dataclass can be imported."""
        from laueindexing.lau_dataclasses.hkls import HKLs
        self.assertTrue(hasattr(HKLs, 'fromString'))
    
    def test_indexing_import(self):
        """Test that the Indexing dataclass can be imported."""
        from laueindexing.lau_dataclasses.indexing import Indexing
        self.assertTrue(hasattr(Indexing, 'set'))
    
    def test_pattern_import(self):
        """Test that the Pattern dataclass can be imported."""
        from laueindexing.lau_dataclasses.pattern import Pattern
        self.assertTrue(hasattr(Pattern, 'getXMLElem'))
    
    def test_peaksxy_import(self):
        """Test that the PeaksXY dataclass can be imported."""
        from laueindexing.lau_dataclasses.peaksXY import PeaksXY
        self.assertTrue(hasattr(PeaksXY, 'addPeak'))
    
    def test_recip_lattice_import(self):
        """Test that the RecipLattice dataclass can be imported."""
        from laueindexing.lau_dataclasses.recipLattice import RecipLattice
        self.assertTrue(hasattr(RecipLattice, 'fromString'))
    
    def test_roi_import(self):
        """Test that the ROI dataclass can be imported."""
        from laueindexing.lau_dataclasses.roi import ROI
        self.assertTrue(hasattr(ROI, 'getXMLElem'))
    
    def test_step_import(self):
        """Test that the Step dataclass can be imported."""
        from laueindexing.lau_dataclasses.step import Step
        self.assertTrue(hasattr(Step, 'fromH5'))
    
    def test_xtl_import(self):
        """Test that the Xtl dataclass can be imported."""
        from laueindexing.lau_dataclasses.xtl import Xtl
        self.assertTrue(hasattr(Xtl, 'set'))

    def test_main_class_import(self):
        """Test that the main PyLaueGo class can be imported."""
        from laueindexing.pyLaueGo import PyLaueGo
        self.assertTrue(hasattr(PyLaueGo, 'index'))
        self.assertTrue(hasattr(PyLaueGo, 'p2q'))


if __name__ == '__main__':
    unittest.main()