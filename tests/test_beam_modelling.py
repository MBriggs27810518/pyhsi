import sys, os
sys.path.insert(0, os.path.abspath("../pyhsi"))

from pyhsi import beam
import unittest
from pyhsi.beam import Beam

bridge = beam.Beam()

class TestBeam(unittest.TestCase):
    def setUp(self):
        # Create a Beam object with default values
        self.beam = Beam()

    def test_elem_length(self):
        # Test if the element length is calculated correctly
        self.assertAlmostEqual(self.beam.elemLength, 5.0)

    def test_I(self):
        # Test if the second moment of area is calculated correctly
        self.assertAlmostEqual(self.beam.I, 0.036, places=2)

    def test_EI(self):
        # Test if the flexural rigidity is calculated correctly
        self.assertAlmostEqual(self.beam.EI, 5066059182.116886, places=2)

    def test_nDOF(self):
        # Test if the number of DOFs is calculated correctly
        self.assertEqual(self.beam.nDOF, 22)

    def test_nBDOF(self):
        # Test if the number of beam-only DOFs is calculated correctly
        self.assertEqual(self.beam.nBDOF, 22)

    def test_RDOF(self):
        # Test if the restrained DOFs are calculated correctly
        self.assertListEqual(self.beam.RDOF, [0, 20])

if __name__ == '__main__':
    unittest.main()
