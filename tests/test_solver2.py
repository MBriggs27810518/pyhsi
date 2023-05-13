import sys, os
sys.path.insert(0, os.path.abspath("../pyhsi"))

from pyhsi import solver
import unittest
from pyhsi.solver import Solver
from pyhsi.beam import Beam
from pyhsi.crowd import Crowd, Pedestrian

class TestSolver(unittest.TestCase):

    def setUp(self):
        # Set up a crowd and a beam for testing
        self.crowd = Crowd(100, 50, 1, 0)
        self.beam = Beam(10, 50, 2, 0.6, 200e9, 0.005, 3, 0.3162, 500)

    def test_calcnDOF(self):
        # Test the calcnDOF function of the Solver class
        solver = Solver(self.crowd, self.beam)
        solver.PedestrianModel = "Spring Mass Damper"
        solver.nBDOF = 10
        solver.crowd = self.crowd
        nDOF = solver.calcnDOF()
        self.assertEqual(nDOF, 21)

    def test_genTimeVector(self):
        # Test the genTimeVector function of the Solver class
        solver = Solver(self.crowd, self.beam)
        solver.beam = self.beam
        solver.numSteps = 5000
        t, dT = solver.genTimeVector(10)
        self.assertEqual(len(t), 5000)
        self.assertAlmostEqual(dT, 0.001651, places=6)

    def test_assembleMCK(self):
        # Test the assembleMCK function of the Solver class
        solver = Solver(self.crowd, self.beam)
        solver.beam = self.beam
        solver.nBDOF = self.beam.nBDOF
        solver.constraints = lambda M, C, K: (M, C, K)  # Disable constraints for testing
        M, C, K = solver.assembleMCK()
        self.assertEqual(M.shape, (8, 8))
        self.assertEqual(C.shape, (8, 8))
        self.assertEqual(K.shape, (8, 8))

if __name__ == '__main__':
    unittest.main()

