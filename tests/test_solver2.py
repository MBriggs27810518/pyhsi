import sys, os

sys.path.insert(0, os.path.abspath("../pyhsi"))
import pytest
import numpy as np
from pyhsi import solver, beam, crowd
from pyhsi.crowd import *
from pyhsi.beam import *
from pyhsi.results import *
from pyhsi.solver import *

@pytest.fixture
def crowd():
    # Create and return an instance of the Crowd class
    return Crowd()

@pytest.fixture
def beam():
    # Create and return an instance of the Beam class
    return Beam()

def test_solver(beam, crowd):
    # Create instances of the Solver class
    crowd = Crowd()
    beam = Beam()
    solver = Solver(crowd, beam)

    # Test the calcnDOF method
    assert solver.calcnDOF() == 22

    # Test the genTimeVector

    # Test the assembleMCK

    # Test the constraints
