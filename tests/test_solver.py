import sys, os
sys.path.insert(0, os.path.abspath("../pyhsi"))

from pyhsi import solver
import pytest
import numpy as np
from pyhsi.solver import Solver
from pyhsi.beam import Beam
from pyhsi.crowd import Crowd, Pedestrian

@pytest.fixture
def beam():
    # create a beam object
    beam = Beam(10, 50, 2, 0.6, 200e9, 0.005, 3, 0.3162, 500)
    return beam

@pytest.fixture
def pedestrian():
    # generate pedestrian
    pedestrian = Pedestrian(50, 0.3, 14.11e3, 2, 0, 0, 1.25, 0)
    return pedestrian

@pytest.fixture
def crowd(num_pedestrians=100):
    # create a group of pedestrians
    crowd = Crowd(70, 10, 1.2, 0.1)
    return crowd


@pytest.fixture
def solver(beam, crowd):
    # create a solver object
    solver = Solver(crowd, beam)
    solver.init()
    return solver

@pytest.fixture
def test_solver_initialization(beam, crowd):
    # create a solver object
    solver = Solver(crowd, beam)

    # check if the crowd and beam objects are correctly assigned
    assert solver.crowd == crowd
    assert solver.beam == beam

    # test if Solver raises a ZeroDivisionError when numstep is 0
    t, dT, dt_error = solver.genTimeVector()
    assert dt_error == True
    with pytest.raises(ZeroDivisionError):
        Solver(crowd, beam)

def test_calcnDOF(solver):
    # test the calcnDOF function
    assert solver.calcnDOF() == 4


def test_genTimeVector(solver):
    # test the genTimeVector function
    t, dT = solver.genTimeVector()
    assert len(t) == solver.numSteps
    assert dT > 0


def test_assembleMCK(solver):
    # test the assembleMCK function
    M, C, K = solver.assembleMCK()
    assert M.shape == (8, 8)
    assert C.shape == (8, 8)
    assert K.shape == (8, 8)


def test_solve(solver):
    # test the solve function
    results = solver.solve()
    #assert isinstance(results, Results)
    assert len(results.displacement) == solver.numSteps
    assert len(results.velocity) == solver.numSteps
    assert len(results.acceleration) == solver.numSteps
    assert len(results.force) == solver.numSteps
    assert isinstance(results.time, np.ndarray)







