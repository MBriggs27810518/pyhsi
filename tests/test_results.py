import pytest
import numpy as np
import pandas as pd

from pyhsi.results import Results

# Test that the midpspanAcceleration property returns the correct results
def test_mnidspan_acceleration():
    t = np.arange(0, 1, 0.1)
    displacement =np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    assert np.array_equal(r.midspanAcceleration, [2, 5, 8])

# test that the calculatedRMS method returns the correct results
def test_calculate_rms():
    t = np.arange(0, 1, 0.1)
    displacement = np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    assert np.array_equal(r.midspanAcceleration, [2, 5, 8])

# Test that the calculatedRMS method returns the correct results
def test_calculate_rms():
    t = np.arange(0, 1, 0.1)
    displacement = np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    expected_rms = np.array([2.16024690, 5.31507291, 8.45765474])
    assert np.allclose(r.calculatedRMS(), expected_rms)

# Test that the maxDisplacement method returns the correct results
def test_max_displacement():
    t = np.arange(0, 1, 0.1)
    displacement = np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    assert r.maxDisplacement == 9

# Test that the minDisplacement method returns the correct results
def test_min_displacement():
    t = np.arange(0, 1, 0.1)
    displacement = np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    assert r.minDisplacement == 0

# Test that the minAcceleration method returns the correct results
def test_min_acceleration():
    t = np.arange(0, 1, 0.1)
    displacement = np.arange(0, 10, 1)
    acceleration = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = Results(t, displacement, None, acceleration)
    assert r.minAcceleration == 1