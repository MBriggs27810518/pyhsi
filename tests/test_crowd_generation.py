import sys, os
sys.path.insert(0, os.path.abspath("../pyhsi"))

from pyhsi import crowd
import unittest
from pyhsi.crowd import Pedestrian
from pyhsi.crowd import Crowd
from pyhsi.crowd import SinglePedestrian
from pyhsi.crowd import DeterministicCrowd
from pyhsi.crowd import RandomCrowd
from pyhsi.crowd import ExactCrowd

class TestPedestrian(unittest.TestCase):
    """
    Test class for the Pedestrian module.
    """

    def test_deterministicPedestrian(self):
        """
        Test creating a deterministic pedestrian instance and setting its properties.
        """
        Pedestrian.setPopulationProperties({
            'meanMass': 73.85,
            # 'sdMass': 15.68,
            'meanPace': 1.96,
            'sdPace': 0.209,
            # 'meanStride': 0.66,
            # 'sdStride': 0.066,
            # 'meanStiffness': 28000,
            # 'sdStiffness': 2800,
            'meanDamping': 0.3,
            'sdDamping': 0.03,
        })

        # creating the first instance
        p1 = Pedestrian.deterministicPedestrian(location=0, synched=1)
        self.assertEqual(p1.mass, 73.85)
        self.assertEqual(p1.damping, 612.477313865583)
        self.assertEqual(p1.stiff, 14110)
        self.assertEqual(p1.pace, 0)
        self.assertEqual(p1.phase, 0)
        self.assertEqual(p1.location, 0)
        self.assertEqual(p1.velocity, 1.25)
        self.assertEqual(p1.iSync, 1)

        p2 = Pedestrian.deterministicPedestrian(location=1, synched=0)
        self.assertEqual(p2.location, 1)
        self.assertEqual(p2.iSync, 0)

    def test_randomPedestrian(self):
        """
        Test creating a random pedestrian instance and setting its properties.
        """
        Pedestrian.setPopulationProperties({
            'meanMass': 73.85,
            'sdMass': 15.68,
            'meanPace': 1.96,
            'sdPace': 0.209,
            'meanStride': 0.66,
            'sdStride': 0.066,
            'meanStiffness': 28000,
            'sdStiffness': 2800,
            'meanDamping': 0.3,
            'sdDamping': 0.03,
        })

    def test_invalidLocation(self):
        """
        Test creating a pedestrian instance with invalid location parameters.
        """
        with self.assertRaises(KeyError):
            Pedestrian.randomPedestrian(location=1, synched=1)
        with self.assertRaises(ValueError):
            Pedestrian.randomPedestrian(location=0) # change....
        with self.assertRaises(NameError):
            Pedestrian.randomPedestrian(location=2) # change ... don't forget

    if __name__ == '__main__':
        unittest.main()

# test for deterministic crowd
# test for random crowd













