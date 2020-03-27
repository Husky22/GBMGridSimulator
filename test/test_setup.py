import unittest
from GBMGrid.Simulator import Simulator
import numpy as np

class TestSetup(unittest.TestCase):
    def test_coulomb_refining(self):
        simulation = Simulator(name = "Test", directory = "/home/niklas/.virtualenvs/GBM/GBMGridSimulator")
        simulation.setup(2, 50, [-2,-1,3], [100,400, 3],overwrite=True)
        particles_before = simulation.get_coords_from_gridpoints()
        particles_after = simulation.coulomb_refining(100)
        self.assertEqual(len(particles_before), len(particles_after))
        self.assertAlmostEqual(np.linalg.norm(particles_before[1]),1)
        self.assertNotEqual(set(particles_before[1]), set(particles_after[1]))


if __name__=="__main__":
    unittest.main()
