import unittest
from GBMGrid.Simulator import Simulator
import numpy as np


class TestSetup(unittest.TestCase):
    def test_coulomb_refining(self):
        simulation = Simulator(name="Test", directory="/home/niklas/GGS/GBMGridSimulator", overwrite=True)
        simulation.setup_grid(2, 50, [-2, -1, 3], [100, 400, 3])
        particles_before = simulation.get_coords_from_gridpoints()
        particles_after = simulation.coulomb_refining(100)
        self.assertEqual(len(particles_before), len(particles_after))
        self.assertAlmostEqual(np.linalg.norm(particles_before[1]), 1)
        self.assertNotEqual(set(particles_before[1]), set(particles_after[1]))

    def test_gridpoint_insert_pointwise(self):
        simulation = Simulator(name="Test", directory="/home/niklas/GGS/GBMGridSimulator", overwrite=True)
        simulation.setup_pointwise(K=50)
        simulation.insert_gridpoint("test", [0, 1, 0])
        self.assertEqual(len(simulation.grid), 1)
        self.assertEqual(simulation.grid[0].name, "test")
        self.assertEqual(simulation.grid[0].coord, [0, 1, 0])

    def test_gridpoint_delete_pointwise(self):
        simulation = Simulator(name="Test", directory="/home/niklas/GGS/GBMGridSimulator", overwrite=True)
        simulation.setup_pointwise(K=50)
        simulation.insert_gridpoint("test1", [0, 1, 0])
        simulation.insert_gridpoint("test2", [0, 0, 1])
        simulation.delete_gridpoint("test1")
        self.assertEqual(len(simulation.grid), 1)
        self.assertEqual(simulation.grid[0].name, "test2")
        self.assertEqual(simulation.grid[0].coord, [0, 0, 1])

    def test_gridpoint_insert_fibonacci(self):
        simulation = Simulator(name="Test", directory="/home/niklas/GGS/GBMGridSimulator", overwrite=True)
        simulation.setup_grid(1, K=50)
        simulation.insert_gridpoint("test", [0, 1, 0])
        self.assertEqual(len(simulation.grid), 2)
        self.assertEqual(simulation.grid[0].name, "gp0")
        self.assertEqual(simulation.grid[1].name, "test")
        self.assertEqual(simulation.grid[1].coord, [0, 1, 0])

    def test_gridpoint_delete_fibonacci(self):
        simulation = Simulator(name="Test", directory="/home/niklas/GGS/GBMGridSimulator", overwrite=True)
        simulation.setup_grid(1, K=50)
        simulation.delete_gridpoint("gp0")
        self.assertEqual(len(simulation.grid), 0)


if __name__=="__main__":
    unittest.main()
