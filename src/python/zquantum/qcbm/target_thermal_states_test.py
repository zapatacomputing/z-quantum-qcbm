import unittest
import numpy as np
from zquantum.core.bitstring_distribution import BitstringDistribution

from target_thermal_states import (intToising, isingToint, get_random_hj, get_thermal_states_target_distribution)

class TestThermalTarget(unittest.TestCase):

    def setUp(self):
        self.h = [ 0.21876067, -0.69506657, -0.39636277, -0.32194209,  0.30247149]
        self.J = [[ 0, 0.2745898, 0.4551941, 0.71352672, -0.09870992], 
                    [ 0.2745898, 0, -0.07542956, 0.70699875, -0.05742523],
                    [0.4551941, -0.07542956, 0, 0.03228819, -0.10355254],
                    [ 0.71352672, 0.70699875, 0.03228819, 0, -0.34558086],
                    [-0.09870992, -0.05742523, -0.10355254, -0.34558086, 0]]
        self.num = 4 
        self.n_spins = 5 
        self.ising_vector = np.array([-1, -1, 1, -1, -1]) #expected ising_vector from (num, n_spins) = (4, 5)
        self.beta = 1
        self.n_body_interaction = 0

    def test_intToising(self): 
        self.assertEqual(intToising(self.num, self.n_spins).all(), self.ising_vector.all())
    
    def test_isingToint(self): 
        self.assertEqual(isingToint(self.ising_vector), self.num)

    def test_get_random_hj(self): 
        self.assertEqual(len(get_random_hj(self.n_spins)[0]), len(self.h) )
        self.assertEqual(len(get_random_hj(self.n_spins)[1]), len(self.J) )
    
    def test_get_thermal_states_target_distribution(self): 
        self.assertEqual(list(get_thermal_states_target_distribution(self.n_spins, self.beta, self.n_body_interaction).keys()), 
        (['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011', '01100', 
        '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000', '11001', 
        '11010', '11011', '11100', '11101', '11110', '11111'])) 

        self.assertEqual(sum(list(get_thermal_states_target_distribution(self.n_spins, self.beta, self.n_body_interaction).values())), 1.0 )


    
    


