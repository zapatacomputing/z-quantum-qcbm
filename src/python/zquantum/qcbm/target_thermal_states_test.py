import unittest
import numpy as np
from zquantum.core.bitstring_distribution import BitstringDistribution

from target_thermal_states import (intToising, isingToint, get_random_hj, thermal_target_distribution, cumulate, sample, thermal_sample_distribution)
from target_thermal_states import get_target_Bitstring_Dist, get_sample_Bitstring_Dist
from zquantum.core.bitstring_distribution.distance_measures.mmd import compute_mmd

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
        self.seed_dist = 1
        self.ising_vector = np.array([-1, -1, 1, -1, -1]) #expected ising_vector from (num, n_spins) = (4, 5)
        self.beta = 1
        self.n_samples = 1000
        self.seed_sample = 1
        self.Precision = 8


    def test_intToising(self): 
        
        self.assertEqual(intToising(self.num, self.n_spins).all(), self.ising_vector.all())
    
    def test_isingToint(self): 
        
        self.assertEqual(isingToint(self.ising_vector), self.num)

    def test_get_random_hj(self): 
       
        self.assertEqual(len(get_random_hj(self.n_spins, self.seed_dist)[0]), len(self.h) )
        self.assertEqual(len(get_random_hj(self.n_spins, self.seed_dist)[1]), len(self.J) )
    
    def test_thermal_target_distribution(self): 
        
        self.assertEqual(list(thermal_target_distribution(self.n_spins, self.beta, self.seed_dist).keys()), 
        (['00000', '10000', '01000', '11000', '00100', '10100', '01100', '11100', '00010', '10010', '01010', '11010', '00110', '10110', '01110', '11110', '00001', '10001', '01001', '11001', 
        '00101', '10101', '01101', '11101', '00011', '10011', '01011', '11011', '00111', '10111', '01111', '11111'])) 
        
        self.assertEqual(list(thermal_target_distribution(self.n_spins, self.beta, self.seed_dist).values()), 
        ([3.866026604368924e-10, 4.3582393690650837e-13, 2.0290500815391508e-07, 7.098689073007999e-09, 0.0019959711650773073, 2.250093206373007e-06, 2.1912908960303108e-06, 7.666293149172884e-08, 0.001811405641563651, 2.0420292634389063e-06, 
        0.9507003290048913, 0.03326051978023547, 0.001560539213832431, 1.7592231514962936e-06, 1.7132488845533294e-06, 5.9938496574202e-08, 1.7549080710281928e-14, 
        1.978338563962551e-17, 9.210480757140888e-12, 3.22231273159459e-13, 9.060325498543598e-08, 1.0213863308564234e-10, 9.946941583558809e-11, 3.479965633746008e-12, 1.9558754181248016e-05, 
        2.2048925474607784e-08, 0.010265240213664702, 0.00035913233093422725, 1.685000983390945e-05, 1.89953108130846e-08, 1.8498901082890595e-08, 6.471892841607888e-10])) 
        
        self.assertEqual(round(sum(list(thermal_target_distribution(self.n_spins, self.beta, self.seed_dist).values())), self.Precision), 1)
        

    def test_cumulate(self): 
        self.assertEqual(list(cumulate(self.n_spins, self.beta, self.seed_dist)), 
        ([3.866026604368924e-10, 3.870384843737989e-10, 2.0329204663828888e-07, 2.1039073571129688e-07, 0.0019961815558130185, 
        0.0019984316490193913, 0.0020006229399154216, 0.0020006996028469133, 0.0038121052444105644, 0.003814147273674003, 0.9545144762785653, 
        0.9877749960588008, 0.9893355352726333, 0.9893372944957848, 0.9893390077446693, 0.9893390676831659, 0.9893390676831835, 0.9893390676831835, 
        0.989339067692394, 0.9893390676927162, 0.9893391582959712, 0.9893391583981098, 0.9893391584975793, 0.9893391585010592, 0.9893587172552405, 0.989358739304166, 0.9996239795178307, 
        0.9999831118487649, 0.9999999618585987, 0.9999999808539095, 0.9999999993528106, 0.9999999999999999]))

    def test_sample(self): 
        self.assertEqual(len(sample(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample)), self.n_samples)

        self.assertEqual(len(sample(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample)[0]), self.n_spins)

    def test_thermal_sample_distribution(self): 
        self.assertEqual(list(thermal_sample_distribution(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample).keys()), 
        (['00000', '10000', '01000', '11000', '00100', '10100', '01100', '11100', '00010', '10010', '01010', '11010', '00110', '10110', '01110', 
        '11110', '00001', '10001', '01001', '11001', '00101', '10101', '01101', '11101', '00011', '10011', '01011', '11011', '00111', '10111', '01111', '11111'])) 
        
        self.assertEqual(list(thermal_sample_distribution(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample).values()), 
        ([0, 0, 0, 0, 0.002, 0, 0, 0, 0.004, 0, 0.9530000000000007, 0.02900000000000002,
         0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0.009000000000000001, 0, 0, 0, 0, 0])) 
        
        self.assertEqual(round(sum(list(thermal_sample_distribution(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample).values())), self.Precision), 1)

    def test_get_target_Bitstring_Dist(self): 
        self.assertTrue(type(get_target_Bitstring_Dist(self.n_spins, self.beta, self.seed_dist) == BitstringDistribution))

    def get_sample_Bitstring_Dist(self): 
        self.assertTrue(type(get_sample_Bitstring_Dist(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample) == BitstringDistribution))

    def test_samples_from_dist(self): 
        actual = get_target_Bitstring_Dist(self.n_spins, self.beta, self.seed_dist)
        model = get_sample_Bitstring_Dist(self.n_samples, self.n_spins,self.beta, self.seed_dist, self.seed_sample)
        distance_measure = {'epsilon': 1e-6}
        mmd = compute_mmd(actual, model, distance_measure)
        self.assertLess(mmd, 1e-4)

if __name__ == "__main__":
    unittest.main()



    
    




        
