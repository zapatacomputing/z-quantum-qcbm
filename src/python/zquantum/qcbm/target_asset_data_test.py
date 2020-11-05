import unittest
import numpy as np
from scipy.special import comb
from zquantum.core.bitstring_distribution import BitstringDistribution

from target_asset_data import (download_sp500, generate_initial_dataset, objective_function, cost, probability, get_probabilities)

class TestAssetTarget(unittest.TestCase):

    def setUp(self):
        self.start_date = "2017-12-01"
        self.end_date = "2018-02-07"
        self.nbits = 4
        self.M = 2
        self.target_return = 2.5*10**-3

    
    def test_get_probabilities(self): 
        
        self.assertEqual(list(get_probabilities(self.start_date, self.end_date, self.nbits, self.target_return, self.M).keys()), 
        (['0000', '1000', '0100', '1100', '0010', '1010', '0110', '1110', '0001', '1001', '0101', '1101', '0011', '1011', '0111', '1111'])) 
        self.assertEqual(sum(list(get_probabilities(self.start_date, self.end_date, self.nbits, self.target_return, self.M).values())), 1.0 )

if __name__ == "__main__":
   unittest.main()
        
        
       