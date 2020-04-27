import unittest
import random
import numpy as np
import json
import subprocess

from .ansatz import build_qcbm_circuit_ion_trap, generate_random_initial_params
from .optimization import optimize_variational_qcbm_circuit

from zquantum.core.bitstring_distribution import BitstringDistribution 
from zquantum.core.utils import ValueEstimate, RNDSEED, create_object
from zquantum.optimizers import ScipyOptimizer


class TestQCBM(unittest.TestCase):

    def setUp(self):
        self.target_distribution = BitstringDistribution(
            {"0000": 1.0, "0001": 0.0, "0010": 0.0,
             "0011": 1.0, "0100": 0.0, "0101": 1.0,
             "0110": 0.0, "0111": 0.0, "1000": 0.0,
             "1001": 0.0, "1010": 1.0, "1011": 0.0,
             "1100": 1.0, "1101": 0.0, "1110": 0.0,
             "1111": 1.0})


    def test_qcbm_set_initial_params_scipy_forest(self):
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        initial_params = generate_random_initial_params(num_qubits, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        simulator = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})
        optimizer = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockOptimizer'})

        opt_result = optimize_variational_qcbm_circuit(num_qubits,
            topology, epsilon, initial_params,
            distance_measure, simulator, optimizer,
            self.target_distribution)

        self.assertIsInstance(opt_result["opt_value"], float)
        self.assertIn("history", opt_result.keys())

        for evaluation in opt_result['history']:
            self.assertIn("bitstring_distribution", evaluation.keys())
            self.assertIn("value", evaluation.keys())
            self.assertIn("params", evaluation.keys())


    def test_qcbm_set_initial_params_scipy_forest_sampling(self):
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        initial_params = generate_random_initial_params(num_qubits, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        simulator = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})
        optimizer = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockOptimizer'})

        opt_result = optimize_variational_qcbm_circuit(num_qubits,
            topology, epsilon, initial_params,
            distance_measure, simulator, optimizer,
            self.target_distribution)

        self.assertIn("opt_value", opt_result.keys())
        self.assertIn("opt_params", opt_result.keys())
        self.assertIn("history", opt_result.keys())