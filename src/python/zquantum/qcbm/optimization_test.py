import unittest
import random
import numpy as np
import json
import subprocess

from .ansatz import build_qcbm_circuit_ion_trap, generate_random_initial_params, get_qcbm_ansatz
from .optimization import optimize_variational_qcbm_circuit
from .cost_function import QCBMCostFunction

from zquantum.core.bitstring_distribution import BitstringDistribution 
from zquantum.core.utils import ValueEstimate, RNDSEED, create_object

class TestQCBM(unittest.TestCase):

    def setUp(self):
        self.target_distribution = BitstringDistribution(
            {"0000": 1.0, "0001": 0.0, "0010": 0.0,
             "0011": 1.0, "0100": 0.0, "0101": 1.0,
             "0110": 0.0, "0111": 0.0, "1000": 0.0,
             "1001": 0.0, "1010": 1.0, "1011": 0.0,
             "1100": 1.0, "1101": 0.0, "1110": 0.0,
             "1111": 1.0})


    def test_qcbm_set_initial_params(self):
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        n_layers = 1
        initial_params = generate_random_initial_params(num_qubits, n_layers=n_layers, topology=topology, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        ansatz = get_qcbm_ansatz(num_qubits, n_layers, topology)
        simulator = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})
        optimizer = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockOptimizer'})

        cost_function = QCBMCostFunction(ansatz, simulator, distance_measure, self.target_distribution, epsilon)
        opt_result = optimizer.minimize(cost_function, initial_params)

        self.assertIsInstance(opt_result["opt_value"], float)
        self.assertIn("history", opt_result.keys())

        for evaluation in opt_result['history']:
            self.assertIn("bitstring_distribution", evaluation.keys())
            self.assertIn("value", evaluation.keys())
            self.assertIn("params", evaluation.keys())


    def test_qcbm_set_initial_params_sampling(self):
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        n_layers = 1
        initial_params = generate_random_initial_params(num_qubits, n_layers=n_layers, topology=topology, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        simulator = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})
        optimizer = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockOptimizer'})

        ansatz = get_qcbm_ansatz(num_qubits, n_layers, topology)

        cost_function = QCBMCostFunction(ansatz, simulator, distance_measure, self.target_distribution, epsilon)
        opt_result = optimizer.minimize(cost_function, initial_params)

        self.assertIn("opt_value", opt_result.keys())
        self.assertIn("opt_params", opt_result.keys())
        self.assertIn("history", opt_result.keys())