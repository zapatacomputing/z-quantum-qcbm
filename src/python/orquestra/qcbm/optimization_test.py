import unittest
import random
import numpy as np
import json
import subprocess

from .ansatz import build_qcbm_circuit_ion_trap, generate_random_initial_params
from .optimization import optimize_variational_qcbm_circuit

from orquestra.core.bitstring_distribution import BitstringDistribution 
from orquestra.core.utils import ValueEstimate, RNDSEED
from orquestra.forest import ForestSimulator
from orquestra.optimizers import ScipyOptimizer


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
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"
        epsilon = 1e-6
        initial_params = generate_random_initial_params(num_qubits, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        simulator = ForestSimulator("wavefunction-simulator")
        optimizer = ScipyOptimizer(method="L-BFGS-B")

        opt_result = optimize_variational_qcbm_circuit(num_qubits,
            single_qubit_gate, static_entangler, topology, epsilon, initial_params,
            distance_measure, simulator, optimizer,
            self.target_distribution)

        expected_opt_result = {
            "fun": 1.7917594694188614,
            "nfev": 300,
            "nit": 16,
            "opt_value": 1.7917594694188614,
            "status": 0,
            "success": True}

        self.assertAlmostEqual(expected_opt_result["fun"], opt_result["fun"])
        self.assertAlmostEqual(expected_opt_result["nfev"], opt_result["nfev"])
        self.assertAlmostEqual(expected_opt_result["nit"], opt_result["nit"])
        self.assertAlmostEqual(expected_opt_result["opt_value"], opt_result["opt_value"])
        self.assertAlmostEqual(expected_opt_result["status"], opt_result["status"])
        self.assertAlmostEqual(expected_opt_result["success"], opt_result["success"])


    def test_qcbm_set_initial_params_scipy_forest_sampling(self):
        num_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"
        epsilon = 1e-6
        initial_params = generate_random_initial_params(num_qubits, seed=RNDSEED)
        distance_measure = "clipped_log_likelihood"

        simulator = ForestSimulator("wavefunction-simulator", n_samples=1000)
        optimizer = ScipyOptimizer(method="L-BFGS-B")

        opt_result = optimize_variational_qcbm_circuit(num_qubits,
            single_qubit_gate, static_entangler, topology, epsilon, initial_params,
            distance_measure, simulator, optimizer,
            self.target_distribution)

        self.assertIn("fun", opt_result.keys())
        self.assertIn("nfev", opt_result.keys())
        self.assertIn("nit", opt_result.keys())
        self.assertIn("opt_value", opt_result.keys())
        self.assertIn("status", opt_result.keys())
        self.assertIn("success", opt_result.keys())
        self.assertIn("opt_params", opt_result.keys())