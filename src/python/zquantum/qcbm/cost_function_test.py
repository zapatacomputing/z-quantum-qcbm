import unittest
from .cost_function import QCBMCostFunction
from .ansatz import get_qcbm_ansatz
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.utils import create_object
import numpy as np

class TestQCBMCostFunction(unittest.TestCase):

    def test_init(self):
        # Given
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        n_layers = 1
        distance_measure = "clipped_log_likelihood"
        ansatz = get_qcbm_ansatz(num_qubits, n_layers, topology)
        target_bitstring_distribution = BitstringDistribution(
            {"0000": 1.0, "0001": 0.0, "0010": 0.0,
             "0011": 1.0, "0100": 0.0, "0101": 1.0,
             "0110": 0.0, "0111": 0.0, "1000": 0.0,
             "1001": 0.0, "1010": 1.0, "1011": 0.0,
             "1100": 1.0, "1101": 0.0, "1110": 0.0,
             "1111": 1.0})
        backend = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})

        # When
        cost_function = QCBMCostFunction(ansatz, backend, distance_measure, target_bitstring_distribution, epsilon)

        # Then
        self.assertEqual(cost_function.ansatz, ansatz)
        self.assertEqual(cost_function.backend, backend)
        self.assertEqual(cost_function.distance_measure, distance_measure)
        self.assertEqual(cost_function.target_bitstring_distribution, target_bitstring_distribution)
        self.assertEqual(cost_function.epsilon, epsilon)
        self.assertEqual(cost_function.gradient_type, 'finite_difference')
        self.assertEqual(cost_function.save_evaluation_history, True)
        self.assertEqual(cost_function.evaluations_history, [])

    def test_evaluate(self):
        # Given
        num_qubits = 4
        topology = "all"
        epsilon = 1e-6
        n_layers = 1
        params = np.array([0, 0, 0, 0])
        distance_measure = "clipped_log_likelihood"
        ansatz = get_qcbm_ansatz(num_qubits, n_layers, topology)
        target_bitstring_distribution = BitstringDistribution(
            {"0000": 1.0, "0001": 0.0, "0010": 0.0,
             "0011": 1.0, "0100": 0.0, "0101": 1.0,
             "0110": 0.0, "0111": 0.0, "1000": 0.0,
             "1001": 0.0, "1010": 1.0, "1011": 0.0,
             "1100": 1.0, "1101": 0.0, "1110": 0.0,
             "1111": 1.0})
        backend = create_object({'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumSimulator', 'n_samples': 1})
        cost_function = QCBMCostFunction(ansatz, backend, distance_measure, target_bitstring_distribution, epsilon)

        # When
        value = cost_function.evaluate(params)
        history = cost_function.evaluations_history

        # Then
        self.assertEqual(type(value), float)
        self.assertIn("bitstring_distribution", history[0].keys())
        self.assertEqual(dict, type(history[0]["bitstring_distribution"]))
        self.assertIn("value", history[0].keys())
        self.assertIn("params", history[0].keys())



