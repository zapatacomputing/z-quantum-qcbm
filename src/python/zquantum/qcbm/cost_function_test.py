import unittest
from .cost_function import QCBMCostFunction
from .ansatz import QCBMAnsatz
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    compute_clipped_negative_log_likelihood,
)
from zquantum.core.utils import create_object, ValueEstimate
import numpy as np


class TestQCBMCostFunction(unittest.TestCase):
    def setUp(self):
        number_of_layers = 1
        number_of_qubits = 4
        topology = "all"
        self.ansatz = QCBMAnsatz(number_of_layers, number_of_qubits, topology)

        self.distance_measure = compute_clipped_negative_log_likelihood
        self.target_bitstring_distribution = BitstringDistribution(
            {
                "0000": 1.0,
                "0001": 0.0,
                "0010": 0.0,
                "0011": 1.0,
                "0100": 0.0,
                "0101": 1.0,
                "0110": 0.0,
                "0111": 0.0,
                "1000": 0.0,
                "1001": 0.0,
                "1010": 1.0,
                "1011": 0.0,
                "1100": 1.0,
                "1101": 0.0,
                "1110": 0.0,
                "1111": 1.0,
            }
        )

        self.backend = create_object(
            {
                "module_name": "zquantum.core.interfaces.mock_objects",
                "function_name": "MockQuantumSimulator",
                "n_samples": 1,
            }
        )

    def test_init(self):
        # Given
        epsilon = 1e-6

        # When
        cost_function = QCBMCostFunction(
            self.ansatz,
            self.backend,
            self.distance_measure,
            self.target_bitstring_distribution,
            epsilon,
        )

        # Then
        self.assertEqual(cost_function.ansatz, self.ansatz)
        self.assertEqual(cost_function.backend, self.backend)
        self.assertEqual(cost_function.distance_measure, self.distance_measure)
        self.assertEqual(
            cost_function.target_bitstring_distribution,
            self.target_bitstring_distribution,
        )
        self.assertEqual(cost_function.epsilon, epsilon)
        self.assertEqual(cost_function.gradient_type, "finite_difference")
        self.assertEqual(cost_function.save_evaluation_history, True)
        self.assertEqual(cost_function.evaluations_history, [])

    def test_evaluate(self):
        # Given
        epsilon = 1e-6

        cost_function = QCBMCostFunction(
            self.ansatz,
            self.backend,
            self.distance_measure,
            self.target_bitstring_distribution,
            epsilon,
        )

        params = np.array([0, 0, 0, 0])

        # When
        value_estimate = cost_function.evaluate(params)
        history = cost_function.evaluations_history

        # Then
        self.assertEqual(type(value_estimate), ValueEstimate)
        self.assertEqual(type(value_estimate.value), float)
        self.assertIn("bitstring_distribution", history[0].keys())
        self.assertEqual(dict, type(history[0]["bitstring_distribution"]))
        self.assertIn("value", history[0].keys())
        self.assertIn("params", history[0].keys())

