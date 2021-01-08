import unittest
from .cost_function import QCBMCostFunction
from .ansatz import QCBMAnsatz
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    compute_clipped_negative_log_likelihood,
    compute_mmd,
)
from zquantum.core.utils import create_object, ValueEstimate
from zquantum.core.history.recorder import recorder
import numpy as np

param_list = [
    (compute_clipped_negative_log_likelihood, {"epsilon": 1e-6}),
    (compute_mmd, {"sigma": 1}),
]


class TestQCBMCostFunction(unittest.TestCase):
    def setUp(self):
        number_of_layers = 1
        number_of_qubits = 4
        topology = "all"
        self.ansatz = QCBMAnsatz(number_of_layers, number_of_qubits, topology)
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

        self.gradient_types = ["finite_difference"]

    def test_evaluate(self):
        for distance_meas, distance_measure_params in param_list:
            with self.subTest():
                # Given
                self.distance_measure = distance_meas
                distance_measure_parameters = distance_measure_params

                cost_function = QCBMCostFunction(
                    self.ansatz,
                    self.backend,
                    self.distance_measure,
                    distance_measure_parameters,
                    self.target_bitstring_distribution,
                )

                params = np.array([0, 0, 0, 0])

                # When
                value_estimate = cost_function(params)

                # Then
                self.assertEqual(type(value_estimate), ValueEstimate)
                assert isinstance(value_estimate.value, (np.floating, float))

    def test_evaluate_history(self):
        for distance_meas, distance_measure_params in param_list:
            with self.subTest():
                # Given
                self.distance_measure = distance_meas
                distance_measure_parameters = distance_measure_params

                cost_function = QCBMCostFunction(
                    self.ansatz,
                    self.backend,
                    self.distance_measure,
                    distance_measure_parameters,
                    self.target_bitstring_distribution,
                )

                cost_function = recorder(cost_function)

                params = np.array([0, 0, 0, 0])

                # When
                value_estimate = cost_function(params)
                history = cost_function.history

                # Then
                self.assertEqual(len(history), 1)
                self.assertEqual(
                    BitstringDistribution,
                    type(history[0].artifacts["bitstring_distribution"]),
                )
                np.testing.assert_array_equal(params, history[0].params)
                self.assertEqual(value_estimate, history[0].value)

    def test_gradient(self):
        for distance_meas, distance_measure_params in param_list:
            for gradient_type in self.gradient_types:
                with self.subTest():
                    # Given
                    self.distance_measure = distance_meas
                    distance_measure_parameters = distance_measure_params

                    cost_function = QCBMCostFunction(
                        self.ansatz,
                        self.backend,
                        self.distance_measure,
                        distance_measure_parameters,
                        self.target_bitstring_distribution,
                        gradient_type=gradient_type,
                    )

                    params = np.array([0, 0, 0, 0])

                    # When
                    gradient = cost_function.gradient(params)

                    # Then
                    self.assertEqual(len(params), len(gradient))
                    for gradient_val in gradient:
                        self.assertEqual(np.float64, type(gradient_val))

    def test_error_raised_if_gradient_is_not_supported(self):
        for distance_meas, distance_measure_params in param_list:
            for gradient_type in self.gradient_types:
                with self.subTest():
                    # Given
                    self.distance_measure = distance_meas
                    distance_measure_parameters = distance_measure_params

                    self.assertRaises(
                        RuntimeError,
                        lambda: QCBMCostFunction(
                            self.ansatz,
                            self.backend,
                            self.distance_measure,
                            distance_measure_parameters,
                            self.target_bitstring_distribution,
                            gradient_type="UNSUPPORTED GRADIENT TYPE",
                        ),
                    )

    def test_error_raised_if_target_distribution_and_ansatz_are_for_differing_number_of_qubits(
        self,
    ):
        for distance_meas, distance_measure_params in param_list:
            with self.subTest():
                # Given
                self.ansatz.number_of_qubits = 5

                self.distance_measure = distance_meas
                distance_measure_parameters = distance_measure_params

                # When/Then
                self.assertRaises(
                    AssertionError,
                    lambda: QCBMCostFunction(
                        self.ansatz,
                        self.backend,
                        self.distance_measure,
                        distance_measure_parameters,
                        self.target_bitstring_distribution,
                    ),
                )