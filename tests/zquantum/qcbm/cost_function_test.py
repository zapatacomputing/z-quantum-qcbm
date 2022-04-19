################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    compute_clipped_negative_log_likelihood,
    compute_mmd,
)
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.history.recorder import recorder
from zquantum.core.utils import ValueEstimate, create_object
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.cost_function import QCBMCostFunction, create_QCBM_cost_function

number_of_layers = 1
number_of_qubits = 4
topology = "all"
ansatz = QCBMAnsatz(number_of_layers, number_of_qubits, topology)
target_bitstring_distribution = BitstringDistribution(
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

backend = create_object(
    {
        "module_name": "zquantum.core.symbolic_simulator",
        "function_name": "SymbolicSimulator",
    }
)

n_samples = 1


def test_QCBMCostFunction_raises_deprecation_warning():
    with pytest.deprecated_call():
        QCBMCostFunction(
            ansatz,
            backend,
            n_samples,
            distance_measure=compute_clipped_negative_log_likelihood,
            distance_measure_parameters={"epsilon": 1e-6},
            target_bitstring_distribution=target_bitstring_distribution,
        )


class TestQCBMCostFunction:
    @pytest.fixture(params=[QCBMCostFunction, create_QCBM_cost_function])
    def cost_function_factory(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {
                "distance_measure": compute_clipped_negative_log_likelihood,
                "distance_measure_parameters": {"epsilon": 1e-6},
            },
            {
                "distance_measure": compute_mmd,
                "distance_measure_parameters": {"sigma": 1},
            },
        ]
    )
    def distance_measure_kwargs(self, request):
        return request.param

    def test_evaluate_history(self, cost_function_factory, distance_measure_kwargs):
        # Given
        cost_function = cost_function_factory(
            ansatz,
            backend,
            n_samples,
            **distance_measure_kwargs,
            target_bitstring_distribution=target_bitstring_distribution,
        )

        cost_function = recorder(cost_function)

        params = np.array([0, 0, 0, 0])

        # When
        value_estimate = cost_function(params)
        history = cost_function.history

        # Then
        assert len(history) == 1
        np.testing.assert_array_equal(params, history[0].params)
        assert value_estimate == history[0].value

    @pytest.mark.parametrize(
        "gradient_kwargs, cf_factory",
        [
            ({"gradient_type": "finite_difference"}, QCBMCostFunction),
            (
                {"gradient_function": finite_differences_gradient},
                create_QCBM_cost_function,
            ),
        ],
    )
    def test_gradient(self, gradient_kwargs, cf_factory, distance_measure_kwargs):
        # Given
        cost_function = cf_factory(
            ansatz,
            backend,
            n_samples,
            **distance_measure_kwargs,
            target_bitstring_distribution=target_bitstring_distribution,
            **gradient_kwargs,
        )

        params = np.array([0, 0, 0, 0])

        # When
        gradient = cost_function.gradient(params)

        # Then
        assert len(params) == len(gradient)

    def test_error_raised_if_gradient_is_not_supported(self, distance_measure_kwargs):
        # Given
        gradient_type = "UNSUPPORTED GRADIENT TYPE"

        # When/then
        with pytest.raises(RuntimeError):
            cost_function = (
                QCBMCostFunction(
                    ansatz,
                    backend,
                    n_samples,
                    **distance_measure_kwargs,
                    target_bitstring_distribution=target_bitstring_distribution,
                    gradient_type=gradient_type,
                ),
            )
            params = np.array([0, 0, 0, 0])
            cost_function(params)

    def test_error_raised_if_target_distribution_and_ansatz_are_for_differing_number_of_qubits(  # noqa E501
        self, cost_function_factory, distance_measure_kwargs
    ):
        # Given
        ansatz.number_of_qubits = 5

        # When/Then
        with pytest.raises(AssertionError):
            cost_function = (
                cost_function_factory(
                    ansatz,
                    backend,
                    n_samples,
                    **distance_measure_kwargs,
                    target_bitstring_distribution=target_bitstring_distribution,
                ),
            )
            params = np.array([0, 0, 0, 0])
            cost_function(params)
