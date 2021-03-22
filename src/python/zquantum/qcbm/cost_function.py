from zquantum.core.interfaces.functions import function_with_gradient, StoreArtifact
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    evaluate_distribution_distance,
)
from zquantum.core.utils import ValueEstimate
from zquantum.core.gradients import finite_differences_gradient
from typing import Union, Callable
import numpy as np


def QCBMCostFunction(
    ansatz: Ansatz,
    backend: QuantumBackend,
    distance_measure: Callable,
    distance_measure_parameters: dict,
    target_bitstring_distribution: BitstringDistribution,
    gradient_type: str = "finite_difference",
):
    """Cost function used for evaluating QCBM.

    Args:
        ansatz (zquantum.core.interfaces.ansatz.Ansatz): the ansatz used to construct the variational circuits
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for QCBM evaluation
        distance_measure (callable): function used to calculate the distance measure
        distance_measure_parameters (dict): dictionary containing the relevant parameters for the chosen distance measure
        target_bitstring_distribution (zquantum.core.bitstring_distribution.BitstringDistribution): bistring distribution which QCBM aims to learn
        gradient_type (str): parameter indicating which type of gradient should be used.

    Returns:
        Callable that evaluates the parametrized circuit produced by the ansatz with the given parameters and returns
            the distance between the produced bitstring distribution and the target distribution
    """

    assert (
        int(target_bitstring_distribution.get_qubits_number())
        == ansatz.number_of_qubits
    )

    def cost_function(
        parameters: np.ndarray, store_artifact: StoreArtifact = None
    ) -> ValueEstimate:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            (float): cost function value for given parameters
            zquantum.core.bitstring_distribution.BitstringDistribution: distribution obtained
        """
        circuit = ansatz.get_executable_circuit(parameters)
        distribution = backend.get_bitstring_distribution(circuit)
        value = evaluate_distribution_distance(
            target_bitstring_distribution,
            distribution,
            distance_measure,
            distance_measure_parameters=distance_measure_parameters,
        )

        if store_artifact:
            store_artifact("bitstring_distribution", distribution)

        return ValueEstimate(value)

    if gradient_type == "finite_difference":
        cost_function = function_with_gradient(
            cost_function, finite_differences_gradient(cost_function)
        )
    else:
        raise RuntimeError("Unsupported gradient type: ", gradient_type)

    return cost_function
