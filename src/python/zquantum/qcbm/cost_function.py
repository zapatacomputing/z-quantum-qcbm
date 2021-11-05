import warnings
from numbers import Number
from typing import Callable

import numpy as np
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    evaluate_distribution_distance,
)
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.functions import StoreArtifact, function_with_gradient
from zquantum.core.utils import ValueEstimate

GradientFactory = Callable[[Callable], Callable[[np.ndarray], np.ndarray]]
DistanceMeasure = Callable[..., Number]


def create_QCBM_cost_function(
    ansatz: Ansatz,
    backend: QuantumBackend,
    n_samples: int,
    distance_measure: DistanceMeasure,
    distance_measure_parameters: dict,
    target_bitstring_distribution: BitstringDistribution,
    gradient_function: GradientFactory = finite_differences_gradient,
) -> CostFunction:
    """Cost function used for evaluating QCBM.

    Args:
        ansatz: the ansatz used to construct the variational circuits
        backend: backend used for QCBM evaluation
        distance_measure: function used to calculate the distance measure
        distance_measure_parameters: dictionary containing the relevant parameters
            for the chosen distance measure
        target_bitstring_distribution: bistring distribution which QCBM aims to learn
        gradient_function: a function which returns a function used to compute
            the gradient of the cost function
            (see zquantum.core.gradients.finite_differences_gradient for reference)

    Returns:
        Callable CostFunction object that evaluates the parametrized circuit produced
        by the ansatz with the given parameters and returns the distance between
        the produced bitstring distribution and the target distribution
    """

    cost_function = _create_QCBM_cost_function(
        ansatz,
        backend,
        n_samples,
        distance_measure,
        distance_measure_parameters,
        target_bitstring_distribution,
    )

    return function_with_gradient(cost_function, gradient_function(cost_function))


def QCBMCostFunction(
    ansatz: Ansatz,
    backend: QuantumBackend,
    n_samples: int,
    distance_measure: DistanceMeasure,
    distance_measure_parameters: dict,
    target_bitstring_distribution: BitstringDistribution,
    gradient_type: str = "finite_difference",
    gradient_kwargs: dict = None,
) -> CostFunction:
    """Cost function used for evaluating QCBM.

    Args:
        ansatz: the ansatz used to construct the variational circuits
        backend: backend used for QCBM evaluation
        distance_measure: function used to calculate the distance measure
        distance_measure_parameters: dictionary containing the relevant parameters
            for the chosen distance measure
        target_bitstring_distribution: bistring distribution which QCBM aims to learn
        gradient_type: parameter indicating which type of gradient should be used.

    Returns:
        Callable CostFunction object that evaluates the parametrized circuit produced
            by the ansatz with the given
        parameters and returns the distance between the produced bitstring distribution
            and the target distribution
    """

    warnings.warn(
        "QCBMCostFunction is deprecated in favour of create_QCBM_cost_function.",
        DeprecationWarning,
    )

    cost_function = _create_QCBM_cost_function(
        ansatz,
        backend,
        n_samples,
        distance_measure,
        distance_measure_parameters,
        target_bitstring_distribution,
    )

    if gradient_kwargs is None:
        gradient_kwargs = {}

    if gradient_type == "finite_difference":
        cost_function = function_with_gradient(
            cost_function,
            finite_differences_gradient(cost_function, **gradient_kwargs),
        )
    else:
        raise RuntimeError("Unsupported gradient type: ", gradient_type)

    return cost_function


def _create_QCBM_cost_function(
    ansatz: Ansatz,
    backend: QuantumBackend,
    n_samples: int,
    distance_measure: DistanceMeasure,
    distance_measure_parameters: dict,
    target_bitstring_distribution: BitstringDistribution,
):
    assert (
        int(target_bitstring_distribution.get_number_of_subsystems())
        == ansatz.number_of_qubits
    )

    def cost_function(
        parameters: np.ndarray, store_artifact: StoreArtifact = None
    ) -> ValueEstimate:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.
            store_artifact: callable defining how the bitstring distributions
                should be stored.
        """
        # TODO: we use private method here due to performance reasons.
        # This should be fixed once better mechanism for handling
        # it will be implemented.
        # In case of questions ask mstechly.
        # circuit = ansatz.get_executable_circuit(parameters)
        circuit = ansatz._generate_circuit(parameters)
        distribution = backend.get_bitstring_distribution(circuit, n_samples)
        value = evaluate_distribution_distance(
            target_bitstring_distribution,
            distribution,
            distance_measure,
            distance_measure_parameters=distance_measure_parameters,
        )

        if store_artifact:
            store_artifact("bitstring_distribution", distribution)

        return ValueEstimate(value)

    return cost_function
