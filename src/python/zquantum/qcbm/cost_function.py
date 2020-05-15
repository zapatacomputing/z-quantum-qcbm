from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.bitstring_distribution import BitstringDistribution, evaluate_distribution_distance
from zquantum.core.circuit import build_ansatz_circuit
from typing import Union, Dict
import numpy as np

class QCBMCostFunction(CostFunction):
    """
    Cost function used for evaluating QCBM.

    Args:
        ansatz (dict): dictionary representing the ansatz
        backend (zquantu.core.interfaces.backend.QuantumBackend): backend used for QCBM evaluation
        distance_measure (str): string describing which distance measure should be used. See zquantum.core.bitstring_distribution.evaluate_distribution_distance
        target_bitstring_distribution (zquantum.core.bitstring_distribution.BitstringDistribution): bistring distribution which QCBM aims to learn
        epsilon (float): clipping value used for calculating distribution distance
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        use_analytical_gradient (bool): flag indicating whether we want to use analytical or numerical gradient.

    Params:
        ansatz (dict): see Args
        backend (zquantu.core.interfaces.backend.QuantumBackend): see Args
        distance_measure (str): see Args
        target_bitstring_distribution (zquantum.core.bitstring_distribution.BitstringDistribution): see Args
        epsilon (float): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        use_analytical_gradient (bool): see Args
    """

    def __init__(self,  ansatz:Dict, 
                        backend:QuantumBackend, 
                        distance_measure:str,
                        target_bitstring_distribution:BitstringDistribution,
                        epsilon:float,
                        save_evaluation_history:bool=True, 
                        use_analytical_gradient:bool=False):
        self.ansatz = ansatz
        self.backend = backend
        self.distance_measure = distance_measure
        self.target_bitstring_distribution = target_bitstring_distribution
        self.epsilon = epsilon
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.use_analytical_gradient = use_analytical_gradient

    def evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters and saves the results (if specified).

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value, distribution = self._evaluate(parameters)
        if self.save_evaluation_history:
            self.evaluations_history.append({'value':value, 'params': parameters, 'bitstring_distribution': distribution})
        return value
        

    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            (float): cost function value for given parameters
            zquantum.core.bitstring_distribution.BitstringDistribution: distribution obtained
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        distribution = self.backend.get_bitstring_distribution(circuit)
        value = evaluate_distribution_distance(
                                            self.target_bitstring_distribution,
                                            distribution,
                                            self.distance_measure,
                                            epsilon=self.epsilon)

        return value, distribution

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.use_analytical_gradient:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_numerical_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        raise NotImplementedError