import copy
from zquantum.core.bitstring_distribution import evaluate_distribution_distance
from zquantum.core.utils import ValueEstimate
from .ansatz import build_qcbm_circuit_ion_trap

def optimize_variational_qcbm_circuit(n_qubits, single_qubit_gate,
    static_entangler, topology, epsilon, initial_params, distance_measure, 
    backend, optimizer, target_bitstring_distribution,**kwargs):
    """Optimize a quantum circuit Born machine (QCBM).

    Args:
        n_qubits (int): The number of qubits in the qcbm circuit
        single_qubit_gate (str): See build_qcbm_circuit_ion_trap. 
        static_entangler (str): See build_qcbm_circuit_ion_trap.
        topology (str): See build_qcbm_circuit_ion_trap.
        epsilon (float):  See evaluate_distribution_distance.
        initial_params: Initial parameters for the optimization
        distance_measure (str): See evaluate_distribution_distance.
        backend (zquantum.core.interfaces.backend.QuantumSimulator): the backend to run the circuits on 
        optimizer (zquantum.core.interfaces.optimizer.Optimizer): the optimizer used to manage the optimization process
        target_bitstring_distribution : See evaluate_distribution_distance.
    Returns:
    """
    distribution_history = []

    def cost_function(params):
        # Build the ansatz circuit
        qcbm_circuit = build_qcbm_circuit_ion_trap(
                    n_qubits, params, single_qubit_gate, static_entangler,
                    topology=topology)

        measured_distr = backend.get_bitstring_distribution(qcbm_circuit)

        distribution_history.append(measured_distr)

        value_estimate = ValueEstimate(evaluate_distribution_distance(
                                            target_bitstring_distribution,
                                            measured_distr,
                                            distance_measure,
                                            epsilon=epsilon))

        return value_estimate.value

    optimization_results = optimizer.minimize(cost_function, initial_params)

    for i, evaluation in enumerate(optimization_results['history']):
        evaluation['bitstring_distribution'] = distribution_history[i].distribution_dict

    return optimization_results
