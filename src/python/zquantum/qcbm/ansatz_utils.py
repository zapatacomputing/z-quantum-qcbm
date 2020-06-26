from zquantum.core.circuit import Circuit, Gate, Qubit
import numpy as np
from typing import List


def get_single_qubit_layer(
    params: np.ndarray, n_qubits: int, single_qubit_gates: List[str]
) -> Circuit:
    """Builds a circuit representing a layer of single-qubit gates acting on all qubits.

    Args:
        params (numpy.array): parameters of the single-qubit gates.
        n_qubits (int): number of qubits in the circuit.
        single_qubit_gates (str): a list of single qubit gates to be applied to each qubit.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert len(params) == len(single_qubit_gates) * n_qubits

    circuit = Circuit()
    circuit.qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]

    parameter_index = 0
    for gate_type in single_qubit_gates:
        for qubit_index in range(n_qubits):
            # Add single_qubit_gate to each qubit
            circuit.gates.append(
                Gate(
                    gate_type, [circuit.qubits[qubit_index]], [params[parameter_index]]
                )
            )
            parameter_index += 1

    return circuit


def get_entangling_layer(
    params: np.ndarray, n_qubits: int, entangling_gate: str, topology: str
) -> Circuit:
    """Builds an entangling layer in the circuit.

    Args:
        params (numpy.array): parameters of the circuit.
        n_qubits (int): number of qubits in the circuit.
        entangling_gate (str): gate specification for the entangling layer.
        topology (str): describes connectivity of the qubits in the desired circuit

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    if topology == "all":
        return get_entangling_layer_all_topology(params, n_qubits, entangling_gate)
    elif topology == "line":
        return get_entangling_layer_line_topology(params, n_qubits, entangling_gate)
    else:
        raise RuntimeError("Topology: {} is not supported".format(topology))


def get_entangling_layer_all_topology(
    params: np.ndarray, n_qubits: int, entangling_gate: str
) -> Circuit:
    """Builds a circuit representing an entangling layer according to the all-to-all topology.

    Args:
        params (numpy.array): parameters of the circuit.
        n_qubits (int): number of qubits in the circuit.
        entangling_gate (str): gate specification for the entangling layer.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """

    assert params.shape[0] == int((n_qubits * (n_qubits - 1)) / 2)

    circuit = Circuit()
    circuit.qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    i = 0
    for qubit1_index in range(0, n_qubits - 1):
        for qubit2_index in range(qubit1_index + 1, n_qubits):
            circuit.gates.append(
                Gate(
                    entangling_gate,
                    [circuit.qubits[qubit1_index], circuit.qubits[qubit2_index]],
                    [params[i]],
                )
            )
            i += 1
    return circuit


def get_entangling_layer_line_topology(
    params: np.ndarray, n_qubits: int, entangling_gate: str
) -> Circuit:
    """Builds a circuit representing an entangling layer according to the line topology.

    Args:
        params (numpy.array): parameters of the circuit.
        n_qubits (int): number of qubits in the circuit.
        entangling_gate (str): gate specification for the entangling layer.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert params.shape[0] == n_qubits - 1

    circuit = Circuit()
    circuit.qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    for qubit1_index in range(0, n_qubits - 1):
        circuit.gates.append(
            Gate(
                entangling_gate,
                [circuit.qubits[qubit1_index], circuit.qubits[qubit1_index + 1]],
                [params[qubit1_index]],
            )
        )
    return circuit
