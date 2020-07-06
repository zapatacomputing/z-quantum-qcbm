from zquantum.core.circuit import Circuit, Gate, Qubit
import numpy as np


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
