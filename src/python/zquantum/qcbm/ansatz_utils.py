from zquantum.core.wip.circuits import Circuit, GatePrototype
import numpy as np


def get_entangling_layer(
    params: np.ndarray, n_qubits: int, entangling_gate: GatePrototype, topology: str
) -> Circuit:
    """Builds an entangling layer in the circuit.

    Args:
        params: parameters of the circuit.
        n_qubits: number of qubits in the circuit.
        entangling_gate: gate specification for the entangling layer.
        topology: describes connectivity of the qubits in the desired circuit
    """
    if topology == "all":
        return get_entangling_layer_all_topology(params, n_qubits, entangling_gate)
    elif topology == "line":
        return get_entangling_layer_line_topology(params, n_qubits, entangling_gate)
    else:
        raise RuntimeError("Topology: {} is not supported".format(topology))


def get_entangling_layer_all_topology(
    params: np.ndarray, n_qubits: int, entangling_gate: GatePrototype
) -> Circuit:
    """Builds a circuit representing an entangling layer according to the all-to-all topology.

    Args:
        params: parameters of the circuit.
        n_qubits: number of qubits in the circuit.
        entangling_gate: gate specification for the entangling layer.
    """

    assert params.shape[0] == int((n_qubits * (n_qubits - 1)) / 2)

    circuit = Circuit()
    i = 0
    for qubit1_index in range(n_qubits - 1):
        for qubit2_index in range(qubit1_index + 1, n_qubits):
            circuit += entangling_gate(params[i])(qubit1_index, qubit2_index)
            i += 1
    return circuit


def get_entangling_layer_line_topology(
    params: np.ndarray, n_qubits: int, entangling_gate: GatePrototype
) -> Circuit:
    """Builds a circuit representing an entangling layer according to the line topology.

    Args:
        params: parameters of the circuit.
        n_qubits: number of qubits in the circuit.
        entangling_gate: gate specification for the entangling layer.
    """
    assert params.shape[0] == n_qubits - 1

    circuit = Circuit()
    for qubit1_index in range(n_qubits - 1):
        circuit += entangling_gate(params[qubit1_index])(qubit1_index, qubit1_index + 1)
    return circuit
