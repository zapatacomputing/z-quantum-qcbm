from zquantum.core.circuit import Circuit, Gate, Qubit
import numpy as np


def get_entangling_layer(
    params: np.ndarray,
    n_qubits: int,
    entangling_gate: str,
    topology: str,
    topology_kwargs=None,
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
    elif topology == "star":
        center_qubit = topology_kwargs["center_qubit"]
        return get_entangling_layer_star_topology(
            params, n_qubits, entangling_gate, center_qubit
        )
    elif topology == "graph":
        # support adjacency matrix or adjacency list
        if (
            "adjacency_matrix" in topology_kwargs.keys()
            and "adjacency_list" in topology_kwargs.keys()
        ):
            raise RuntimeError("Only one of adjacency list/matrix can be specified.")
        if "adjacency_matrix" in topology_kwargs.keys():
            adjacency_matrix = topology_kwargs["adjacency_matrix"]
            if np.array_equal(adjacency_matrix, adjacency_matrix.T):
                print("Warning: This matrix is not symmetric.")
        else:
            adjacency_matrix = adjacency_list_to_matrix(
                n_qubits, topology_kwargs["adjacency_list"]
            )
        return get_entangling_layer_graph_topology(
            params, n_qubits, entangling_gate, adjacency_matrix
        )
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

    return get_entangling_layer_graph_topology(
        params, n_qubits, entangling_gate, np.ones((n_qubits, n_qubits))
    )


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

    line_graph = np.zeros((n_qubits, n_qubits))
    for i in range(0, n_qubits - 1):
        line_graph[i][i + 1] = 1
    return get_entangling_layer_graph_topology(
        params, n_qubits, entangling_gate, line_graph
    )


def get_entangling_layer_star_topology(
    params: np.ndarray, n_qubits: int, entangling_gate: str, center_qubit: int
) -> Circuit:
    """Builds a circuit representing an entangling layer according to the star topology.

    Args:
        params (numpy.array): parameters of the circuit.
        n_qubits (int): number of qubits in the circuit.
        entangling_gate (str): gate specification for the entangling layer.
        center_qubit (int): the center qubit of the star topology.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert center_qubit < n_qubits
    assert params.shape[0] == n_qubits - 1

    star_graph = np.zeros((n_qubits, n_qubits))
    for i in range(0, n_qubits):
        star_graph[i][center_qubit] = 1
    return get_entangling_layer_graph_topology(
        params, n_qubits, entangling_gate, star_graph
    )


def get_entangling_layer_graph_topology(
    params: np.ndarray,
    n_qubits: int,
    entangling_gate: str,
    adjacency_matrix: np.ndarray,
) -> Circuit:
    """Builds a circuit representing an entangling layer according to a general graph topology.

    Args:
        params (numpy.array): parameters of the circuit.
        n_qubits (int): number of qubits in the circuit.
        entangling_gate (str): gate specification for the entangling layer.
        adjacency_matrix: (numpy.array): adjacency matrix for the entangling layer.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == n_qubits

    circuit = Circuit()
    circuit.qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    i = 0
    for qubit1_index in range(0, n_qubits - 1):
        for qubit2_index in range(qubit1_index + 1, n_qubits):
            if (
                adjacency_matrix[qubit1_index][qubit2_index]
                or adjacency_matrix[qubit2_index][qubit1_index]
            ):
                circuit.gates.append(
                    Gate(
                        entangling_gate,
                        [circuit.qubits[qubit1_index], circuit.qubits[qubit2_index]],
                        [params[i]],
                    )
                )
                i += 1
    return circuit


def adjacency_list_to_matrix(n_qubits: int, adj_list: np.ndarray):
    """Converts an adjacency list to an adjacency matrix.

    Args:
        n_qubits (int): number of qubits in the circuit.
        adjacency_list: (numpy.array): adjacency list for the entangling layer.

    Returns:
        adj_matrix: a numpy.array object
    """
    assert adj_list.shape[1] == 2
    adj_matrix = np.zeros((n_qubits, n_qubits))
    for pair in adj_list:
        assert 0 <= pair[0] < n_qubits and 0 <= pair[1] < n_qubits
        adj_matrix[pair[0], pair[1]] = max(1, adj_matrix[pair[0], pair[1]])
    return adj_matrix