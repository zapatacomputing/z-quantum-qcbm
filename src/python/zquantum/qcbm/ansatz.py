import numpy as np
from zquantum.core.circuit import Circuit, Qubit, Gate

def get_single_qubit_layer(n_qubits, params, single_qubit_gates):
    """Builds a layer of single-qubit gates acting on all qubits in a quantum circuit.

    Args:
        n_qubits (int): number of qubits in the circuit.
        params (numpy.array): parameters of the single-qubit gates.
        single_qubit_gates (str): a list of single qubit gates to be applied to each qubit.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert(len(params) == len(single_qubit_gates)*n_qubits)
    output = Circuit()
    qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    output.qubits = qubits
    parameter_index = 0
    for gate in single_qubit_gates:
        for qubit_index in range(n_qubits):
            # Add single_qubit_gate to each qubit
            output.gates.append(Gate(gate,[qubits[qubit_index]], [params[parameter_index]]))
            parameter_index += 1

    return output


def get_all_topology(n_qubits, params, static_entangler):
    """Builds an entangling layer according to the all-to-all topology.

    Args:
        n_qubits (int): number of qubits in the circuit.
        params (numpy.array): parameters of the circuit.
        static_entangler (str): gate specification for the entangling layer.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """

    assert(params.shape[0] == int((n_qubits*(n_qubits-1))/2))
    output=Circuit()
    qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    output.qubits = qubits
    i = 0
    for qubit1_index in range(0, n_qubits-1):
        for qubit2_index in range(qubit1_index+1,n_qubits):
            output.gates.append(Gate(static_entangler,[qubits[qubit1_index],qubits[qubit2_index]],[params[i]]))
            i+=1
    return output


def get_line_topology(n_qubits, params, static_entangler):
    """Builds an entangling layer according to the line topology.

    Args:
        n_qubits (int): number of qubits in the circuit.
        params (numpy.array): parameters of the circuit.
        static_entangler (str): gate specification for the entangling layer.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    assert(params.shape[0] == n_qubits-1)
    output=Circuit()
    qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    output.qubits = qubits
    for qubit1_index in range(0, n_qubits-1):
        output.gates.append(Gate(static_entangler,[qubits[qubit1_index],qubits[qubit1_index+1]],[params[qubit1_index]]))
    return output


def get_entangling_layer(n_qubits, params, static_entangler, topology):
    """Builds an entangling layer in the circuit.

    Args:
        n_qubits (int): number of qubits in the circuit.
        params (numpy.ndarray): parameters of the circui.t
        static_entangler (str): gate specification for the entangling layer.
        single_qubit_gate (str): gate specification for the single-qubit transformation.
        topology (str): topology (str): describes topology of qubits connectivity.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    if topology == "all":
        return get_all_topology(n_qubits, params, static_entangler)
    elif topology == "line":
        return get_line_topology(n_qubits, params, static_entangler)
    else:
        raise RuntimeError("Topology: {} is not supported".format(topology))


def build_qcbm_circuit_ion_trap(n_qubits, input_params, topology='all'):
    """Builds a qcbm ansatz circuit, using the ansatz in https://advances.sciencemag.org/content/5/10/eaaw9918/tab-pdf (Fig.2 - top).

    Args:
        n_qubits (int): number of qubits initialized for circuit.
        input_params (numpy.array): input parameters of the circuit (1d array).
        single_qubit_gate(str): Gate specification for the single-qubit layer (L0).
        static_entangler(str): Gate specification for the entangling layers (L1, L2, ... , Ln).
        topology (str): describes topology of qubits connectivity.

    Returns:
        Circuit: the qcbm circuit
    """
    if len(input_params) == 1:
        # Only one layer, should be a single layer of rotations with Rx
        return get_single_qubit_layer(n_qubits, input_params[0], ["Rx"])

    circuit = Circuit()
    n_layers = len(input_params)
    for layer_index in range(n_layers):
        if layer_index == 0:
            # First layer is always 2 single qubit rotations on Rx Rz
            circuit += get_single_qubit_layer(n_qubits, input_params[layer_index], ["Rx", "Rz"])
        elif n_layers%2 == 1 and layer_index == n_layers-1:
            # Last layer for odd number of layers is rotations on Rx Rz
            circuit += get_single_qubit_layer(n_qubits, input_params[layer_index], ["Rz", "Rx"])
        elif n_layers%2 == 0 and layer_index == n_layers-2:
            # Even number of layers, second to last layer is 3 rotation layer with Rx Rz Rx
            circuit += get_single_qubit_layer(n_qubits, input_params[layer_index], ["Rx", "Rz", "Rx"])
        elif n_layers%2 == 1 and layer_index == n_layers-3:
            # Odd number of layers, third to last layer is 3 rotation layer with Rx Rz Rx
            circuit += get_single_qubit_layer(n_qubits, input_params[layer_index], ["Rx", "Rz", "Rx"])
        elif layer_index%2 == 1:
            # Currently on an entangling layer
            circuit += get_entangling_layer(n_qubits, input_params[layer_index], "XX", topology)
        else:
            # A normal single qubit rotation layer of Rx Rz
            circuit += get_single_qubit_layer(n_qubits, input_params[layer_index], ["Rx", "Rz"])
    # # Add first layer of single-qubit gates consisting of single_qubit_gate on each qubit, followed by Rz on each qubit
    # circuit += get_single_qubit_layer(n_qubits, input_params[0], ["Rx", "Rz"])

    # # Iterate through remaining input parameters, alternating between adding single qubit layers and entangling layers
    # for layer_index, params in enumerate(input_params[1:-1]):
    #     if layer_index%2 == 0:
    #         circuit += get_entangling_layer(n_qubits, params, "XX", topology)
    #     else:
    #         circuit += get_single_qubit_layer(n_qubits, params, ["Rx", "Rz"])

    # if len(input_params) > 1:
    #     if len(input_params)%2 == 0:
    #         circuit += get_entangling_layer(n_qubits, input_params[-1], "XX", topology)
    #     else:
    #         circuit += get_single_qubit_layer(n_qubits, input_params[-1], ["Rx", "Rz", "Rx"])

    return circuit


def generate_random_initial_params(n_qubits, n_layers=2, topology='all', min_val=0., max_val=1., seed=None):
    """Generate random parameters for the QCBM circuit (iontrap ansatz).

    Args:
        n_qubits (int): number of qubits in the circuit.
        n_layers (int): number of entangling layers in the circuit. If n_layers=-1, you can specify a custom number of parameters (see below).
        topology (str): describes topology of qubits connectivity.
        min_val (float): minimum parameter value.
        max_val (float): maximum parameter value.
        seed (int): initialize random generator

    Returns:
        numpy.array: the generated parameters, stored in a 1D array.
    """
    gen = np.random.RandomState(seed)
    if n_layers == 1:
        # If only one layer, then only need parameters for a single layer of Rx gates
        return [gen.uniform(min_val, max_val, n_qubits)]
    
    if topology == "all":
        n_params_per_ent_layer = int((n_qubits*(n_qubits-1))/2)
    elif topology == "line":
        n_params_per_ent_layer = n_qubits-1
    else:
        raise RuntimeError("Topology: {} is not supported".format(topology))

    params = []
    for layer_index in range(n_layers):
        if layer_index == 0:
            # First layer is always 2 parameters per qubit for 2 single qubit rotations
            params.append(gen.uniform(min_val, max_val, n_qubits*2))
        elif n_layers%2 == 1 and layer_index == n_layers-1:
            # Last layer for odd number of layers is 2 layer rotations
            params.append(gen.uniform(min_val, max_val, n_qubits*2))
        elif n_layers%2 == 0 and layer_index == n_layers-2:
            # Even number of layers, second to last layer is 3 rotation layer
            params.append(gen.uniform(min_val, max_val, n_qubits*3))
        elif n_layers%2 == 1 and layer_index == n_layers-3:
            # Odd number of layers, third to last layer is 3 rotation layer
            params.append(gen.uniform(min_val, max_val, n_qubits*3))
        elif layer_index%2 == 1:
            # Currently on an entangling layer
            params.append(gen.uniform(min_val, max_val, n_params_per_ent_layer))
        else:
            # A normal single qubit rotation layer
            params.append(gen.uniform(min_val, max_val, n_qubits*2))
    
    return params