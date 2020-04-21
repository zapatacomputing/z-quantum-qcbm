import numpy as np
from zquantum.core.circuit import Circuit, Qubit, Gate

def get_single_qubit_layer(n_qubits, params, single_qubit_gate):
    """Builds a layer of single-qubit gates acting on all qubits in a quantum circuit.

    Args:
        n_qubits (int): number of qubits in the circuit.
        params (numpy.array): parameters of the single-qubit gates.
        single_qubit_gate (str): the gate to be applied to each qubit.

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """

    output = Circuit()
    qubits = [Qubit(qubit_index) for qubit_index in range(n_qubits)]
    output.qubits = qubits
    for qubit_index in range(n_qubits):
        output.gates.append(Gate(single_qubit_gate,[qubits[qubit_index]], [params[qubit_index]]))

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


def get_entangling_layer(n_qubits, params, static_entangler, single_qubit_gate, topology):
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
    circuit = Circuit()
    if topology == 'all':
        circuit += get_all_topology(n_qubits, params, static_entangler)
    return circuit


def build_qcbm_circuit_ion_trap(n_qubits, input_params, single_qubit_gate, static_entangler, topology='all'):
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

    assert(n_qubits > 1)

    # Extract params of layer zero to 1d array of size 2*n_qubits (Rx layer and Rz layer)
    n_params_layer_zero = 2*n_qubits
    params_layer_zero = np.take(input_params, list(range(2*n_qubits)))
    assert(params_layer_zero.shape[0] == n_params_layer_zero)
    assert(params_layer_zero.ndim == 1)

    n_params_per_layer = int((n_qubits*(n_qubits-1))/2)

    if (input_params.shape[0]-2*n_qubits)%n_params_per_layer==0:
        n_layers = int((input_params.shape[0]-2*n_qubits)/n_params_per_layer) # number of entangling layers
    else:
        raise RuntimeError("Incomplete layers are not supported yet.")
        #In thise case we should:
        # 1. redefine n_layers
        #n_layers = int((input_params.shape[0]-n_qubits)/n_params_per_layer)+1
        # 2. create a 0s array to fill the last layer
        # fill = np.zeros(n_params_per_layer-(input_params.shape[0]-n_qubits)%n_params_per_layer)
        # 3. fill the last layer
        # input_params = np.append(input_params,fill)
    assert(n_layers > 0)

    # Extract params of entangling layers to 1d array and then reshape to 2d array (matrix) of size n_layers*n_params_per_layer
    params_from_input = np.take(input_params, list(range(2*n_qubits,input_params.shape[0])))
    params=np.reshape(params_from_input, (n_layers, n_params_per_layer))
    assert(params.shape[1] == n_params_per_layer)

    assert(single_qubit_gate in ['Rx'])

    assert(static_entangler in ['XX'])

    assert(topology in ['all'])

    # Initialize quantum circuit
    circuit = Circuit()

    # Add first layer of single-qubit gates Rx
    circuit += get_single_qubit_layer(n_qubits,params_layer_zero[0:n_qubits], single_qubit_gate)
    # Add second layer of single-qubit gates Rz
    circuit += get_single_qubit_layer(n_qubits,params_layer_zero[n_qubits:], 'Rz')

    # Creating a counter system to iterate through the circuit (over the entangling layers)
    counter = 0
    for n in range(0,n_layers):
        circuit += get_entangling_layer(n_qubits, params[counter], static_entangler, single_qubit_gate, topology)
        counter += 1

    return circuit


def generate_random_initial_params(n_qubits, n_layers=1, topology='all', min_val=0., max_val=1., n_par=0, seed=None):
    """Generate random parameters for the QCBM circuit (iontrap ansatz).

    Args:
        n_qubits (int): number of qubits in the circuit.
        n_layers (int): number of entangling layers in the circuit. If n_layers=-1, you can specify a custom number of parameters (see below).
        topology (str): describes topology of qubits connectivity.
        min_val (float): minimum parameter value.
        max_val (float): maximum parameter value.
        n_par (int): specifies number of parameters to be generated in case of incomplete layers (i.e. n_layers=-1).
        seed (int): initialize random generator

    Returns:
        numpy.array: the generated parameters, stored in a 1D array.
    """
    gen = np.random.RandomState(seed)
    assert(topology == 'all')
    n_params_layer_zero = 2*n_qubits
    n_params_per_layer = int((n_qubits*(n_qubits-1))/2)

    if n_layers==-1:
        n_params=n_par
    else:
        assert(n_layers>0)
        if n_par!=0:
            raise ValueError("If n_layers is specified, n_par is automatically computed.")
        n_params = n_params_layer_zero+n_layers*n_params_per_layer
    params = gen.uniform(min_val, max_val, n_params)
    return(params)
