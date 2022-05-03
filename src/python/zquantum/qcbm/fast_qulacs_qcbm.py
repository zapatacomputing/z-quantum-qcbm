import random
from qulacs import ParametricQuantumCircuit, QuantumCircuit, QuantumState
from itertools import combinations
import numpy as np


class QCBM:
    _allowed_single_qubit_gates = ["RX", "RZ"]
    _allowed_two_qubit_gates = ["XX"]

    def __init__(self, n_qubits: int, n_layers: int, topology: str = "all"):
        assert topology == "all", "For the moment I only support all topology"
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self.circuit = self._init_circuit()
        self.n_params = self.circuit.get_parameter_count()
        self.state = QuantumState(n_qubits)

    def create_layer_of_gates(self, circuit: QuantumCircuit, gate: str = "RX"):
        assert gate in self._allowed_single_qubit_gates
        add_gate = {
            "RX": circuit.add_parametric_RX_gate,
            "RZ": circuit.add_parametric_RZ_gate,
        }[gate]
        angle = random.random()
        for i in range(self._n_qubits):
            add_gate(i, angle)

    def create_entangling_layer(self, circuit: QuantumCircuit, gate: str = "XX"):
        assert gate in self._allowed_two_qubit_gates
        add_gate = {
            "XX": lambda target, angle: circuit.add_parametric_multi_Pauli_rotation_gate(
                target, [1, 1], angle
            )
        }[gate]
        angle = random.random()
        for pair in combinations(list(range(0, self._n_qubits)), 2):
            add_gate(pair, angle)

    def _init_circuit(self):
        circuit = ParametricQuantumCircuit(self._n_qubits)
        if self._n_layers == 1:
            self.create_layer_of_gates(circuit, "RX")
            return circuit
        for layer_index in range(self._n_layers):
            if layer_index == 0:
                self.create_layer_of_gates(circuit, "RX")
                self.create_layer_of_gates(circuit, "RZ")
            elif self._n_layers % 2 == 1 and layer_index == self._n_layers - 1:
                self.create_layer_of_gates(circuit, "RZ")
                self.create_layer_of_gates(circuit, "RX")
            elif self._n_layers % 2 == 0 and layer_index == self._n_layers - 2:
                self.create_layer_of_gates(circuit, "RX")
                self.create_layer_of_gates(circuit, "RZ")
                self.create_layer_of_gates(circuit, "RX")
            elif self._n_layers % 2 == 1 and layer_index == self._n_layers - 3:
                self.create_layer_of_gates(circuit, "RX")
                self.create_layer_of_gates(circuit, "RZ")
                self.create_layer_of_gates(circuit, "RX")
            elif layer_index % 2 == 1:
                self.create_entangling_layer(circuit, "XX")
            else:
                self.create_layer_of_gates(circuit, "RX")
                self.create_layer_of_gates(circuit, "RZ")
        return circuit

    def _generate_circuit(self, args: np.ndarray):
        assert args.size == self.n_params
        for i in range(self.n_params):
            self.circuit.set_parameter(i, args[i])
        return self.circuit

    def get_wavefunction(self, args: np.ndarray):
        self._generate_circuit(args)
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        return MyWaveFunction(self.state.get_vector(), self._n_qubits)


class MyWaveFunction:
    def __init__(self, vector: np.ndarray, n_qubits: int):
        self.probs = np.abs(vector) ** 2
        self._n_qubits = n_qubits

    def get_outcome_probs(self):
        values = [
            format(i, "0" + str(self._n_qubits) + "b")[::-1]
            for i in range(len(self.probs))
        ]
        return dict(zip(values, self.probs))
