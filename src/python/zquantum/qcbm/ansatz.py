import numpy as np
import sympy
from zquantum.core.circuit import Circuit, Qubit, Gate, create_layer_of_gates
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import (
    ansatz_property,
    invalidates_parametrized_circuit,
)
from typing import Optional, List
from .ansatz_utils import get_entangling_layer

from overrides import overrides


class QCBMAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    topology = ansatz_property("topology")

    def __init__(
        self, number_of_layers: int, number_of_qubits: int, topology: str = "all",
    ):
        """
        An ansatz implementation used for running the Quantum Circuit Born Machine.

        Args:
            number_of_layers (int): number of entangling layers in the circuit.
            number_of_qubits (int): number of qubits in the circuit.
            topology (str): the topology representing the connectivity of the qubits.

        Attributes:
            number_of_qubits (int): See Args
            number_of_layers (int): See Args
            topology (str): See Args
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
        """
        super().__init__(number_of_layers)
        self._number_of_qubits = number_of_qubits
        self._topology = topology
        if number_of_layers == 0:
            raise ValueError("QCBMAnsatz is only defined for number_of_layers > 0.")

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return np.sum(self.get_number_of_parameters_by_layer())

    @property
    def n_params_per_ent_layer(self) -> int:
        if self.topology == "all":
            return int((self.number_of_qubits * (self.number_of_qubits - 1)) / 2)
        elif self.topology == "line":
            return self.number_of_qubits - 1
        else:
            raise RuntimeError("Topology {} is not supported".format(self.topology))

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Builds a qcbm ansatz circuit, using the ansatz in https://advances.sciencemag.org/content/5/10/eaaw9918/tab-pdf (Fig.2 - top).

        Args:
            params (numpy.array): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if params is None:
            params = np.asarray(
                [sympy.Symbol("theta_{}".format(i)) for i in range(self.number_of_params)]
            )

        assert len(params) == self.number_of_params

        if self.number_of_layers == 1:
            # Only one layer, should be a single layer of rotations with Rx
            return create_layer_of_gates(self.number_of_qubits, "Rx", params)

        circuit = Circuit()
        parameter_index = 0
        for layer_index in range(self.number_of_layers):
            if layer_index == 0:
                # First layer is always 2 single qubit rotations on Rx Rz
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[parameter_index : parameter_index + self.number_of_qubits],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rz",
                    params[
                        parameter_index
                        + self.number_of_qubits : parameter_index
                        + 2 * self.number_of_qubits
                    ],
                )
                parameter_index += 2 * self.number_of_qubits
            elif (
                self.number_of_layers % 2 == 1
                and layer_index == self.number_of_layers - 1
            ):
                # Last layer for odd number of layers is rotations on Rx Rz
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rz",
                    params[parameter_index : parameter_index + self.number_of_qubits],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[
                        parameter_index
                        + self.number_of_qubits : parameter_index
                        + 2 * self.number_of_qubits
                    ],
                )
                parameter_index += 2 * self.number_of_qubits
            elif (
                self.number_of_layers % 2 == 0
                and layer_index == self.number_of_layers - 2
            ):
                # Even number of layers, second to last layer is 3 rotation layer with Rx Rz Rx
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[parameter_index : parameter_index + self.number_of_qubits],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rz",
                    params[
                        parameter_index
                        + self.number_of_qubits : parameter_index
                        + 2 * self.number_of_qubits
                    ],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[
                        parameter_index
                        + 2 * self.number_of_qubits : parameter_index
                        + 3 * self.number_of_qubits
                    ],
                )
                parameter_index += 3 * self.number_of_qubits
            elif (
                self.number_of_layers % 2 == 1
                and layer_index == self.number_of_layers - 3
            ):
                # Odd number of layers, third to last layer is 3 rotation layer with Rx Rz Rx
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[parameter_index : parameter_index + self.number_of_qubits],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rz",
                    params[
                        parameter_index
                        + self.number_of_qubits : parameter_index
                        + 2 * self.number_of_qubits
                    ],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[
                        parameter_index
                        + 2 * self.number_of_qubits : parameter_index
                        + 3 * self.number_of_qubits
                    ],
                )
                parameter_index += 3 * self.number_of_qubits
            elif layer_index % 2 == 1:
                # Currently on an entangling layer
                circuit += get_entangling_layer(
                    params[
                        parameter_index : parameter_index + self.n_params_per_ent_layer
                    ],
                    self.number_of_qubits,
                    "XX",
                    self.topology,
                )
                parameter_index += self.n_params_per_ent_layer
            else:
                # A normal single qubit rotation layer of Rx Rz
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rx",
                    params[parameter_index : parameter_index + self.number_of_qubits],
                )
                circuit += create_layer_of_gates(
                    self.number_of_qubits,
                    "Rz",
                    params[
                        parameter_index
                        + self.number_of_qubits : parameter_index
                        + 2 * self.number_of_qubits
                    ],
                )
                parameter_index += 2 * self.number_of_qubits

        return circuit

    def get_number_of_parameters_by_layer(self) -> np.ndarray:
        """Determine the number of parameters needed for each layer in the ansatz

        Returns:
            A 1D array of integers 
        """
        if self.number_of_layers == 1:
            # If only one layer, then only need parameters for a single layer of Rx gates
            return np.asarray([self.number_of_qubits])

        num_params_by_layer = []
        for layer_index in range(self.number_of_layers):
            if layer_index == 0:
                # First layer is always 2 parameters per qubit for 2 single qubit rotations
                num_params_by_layer.append(self.number_of_qubits * 2)
            elif (
                self.number_of_layers % 2 == 1
                and layer_index == self.number_of_layers - 1
            ):
                # Last layer for odd number of layers is 2 layer rotations
                num_params_by_layer.append(self.number_of_qubits * 2)
            elif (
                self.number_of_layers % 2 == 0
                and layer_index == self.number_of_layers - 2
            ):
                # Even number of layers, second to last layer is 3 rotation layer
                num_params_by_layer.append(self.number_of_qubits * 3)
            elif (
                self.number_of_layers % 2 == 1
                and layer_index == self.number_of_layers - 3
            ):
                # Odd number of layers, third to last layer is 3 rotation layer
                num_params_by_layer.append(self.number_of_qubits * 3)
            elif layer_index % 2 == 1:
                # Currently on an entangling layer
                num_params_by_layer.append(self.n_params_per_ent_layer)
            else:
                # A normal single qubit rotation layer
                num_params_by_layer.append(self.number_of_qubits * 2)

        return np.asarray(num_params_by_layer)