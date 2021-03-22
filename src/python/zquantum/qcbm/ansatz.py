import json
from typing import Optional, List
import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuit import Circuit, create_layer_of_gates
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import (
    ansatz_property,
)
from zquantum.core.utils import SCHEMA_VERSION
from .ansatz_utils import get_entangling_layer


class QCBMAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    topology = ansatz_property("topology")

    def __init__(
        self,
        number_of_layers: int,
        number_of_qubits: int,
        topology: str = "all",
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
                [
                    sympy.Symbol("theta_{}".format(i))
                    for i in range(self.number_of_params)
                ]
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

    def to_dict(self):
        """Creates a dictionary representing a QCBM ansatz.

        Returns:
            dictionary (dict): the dictionary
        """
        dictionary = {
            "schema": SCHEMA_VERSION + "-qcbm_ansatz",
            "number_of_layers": self.number_of_layers,
            "number_of_qubits": self.number_of_qubits,
            "topology": self.topology,
        }

        return dictionary

    # def from_dict(self, dictionary):
    #     """Loads information of the qcbm_ansatz from a dictionary. This corresponds to the
    #     number of layers, number of qubits, and topplogy.

    #     Args:
    #         dictionary (dict): the dictionary

    #     Returns:
    #         A QCBM_Ansatz object
    #     """
    #     output = QCBMAnsatz(
    #         dictionary["number_of_layers"],
    #         dictionary["number_of_qubits"],
    #         dictionary["topology"],
    #     )
    #     return output


def save_qcbm_ansatz_set(qcbm_ansatz_set: List[QCBMAnsatz], filename: str) -> None:
    """Save a set of qcbm_ansatz to a file.

    Args:
        qcbm_ansatz_set (list): a list ansatz to be saved
        file (str): the name of the file
    """
    dictionary = {}
    dictionary["schema"] = SCHEMA_VERSION + "-qcbm-ansatz-set"
    dictionary["qcbm_ansatz"] = []

    for ansatz in qcbm_ansatz_set:
        dictionary["qcbm_ansatz"].append(ansatz.to_dict())

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_qcbm_ansatz_set(file: str) -> List[QCBMAnsatz]:
    """Load a list of qcbm_ansatz from a json file using a schema.

    Arguments:
        file (str): the name of the file

    Returns:
        object: a list of qcbm_ansatz loaded from the file
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    qcbm_ansatz_set = []
    for item in data["qcbm_ansatz"]:
        qcbm_ansatz_set.append(
            QCBMAnsatz(
                number_of_layers=item["number_of_layers"],
                number_of_qubits=item["number_of_qubits"],
                topology=item["topology"],
            )
        )
    # for i in range(len(data["qcbm_ansatz"])):
    #     qcbm_ansatz_set.append((data["qcbm_ansatz"][i]).from_dict())

    return qcbm_ansatz_set
