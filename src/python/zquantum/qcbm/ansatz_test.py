import json
from unittest import mock
from io import StringIO
import numpy as np
import pytest
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.utils import SCHEMA_VERSION


from .ansatz import QCBMAnsatz, load_qcbm_ansatz_set, save_qcbm_ansatz_set
from .ansatz_utils import get_entangling_layer


class TestQCBMAnsatz(AnsatzTests):
    @pytest.fixture
    def number_of_qubits(self):
        return 4

    @pytest.fixture
    def topology(self):
        return "all"

    @pytest.fixture
    def ansatz(self, number_of_qubits, topology):
        return QCBMAnsatz(
            number_of_layers=2,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )

    def test_get_executable_circuit_too_many_parameters(
        self, number_of_qubits, topology
    ):
        # Given
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
        ]
        params = np.concatenate(params)
        ansatz = QCBMAnsatz(
            number_of_layers=2,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        # When/Then
        with pytest.raises(ValueError):
            ansatz.get_executable_circuit(params)

    def test_ansatz_circuit_one_layer(self, number_of_qubits, topology):
        # Given
        number_of_layers = 1
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )

        params = [np.ones(number_of_qubits)]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        expected_circuit = Circuit(expected_pycircuit)

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_two_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 2
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )

        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_circuit = Circuit(expected_pycircuit)
        expected_circuit += get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_three_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 3
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[2][i + number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_circuit = (
            expected_first_layer + expected_second_layer + expected_third_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_four_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 4
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )

        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[2][i + 2 * number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_five_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 5
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[2][i + 2 * number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[4][i + number_of_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
            + expected_fifth_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_six_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 6
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[4][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[4][i + 2 * number_of_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(
            params[5], number_of_qubits, "XX", topology
        )
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
            + expected_fifth_layer
            + expected_sixth_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_seven_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 7
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[4][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[4][i + 2 * number_of_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(
            params[5], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[6][i + number_of_qubits], i)
            )
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
            + expected_fifth_layer
            + expected_sixth_layer
            + expected_seventh_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_eight_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 8
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[4][i + number_of_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(
            params[5], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[6][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[6][i + 2 * number_of_qubits], i)
            )
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(
            params[7], number_of_qubits, "XX", topology
        )
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
            + expected_fifth_layer
            + expected_sixth_layer
            + expected_seventh_layer
            + expected_eigth_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_nine_layers(self, number_of_qubits, topology):
        # Given
        number_of_layers = 9
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
            topology=topology,
        )
        params = [
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(3 * number_of_qubits),
            np.ones(int((number_of_qubits * (number_of_qubits - 1)) / 2)),
            np.ones(2 * number_of_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[0][i + number_of_qubits], i)
            )
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[2][i + number_of_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[4][i + number_of_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(
            params[5], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RZ(params[6][i + number_of_qubits], i)
            )
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[6][i + 2 * number_of_qubits], i)
            )
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(
            params[7], number_of_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(number_of_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[8][i], i))
        for i in range(number_of_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[8][i + number_of_qubits], i)
            )
        expected_ninth_layer = Circuit(expected_pycircuit)
        expected_circuit = (
            expected_first_layer
            + expected_second_layer
            + expected_third_layer
            + expected_fourth_layer
            + expected_fifth_layer
            + expected_sixth_layer
            + expected_seventh_layer
            + expected_eigth_layer
            + expected_ninth_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    @pytest.fixture
    def mock_open(self):
        mock_open = mock.mock_open()
        with mock.patch("qcbm.ansatz.open", mock_open):
            yield mock_open

    def test_saving_qcbm_ansatz_set_opens_file_for_writing_using_context_manager(
        self,
        mock_open,
    ):
        """Saving qcbm ansatz set opens file for writing using context manager."""
        ansatzes = [
            QCBMAnsatz(number_of_layers=2, number_of_qubits=4, topology="all"),
            QCBMAnsatz(number_of_layers=4, number_of_qubits=8, topology="line"),
        ]
        save_qcbm_ansatz_set(ansatzes, "/some/path/to/ansatz/set.json")

        mock_open.assert_called_once_with("/some/path/to/ansatz/set.json", "w")
        mock_open().__enter__.assert_called_once()
        mock_open().__exit__.assert_called_once()

    def test_saving_qcbm_ansatz_set_writes_correct_json_data_to_file(self, mock_open):
        """Saving qcbm ansatz set writes correct list of json dictionaries to file."""
        ansatzes = [
            QCBMAnsatz(number_of_layers=2, number_of_qubits=4, topology="all"),
            QCBMAnsatz(number_of_layers=4, number_of_qubits=8, topology="line"),
        ]

        expected_dict = {
            "qcbm_ansatz": [ansatz.to_dict() for ansatz in ansatzes],
            "schema": SCHEMA_VERSION + "-qcbm-ansatz-set",
        }

        save_qcbm_ansatz_set(ansatzes, "/some/path/to/ansatz/set.json")

        written_data = mock_open().__enter__().write.call_args[0][0]
        assert json.loads(written_data) == expected_dict

    def test_saved_qcbm_ansatz_set_can_be_loaded(self, mock_open):
        """Saved qcbm ansatz set can be loaded to obtain the same ansatz set."""
        fake_file = StringIO()
        mock_open().__enter__.return_value = fake_file
        ansatzes = [
            QCBMAnsatz(number_of_layers=2, number_of_qubits=4, topology="all"),
            QCBMAnsatz(number_of_layers=4, number_of_qubits=8, topology="line"),
        ]

        save_qcbm_ansatz_set(ansatzes, "ansatzes.json")
        fake_file.seek(0)

        loaded_ansatzes = load_qcbm_ansatz_set(fake_file)
        assert all(
            (
                ansatz.to_dict()[key] == loaded_ansatz.to.dict()[key]
                for key in ansatz.to_dict().keys()
            )
            for ansatz, loaded_ansatz in zip(ansatzes, loaded_ansatzes)
        )

        assert all(
            ansatz.to_dict().keys() == loaded_ansatz.to_dict().keys()
            for ansatz, loaded_ansatz in zip(ansatzes, loaded_ansatzes)
        )
