import pytest
import numpy as np
import itertools
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.interfaces.ansatz_test import AnsatzTests

from .ansatz import QCBMAnsatz
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
            ansatz.get_executable_circuit(params),

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
