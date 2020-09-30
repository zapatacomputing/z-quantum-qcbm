import unittest
import numpy as np
import itertools
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.interfaces.ansatz_test import AnsatzTests

from .ansatz import QCBMAnsatz
from .ansatz_utils import get_entangling_layer


class TestQCBMAnsatz(unittest.TestCase, AnsatzTests):
    def setUp(self):
        self.n_layers = 2
        self.n_qubits = 4
        self.topology = "all"
        self.ansatz = QCBMAnsatz(self.n_layers, self.n_qubits, self.topology)

    def test_get_executable_circuit_too_many_parameters(self):
        # Given
        params = [
            np.ones(2 * self.n_qubits),
            np.ones(int((self.n_qubits * (self.n_qubits - 1)) / 2)),
            np.ones(2 * self.n_qubits),
        ]
        params = np.concatenate(params)

        # When/Then
        self.assertRaises(
            ValueError, lambda: self.ansatz.get_executable_circuit(params),
        )

    def test_ansatz_circuit_one_layer(self):
        # Given
        n_layers = 1
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [np.ones(n_qubits)]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        expected_circuit = Circuit(expected_pycircuit)

        params = np.concatenate(params)
        n_layers = 1

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_two_layers(self):
        # Given
        n_layers = 2
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [np.ones(2 * n_qubits), np.ones(int((n_qubits * (n_qubits - 1)) / 2))]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_circuit = Circuit(expected_pycircuit)
        expected_circuit += get_entangling_layer(params[1], n_qubits, "XX", topology)

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_three_layers(self):
        # Given
        n_layers = 3
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i + n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_circuit = (
            expected_first_layer + expected_second_layer + expected_third_layer
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_four_layers(self):
        # Given
        n_layers = 4
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[2][i + 2 * n_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
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
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_five_layers(self):
        # Given
        n_layers = 5
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[2][i + 2 * n_qubits], i)
            )
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i + n_qubits], i))
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
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_six_layers(self):
        # Given
        n_layers = 6
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[4][i + 2 * n_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
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
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_seven_layers(self):
        # Given
        n_layers = 7
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[4][i + 2 * n_qubits], i)
            )
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i + n_qubits], i))
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
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_eight_layers(self):
        # Given
        n_layers = 8
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i + n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[6][i + 2 * n_qubits], i)
            )
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(params[7], n_qubits, "XX", topology)
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
        self.assertEqual(circuit, expected_circuit)

    def test_ansatz_circuit_nine_layers(self):
        # Given
        n_layers = 9
        n_qubits = 4
        topology = "all"
        ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
        params = [
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(3 * n_qubits),
            np.ones(int((n_qubits * (n_qubits - 1)) / 2)),
            np.ones(2 * n_qubits),
        ]

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i + n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(
            params[1], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i + n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(
            params[3], n_qubits, "XX", topology
        )
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i + n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i + n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(
                pyquil.gates.RX(params[6][i + 2 * n_qubits], i)
            )
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(params[7], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[8][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[8][i + n_qubits], i))
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
        self.assertEqual(circuit, expected_circuit)

