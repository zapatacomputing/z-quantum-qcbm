import unittest
import numpy as np
import cirq
import itertools
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.utils import RNDSEED, compare_unitary

from .ansatz_utils import (
    get_single_qubit_layer,
    get_entangling_layer,
    get_entangling_layer_all_topology,
    get_entangling_layer_line_topology,
)


class TestAnsatzUtils(unittest.TestCase):
    def test_get_single_qubit_layer_wrong_num_params(self):
        # Given
        single_qubit_gates = ["Ry"]
        n_qubits = 2
        params = np.ones(3)
        # When/Then
        self.assertRaises(
            AssertionError,
            lambda: get_single_qubit_layer(params, n_qubits, single_qubit_gates),
        )

    def test_get_single_qubit_layer_one_gate(self):
        # Given
        single_qubit_gates = ["Ry"]
        n_qubits_list = [2, 3, 4, 10]

        for n_qubits in n_qubits_list:
            # Given
            params = [x for x in range(0, n_qubits)]
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n_qubits)]
            for i in range(0, n_qubits):
                test.append(cirq.Ry(params[i]).on(qubits[i]))
            u_cirq = test._unitary_()

            # When
            circ = get_single_qubit_layer(params, n_qubits, single_qubit_gates)
            unitary = circ.to_cirq()._unitary_()

            # Then
            self.assertEqual(circ.n_multiqubit_gates, 0)
            self.assertEqual(compare_unitary(unitary, u_cirq, tol=1e-10), True)

    def test_get_single_qubit_layer_multiple_gates(self):
        # Given
        single_qubit_gates = ["Ry", "Rx", "Rz"]
        n_qubits_list = [2, 3, 4, 10]

        for n_qubits in n_qubits_list:
            # Given
            params = [x for x in range(0, 3 * n_qubits)]
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n_qubits)]
            for i in range(0, n_qubits):
                test.append(cirq.Ry(params[i]).on(qubits[i]))
            for i in range(0, n_qubits):
                test.append(cirq.Rx(params[n_qubits + i]).on(qubits[i]))
            for i in range(0, n_qubits):
                test.append(cirq.Rz(params[2 * n_qubits + i]).on(qubits[i]))
            u_cirq = test._unitary_()

            # When
            circ = get_single_qubit_layer(params, n_qubits, single_qubit_gates)
            unitary = circ.to_cirq()._unitary_()

            # Then
            self.assertEqual(circ.n_multiqubit_gates, 0)
            self.assertEqual(compare_unitary(unitary, u_cirq, tol=1e-10), True)

    def test_get_entangling_layer_all_topology(self):
        # Given
        n_qubits = 4
        static_entangler = "XX"
        topology = "all"
        params = np.asarray([0, 0, 0, 0, 0, 0])

        # When
        ent_layer = get_entangling_layer_all_topology(
            params, n_qubits, static_entangler
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        self.assertEqual(ent_layer.gates[0].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[0].qubits[1].index, 1)
        # XX on 0, 2
        self.assertEqual(ent_layer.gates[1].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[1].qubits[1].index, 2)
        # XX on 0, 3
        self.assertEqual(ent_layer.gates[2].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[2].qubits[1].index, 3)
        # XX on 1, 2
        self.assertEqual(ent_layer.gates[3].qubits[0].index, 1)
        self.assertEqual(ent_layer.gates[3].qubits[1].index, 2)
        # XX on 1, 3
        self.assertEqual(ent_layer.gates[4].qubits[0].index, 1)
        self.assertEqual(ent_layer.gates[4].qubits[1].index, 3)
        # XX on 2, 3
        self.assertEqual(ent_layer.gates[5].qubits[0].index, 2)
        self.assertEqual(ent_layer.gates[5].qubits[1].index, 3)

    def test_get_entangling_layer_line_topology(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "line"
        params = np.asarray([0, 0, 0])

        # When
        ent_layer = get_entangling_layer_line_topology(
            params, n_qubits, static_entangler
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        self.assertEqual(ent_layer.gates[0].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[0].qubits[1].index, 1)
        # XX on 1, 2
        self.assertEqual(ent_layer.gates[1].qubits[0].index, 1)
        self.assertEqual(ent_layer.gates[1].qubits[1].index, 2)
        # XX on 2, 3
        self.assertEqual(ent_layer.gates[2].qubits[0].index, 2)
        self.assertEqual(ent_layer.gates[2].qubits[1].index, 3)

    def test_get_entangling_layer_toplogy_supported(self):
        # Given
        n_qubits_list = [2, 3, 4, 5]
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"

        for n_qubits in n_qubits_list:
            # Given
            params = np.zeros((int((n_qubits * (n_qubits - 1)) / 2)))
            all_topology_layer = get_entangling_layer_all_topology(
                params, n_qubits, static_entangler
            )
            # When
            entangling_layer = get_entangling_layer(
                params, n_qubits, static_entangler, topology
            )
            # Then
            self.assertEqual(all_topology_layer, entangling_layer)

        # Given
        topology = "line"

        for n_qubits in n_qubits_list:
            # Given
            params = np.zeros((n_qubits - 1))
            line_topology_layer = get_entangling_layer_line_topology(
                params, n_qubits, static_entangler
            )
            # When
            entangling_layer = get_entangling_layer(
                params, n_qubits, static_entangler, topology
            )
            # Then
            self.assertEqual(line_topology_layer, entangling_layer)

    def test_get_entangling_layer_toplogy_not_supported(self):
        # Given
        n_qubits = 2
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "NOT SUPPORTED"
        params = np.zeros(1)

        # When
        self.assertRaises(
            RuntimeError,
            lambda: get_entangling_layer(params, n_qubits, static_entangler, topology),
        )

