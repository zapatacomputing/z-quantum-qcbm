import unittest
import numpy as np
import itertools
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit, Qubit, Gate

from .ansatz_utils import (
    get_entangling_layer,
    get_entangling_layer_all_topology,
    get_entangling_layer_line_topology,
)


class TestAnsatzUtils(unittest.TestCase):
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

