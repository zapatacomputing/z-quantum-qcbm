import pytest
import numpy as np
from zquantum.core.circuits import XX

from zquantum.qcbm.ansatz_utils import (
    get_entangling_layer,
    get_entangling_layer_all_topology,
    get_entangling_layer_line_topology,
    get_entangling_layer_star_topology,
    get_entangling_layer_graph_topology,
    adjacency_list_to_matrix,
)


class TestAnsatzUtils:
    def test_get_entangling_layer_all_topology(self):
        # Given
        n_qubits = 4
        static_entangler = XX
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
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 0, 2
        assert ent_layer.operations[1].qubit_indices == (0, 2)
        # XX on 0, 3
        assert ent_layer.operations[2].qubit_indices == (0, 3)
        # XX on 1, 2
        assert ent_layer.operations[3].qubit_indices == (1, 2)
        # XX on 1, 3
        assert ent_layer.operations[4].qubit_indices == (1, 3)
        # XX on 2, 3
        assert ent_layer.operations[5].qubit_indices == (2, 3)

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
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 1, 2
        assert ent_layer.operations[1].qubit_indices == (1, 2)
        # XX on 2, 3
        assert ent_layer.operations[2].qubit_indices == (2, 3)

    def test_get_entangling_layer_star_topology(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "star"
        center_qubit = 1
        params = np.asarray([0, 0, 0])

        # When
        ent_layer = get_entangling_layer_star_topology(
            params, n_qubits, static_entangler, center_qubit
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 1, 2
        assert ent_layer.operations[1].qubit_indices == (1, 2)
        # XX on 2, 3
        assert ent_layer.operations[2].qubit_indices == (1, 3)

    def test_get_entangling_layer_graph_topology_matrix1(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = np.asarray(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        params = np.asarray([0, 0])

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params, n_qubits, static_entangler, adjacency_matrix=connectivity
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 2, 3
        assert ent_layer.operations[1].qubit_indices == (2, 3)

    def test_get_entangling_layer_graph_topology_graph1(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = adjacency_list_to_matrix(n_qubits, np.array([[1, 0], [3, 2]]))
        params = np.asarray([0, 0])

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params, n_qubits, static_entangler, adjacency_matrix=connectivity
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 2, 3
        assert ent_layer.operations[1].qubit_indices == (2, 3)

    def test_get_entangling_layer_graph_topology_matrix2(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = np.asarray(
            [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        params = np.asarray([0, 0, 0])

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params, n_qubits, static_entangler, adjacency_matrix=connectivity
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 1, 2
        assert ent_layer.operations[1].qubit_indices == (0, 3)
        # XX on 2, 3
        assert ent_layer.operations[2].qubit_indices == (2, 3)

    def test_get_entangling_layer_graph_topology_graph2(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = adjacency_list_to_matrix(
            n_qubits, np.array([[0, 1], [2, 3], [0, 3]])
        )
        params = np.asarray([0, 0, 0])

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params, n_qubits, static_entangler, adjacency_matrix=connectivity
        )

        # Then
        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        # XX on 0, 1
        assert ent_layer.operations[0].qubit_indices == (0, 1)
        # XX on 1, 2
        assert ent_layer.operations[1].qubit_indices == (0, 3)
        # XX on 2, 3
        assert ent_layer.operations[2].qubit_indices == (2, 3)

    def test_get_entangling_layer_graph_topology_bad_matrix1(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = np.asarray([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        params = np.asarray([0, 0])

        # When
        self.assertRaises(
            Exception,
            lambda: get_entangling_layer_graph_topology(
                params, n_qubits, static_entangler, adjacency_matrix=connectivity
            ),
        )

    def test_get_entangling_layer_graph_topology_bad_matrix2(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "graph"
        connectivity = np.asarray([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        params = np.asarray([0, 0])

        # When
        self.assertRaises(
            Exception,
            lambda: get_entangling_layer_graph_topology(
                params, n_qubits, static_entangler, adjacency_matrix=connectivity
            ),
        )

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
