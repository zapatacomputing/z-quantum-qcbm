import pytest
import numpy as np
from zquantum.core.circuits import XX, RX

from zquantum.qcbm.ansatz_utils import (
    get_entangling_layer,
    get_entangling_layer_all_topology,
    get_entangling_layer_line_topology,
    get_entangling_layer_star_topology,
    get_entangling_layer_graph_topology,
    adjacency_list_to_matrix,
)

from zquantum.qcbm.ansatz import QCBMAnsatz


class TestAnsatzUtils:
    @pytest.mark.parametrize(
        "n_qubits,expected", [(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])]
    )
    def test_get_entangling_layer_all_topology(self, n_qubits, expected):
        # Given
        static_entangler = XX
        topology = "all"
        params = np.asarray([0] * int(n_qubits * (n_qubits - 1) / 2))

        # When
        ent_layer = get_entangling_layer_all_topology(
            params, n_qubits, static_entangler
        )

        # Then
        for operation in ent_layer.operations:
            assert operation.gate.name == "XX"

        for i in range(len(params)):
            assert ent_layer.operations[i].qubit_indices == expected[i]

    @pytest.mark.parametrize("n_qubits,expected", [(4, [(0, 1), (1, 2), (2, 3)])])
    def test_get_entangling_layer_line_topology(self, n_qubits, expected):
        # Given
        static_entangler = XX
        topology = "line"
        params = np.asarray([0] * int(n_qubits - 1))

        # When
        ent_layer = get_entangling_layer_line_topology(
            params, n_qubits, static_entangler
        )

        for operation in ent_layer.operations:
            assert operation.gate.name == "XX"

        for i in range(len(params)):
            assert ent_layer.operations[i].qubit_indices == expected[i]

    @pytest.mark.parametrize("n_qubits,expected", [(4, [(0, 1), (1, 2), (1, 3)])])
    def test_get_entangling_layer_star_topology(self, n_qubits, expected):
        # Given
        single_qubit_gate = RX
        static_entangler = XX
        topology = "star"
        center_qubit = 1
        params = np.asarray([0] * int(n_qubits - 1))

        # When
        ent_layer = get_entangling_layer_star_topology(
            params, n_qubits, static_entangler, center_qubit
        )

        # Then
        for operation in ent_layer.operations:
            assert operation.gate.name == "XX"

        for i in range(len(params)):
            assert ent_layer.operations[i].qubit_indices == expected[i]

    @pytest.mark.parametrize(
        "n_qubits,matrix,expected,n_connections",
        [
            (
                4,
                np.asarray([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
                [(0, 1), (2, 3)],
                2,
            ),
            (
                4,
                np.asarray([[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
                [(0, 1), (0, 3), (2, 3)],
                3,
            ),
        ],
    )
    def test_get_entangling_layer_graph_topology_matrix(
        self, n_qubits, matrix, expected, n_connections
    ):
        # Given
        n_qubits = 4
        single_qubit_gate = RX
        static_entangler = XX
        topology = "graph"
        params = np.asarray([0] * n_connections)

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params, n_qubits, static_entangler, adjacency_matrix=matrix
        )

        # Then
        for operation in ent_layer.operations:
            assert operation.gate.name == "XX"

        for i in range(len(params)):
            assert ent_layer.operations[i].qubit_indices == expected[i]

    @pytest.mark.parametrize(
        "n_qubits,matrix,expected,n_connections",
        [
            (
                4,
                np.array([[1, 0], [3, 2]]),
                [(0, 1), (2, 3)],
                2,
            ),
            (4, np.array([[0, 1], [2, 3], [0, 3]]), [(0, 1), (0, 3), (2, 3)], 3),
        ],
    )
    def test_get_entangling_layer_graph_topology_graph(
        self, n_qubits, matrix, expected, n_connections
    ):
        # Given
        single_qubit_gate = RX
        static_entangler = XX
        topology = "graph"
        params = np.asarray([0] * n_connections)

        # When
        ent_layer = get_entangling_layer_graph_topology(
            params,
            n_qubits,
            static_entangler,
            adjacency_matrix=adjacency_list_to_matrix(n_qubits, matrix),
        )

        # Then
        for operation in ent_layer.operations:
            assert operation.gate.name == "XX"

        for i in range(len(params)):
            assert ent_layer.operations[i].qubit_indices == expected[i]

    @pytest.mark.parametrize(
        "n_qubits,matrix,n_connections",
        [
            (
                4,
                np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
                2,
            ),
            (4, np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]), 2),
        ],
    )
    def test_get_entangling_layer_graph_topology_wrong_matrix(
        self, n_qubits, matrix, n_connections
    ):
        # Given
        single_qubit_gate = RX
        static_entangler = XX
        topology = "graph"

        params = np.asarray([0] * n_connections)
        # When
        with pytest.raises(Exception):
            _ = get_entangling_layer(
                params, n_qubits, static_entangler, topology, matrix
            )

    def test_get_entangling_layer_topology_supported(self):
        # Given
        n_qubits_list = [2, 3, 4, 5]
        static_entangler = XX
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
            assert all_topology_layer == entangling_layer

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
            assert line_topology_layer == entangling_layer

    def test_get_entangling_layer_toplogy_not_supported(self):
        # Given
        n_qubits = 2
        static_entangler = XX
        topology = "NOT SUPPORTED"
        params = np.zeros(1)

        # When
        with pytest.raises(RuntimeError):
            _ = get_entangling_layer(params, n_qubits, static_entangler, topology)

    @pytest.mark.parametrize("type,n_qubits", [("line", 5), ("star", 5)])
    def test_qubit_count_topology(self, type, n_qubits):
        test_ansatz = QCBMAnsatz(1, n_qubits, type)
        assert test_ansatz.n_params_per_ent_layer == n_qubits - 1

    @pytest.mark.parametrize(
        "n_qubits,matrix,n_connections",
        [(4, np.asarray([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]), 3)],
    )
    def test_qubit_count_matrix_params(self, n_qubits, matrix, n_connections):
        test_ansatz = QCBMAnsatz(1, n_qubits, "graph", adjacency_matrix=matrix)
        assert test_ansatz.n_params_per_ent_layer == n_connections

    @pytest.mark.parametrize(
        "n_qubits,list,n_connections",
        [(4, np.asarray([[[0, 1], [2, 3], [0, 3]]]), 3)],
    )
    def test_qubit_count_list_params(self, n_qubits, list, n_connections):
        # Given
        test_ansatz = QCBMAnsatz(
            1,
            n_qubits,
            "graph",
            adjacency_list=list,
        )
        # When
        assert test_ansatz.n_params_per_ent_layer == n_connections

    @pytest.mark.parametrize("n_qubits", [4])
    def test_default_star_qubit(self, n_qubits):
        default_layer = get_entangling_layer(
            np.zeros(n_qubits - 1), n_qubits, XX, "star"
        )
        for i in range(n_qubits - 1):
            assert default_layer.operations[i].qubit_indices == (0, i + 1)

    def test_get_entangling_layers_fails_with_incorrect_graph_kwargs(self):
        # Given
        n_qubits = 4
        single_qubit_gate = RX
        static_entangler = XX
        topology = "graph"
        connectivity = np.asarray(
            [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        connectivity2 = np.asarray([[[0, 1], [2, 3], [0, 3]]])
        params = np.asarray([0, 0, 0])

        with pytest.raises(TypeError):
            _ = get_entangling_layer(
                params,
                n_qubits,
                static_entangler,
                topology,
                connectivity,
                connectivity2,
            )
