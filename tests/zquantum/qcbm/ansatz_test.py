import pytest
import numpy as np

from zquantum.core.wip.circuits import Circuit, RX, RZ, XX
from zquantum.core.interfaces.ansatz_test import AnsatzTests

from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.ansatz_utils import get_entangling_layer


class TestQCBMAnsatz(AnsatzTests):
    @pytest.fixture
    def n_qubits(self):
        return 4

    @pytest.fixture
    def topology(self):
        return "all"

    @pytest.fixture
    def ansatz(self, n_qubits, topology):
        return QCBMAnsatz(
            number_of_layers=2,
            number_of_qubits=n_qubits,
            topology=topology,
        )

    def test_get_executable_circuit_too_many_parameters(self, n_qubits, topology):
        # Given
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
        ]
        params = np.concatenate(params)
        ansatz = QCBMAnsatz(
            number_of_layers=2,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        # When/Then
        with pytest.raises(ValueError):
            ansatz.get_executable_circuit(params),

    def test_ansatz_circuit_one_layer(self, n_qubits, topology):
        # Given
        number_of_layers = 1
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )

        params = [np.random.rand(n_qubits)]

        expected_circuit = Circuit()
        for i in range(n_qubits):
            expected_circuit += RX(params[0][i])(i)

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_two_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 2
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )

        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_three_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 3
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
        ]
        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RZ(params[2][i])(i) for i in range(n_qubits)],
                *[RX(params[2][i + n_qubits])(i) for i in range(n_qubits)],
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_four_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 4
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )

        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[2][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_five_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 5
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
        ]
        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[2][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
                # Fifth layer
                *[RZ(params[4][i])(i) for i in range(n_qubits)],
                *[RX(params[4][i + n_qubits])(i) for i in range(n_qubits)],
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_six_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 6
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
                # Fifth layer
                *[RX(params[4][i])(i) for i in range(n_qubits)],
                *[RZ(params[4][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[4][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Sixth layer
                *get_entangling_layer(params[5], n_qubits, XX, topology).operations,
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_seven_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 7
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
        ]

        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
                # Fifth layer
                *[RX(params[4][i])(i) for i in range(n_qubits)],
                *[RZ(params[4][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[4][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Sixth layer
                *get_entangling_layer(params[5], n_qubits, XX, topology).operations,
                # Seventh layer
                *[RZ(params[6][i])(i) for i in range(n_qubits)],
                *[RX(params[6][i + n_qubits])(i) for i in range(n_qubits)],
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_eight_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 8
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
        ]

        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
                # Fifth layer
                *[RX(params[4][i])(i) for i in range(n_qubits)],
                *[RZ(params[4][i + n_qubits])(i) for i in range(n_qubits)],
                # Sixth layer
                *get_entangling_layer(params[5], n_qubits, XX, topology).operations,
                # Seventh layer
                *[RX(params[6][i])(i) for i in range(n_qubits)],
                *[RZ(params[6][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[6][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Eigth layer
                *get_entangling_layer(params[7], n_qubits, XX, topology).operations,
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit

    def test_ansatz_circuit_nine_layers(self, n_qubits, topology):
        # Given
        number_of_layers = 9
        ansatz = QCBMAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=n_qubits,
            topology=topology,
        )
        params = [
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(3 * n_qubits),
            np.random.rand(int((n_qubits * (n_qubits - 1)) / 2)),
            np.random.rand(2 * n_qubits),
        ]
        expected_circuit = Circuit(
            [
                # First layer
                *[RX(params[0][i])(i) for i in range(n_qubits)],
                *[RZ(params[0][i + n_qubits])(i) for i in range(n_qubits)],
                # Second layer
                *get_entangling_layer(params[1], n_qubits, XX, topology).operations,
                # Third layer
                *[RX(params[2][i])(i) for i in range(n_qubits)],
                *[RZ(params[2][i + n_qubits])(i) for i in range(n_qubits)],
                # Fouth layer
                *get_entangling_layer(params[3], n_qubits, XX, topology).operations,
                # Fifth layer
                *[RX(params[4][i])(i) for i in range(n_qubits)],
                *[RZ(params[4][i + n_qubits])(i) for i in range(n_qubits)],
                # Sixth layer
                *get_entangling_layer(params[5], n_qubits, XX, topology).operations,
                # Seventh layer
                *[RX(params[6][i])(i) for i in range(n_qubits)],
                *[RZ(params[6][i + n_qubits])(i) for i in range(n_qubits)],
                *[RX(params[6][i + 2 * n_qubits])(i) for i in range(n_qubits)],
                # Eigth layer
                *get_entangling_layer(params[7], n_qubits, XX, topology).operations,
                # Ningth layer
                *[RZ(params[8][i])(i) for i in range(n_qubits)],
                *[RX(params[8][i + n_qubits])(i) for i in range(n_qubits)],
            ]
        )

        params = np.concatenate(params)

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert circuit == expected_circuit
