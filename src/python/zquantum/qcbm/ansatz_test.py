import unittest
import numpy as np
import cirq
import itertools
from pyquil import Program
import pyquil.gates

from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.utils import compare_unitary, RNDSEED

from .ansatz import ( get_qcbm_ansatz, build_qcbm_circuit_ion_trap, get_single_qubit_layer,
    get_all_topology, get_line_topology, get_entangling_layer, generate_random_initial_params )


class TestQCBMAnsatz(unittest.TestCase):

    def test_get_qcbm_ansatz(self):
        # Given
        n_qubits = 5
        n_layers = 2
        topology = 'all'
        target_ansatz = {'ansatz_type': 'QCMB_ion_trap',
            'ansatz_module': 'zquantum.qcbm.ansatz',
            'ansatz_func' : 'build_qcbm_circuit_ion_trap',
            'ansatz_kwargs' : {
                'n_qubits' : n_qubits,
                'n_layers' : n_layers,
                'topology' : topology}}
        
        # When
        ansatz = get_qcbm_ansatz(n_qubits, n_layers, topology)

        # Then
        self.assertEqual(ansatz, target_ansatz)

    def test_get_single_qubit_layer_wrong_num_params(self):
        # Given
        single_qubit_gates = ["Ry"]
        n_qubits = 2
        params = np.ones(3)
        # When/Then
        self.assertRaises(AssertionError, lambda: get_single_qubit_layer(params, n_qubits, single_qubit_gates))

    def test_get_single_qubit_layer_one_gate(self):
        # Given
        single_qubit_gates = ["Ry"]
        n_qubits_list = [2,3,4,10]
        
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
        n_qubits_list = [2,3,4,10]
        
        for n_qubits in n_qubits_list:
            # Given
            params = [x for x in range(0, 3*n_qubits)]
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n_qubits)]
            for i in range(0, n_qubits):
                test.append(cirq.Ry(params[i]).on(qubits[i]))
            for i in range(0, n_qubits):
                test.append(cirq.Rx(params[n_qubits + i]).on(qubits[i]))
            for i in range(0, n_qubits):
                test.append(cirq.Rz(params[2*n_qubits + i]).on(qubits[i]))
            u_cirq = test._unitary_()
            
            # When
            circ = get_single_qubit_layer(params, n_qubits, single_qubit_gates)
            unitary = circ.to_cirq()._unitary_()

            # Then
            self.assertEqual(circ.n_multiqubit_gates, 0)
            self.assertEqual(compare_unitary(unitary, u_cirq, tol=1e-10), True)

    def test_get_all_topology(self):
        # Given
        n_qubits = 4
        static_entangler = "XX"
        topology = "all"
        params = np.asarray([0,0,0,0,0,0])

        # When
        ent_layer = get_all_topology(params, n_qubits, static_entangler)

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

    def test_get_line_topology(self):
        # Given
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "line"
        params = np.asarray([0,0,0])

        # When
        ent_layer = get_line_topology(params, n_qubits, static_entangler)

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
        n_qubits_list = [2,3,4,5]
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"
        
        for n_qubits in n_qubits_list:
            # Given
            params = np.zeros((int((n_qubits*(n_qubits-1))/2)))
            all_topology_layer = get_all_topology(params, n_qubits, static_entangler)
            # When
            entangling_layer = get_entangling_layer(params, n_qubits, static_entangler, topology)
            # Then
            self.assertEqual(all_topology_layer, entangling_layer)

        # Given
        topology = "line"
        
        for n_qubits in n_qubits_list:
            # Given
            params = np.zeros((n_qubits-1))
            line_topology_layer = get_line_topology(params, n_qubits, static_entangler)
            # When
            entangling_layer = get_entangling_layer(params, n_qubits, static_entangler, topology)
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
        self.assertRaises(RuntimeError, lambda: get_entangling_layer(params, n_qubits, static_entangler, topology))

    def test_build_qcbm_circuit_iontrap_too_many_parameters(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits)]
        params = np.concatenate(params)
        topology = "all"
        n_layers = 2

        # When/Then
        self.assertRaises(RuntimeError, lambda: build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology))


    def test_build_qcbm_circuit_iontrap_one_layer(self):
        # Given
        n_qubits = 4
        params = [np.ones(n_qubits)]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        expected_circuit = Circuit(expected_pycircuit)

        params = np.concatenate(params)
        n_layers = 1

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_two_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2))]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_circuit = Circuit(expected_pycircuit)
        expected_circuit += get_entangling_layer(params[1], n_qubits, "XX", topology)
        

        params = np.concatenate(params)
        n_layers = 2

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_three_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits)]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i+n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer
        

        params = np.concatenate(params)
        n_layers = 3

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_four_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2))]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i+2*n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer
        

        params = np.concatenate(params)
        n_layers = 4

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_five_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits)]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i+2*n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i+n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer + expected_fifth_layer
        

        params = np.concatenate(params)
        n_layers = 5

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_six_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2))]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i+2*n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer + expected_fifth_layer + expected_sixth_layer
        

        params = np.concatenate(params)
        n_layers = 6

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_seven_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits)]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i+2*n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i+n_qubits], i))
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer + expected_fifth_layer + expected_sixth_layer + expected_seventh_layer
        

        params = np.concatenate(params)
        n_layers = 7

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)
        

    def test_build_qcbm_circuit_iontrap_eight_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2))]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i+n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i+2*n_qubits], i))
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(params[7], n_qubits, "XX", topology)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer + expected_fifth_layer + expected_sixth_layer + expected_seventh_layer + expected_eigth_layer
        

        params = np.concatenate(params)
        n_layers = 8

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)

    def test_build_qcbm_circuit_iontrap_nine_layers(self):
        # Given
        n_qubits = 4
        params = [np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(3*n_qubits), np.ones(int((n_qubits*(n_qubits-1))/2)), np.ones(2*n_qubits)]
        topology = "all"

        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[0][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[0][i+n_qubits], i))
        expected_first_layer = Circuit(expected_pycircuit)
        expected_second_layer = get_entangling_layer(params[1], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[2][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[2][i+n_qubits], i))
        expected_third_layer = Circuit(expected_pycircuit)
        expected_fourth_layer = get_entangling_layer(params[3], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[4][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[4][i+n_qubits], i))
        expected_fifth_layer = Circuit(expected_pycircuit)
        expected_sixth_layer = get_entangling_layer(params[5], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[6][i+n_qubits], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[6][i+2*n_qubits], i))
        expected_seventh_layer = Circuit(expected_pycircuit)
        expected_eigth_layer = get_entangling_layer(params[7], n_qubits, "XX", topology)
        expected_pycircuit = Program()
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RZ(params[8][i], i))
        for i in range(n_qubits):
            expected_pycircuit += Program(pyquil.gates.RX(params[8][i+n_qubits], i))
        expected_ninth_layer = Circuit(expected_pycircuit)
        expected_circuit = expected_first_layer + expected_second_layer + expected_third_layer + expected_fourth_layer + expected_fifth_layer + expected_sixth_layer + expected_seventh_layer + expected_eigth_layer + expected_ninth_layer
        

        params = np.concatenate(params)
        n_layers = 9

        # When
        circuit = build_qcbm_circuit_ion_trap(params, n_qubits, n_layers, topology)

        # Then
        self.assertEqual(circuit, expected_circuit)

    def test_generate_random_params_all_toplogy(self):
        # Given
        n_qubits = 4
        n_layers = 2
        topology = "all"

        # When
        params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

        # Then
        self.assertEqual(len(params), 2*n_qubits+int((n_qubits*(n_qubits-1))/2))
        

    def test_generate_random_params_line_toplogy(self):
        # Given
        n_qubits = 4
        n_layers = 2
        topology = "line"

        # When
        params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

        # Then
        self.assertEqual(len(params), 2*n_qubits+n_qubits-1)
        

    def test_generate_random_params_toplogy_not_supported(self):
        # Given
        n_qubits = 4
        n_layers = 2
        topology = "NOT SUPPORTED"

        # When/Then
        self.assertRaises(RuntimeError, lambda: generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED))
        

    def test_generate_random_params_one_layer(self):
        # Given
        n_qubits = 4
        topology = "line"
        n_layers = 1

        # When
        params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

        # Then
        self.assertEqual(len(params), n_qubits)


    def test_generate_random_params_two_layers(self):
        # Given
        n_qubits = 4
        topology = "line"
        n_layers = 2

        # When
        params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

        # Then
        self.assertEqual(len(params), 2*n_qubits+n_qubits-1)


    def test_generate_random_params_three_layers(self):
        # Given
        n_qubits = 4
        topology = "line"
        n_layers = 3

        # When
        params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

        # Then
        self.assertEqual(len(params), 4*n_qubits+n_qubits-1)

    def test_generate_random_params_even_layers(self):
        # Given
        n_qubits = 4
        topology = "line"
        n_layers_list = [4, 6, 8, 10, 12]
        for n_layers in n_layers_list:
            expected_num_params = n_qubits*n_layers + (n_qubits-1)*int(n_layers/2) + n_qubits

            # When
            params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

            # Then
            self.assertEqual(len(params), expected_num_params)

    def test_generate_random_params_odd_layers(self):
        # Given
        n_qubits = 4
        topology = "line"
        n_layers_list = [5, 7, 9, 11]
        for n_layers in n_layers_list:
            expected_num_params = n_qubits*n_layers + (n_qubits-1)*int(n_layers/2) + 2*n_qubits

            # When
            params = generate_random_initial_params(n_qubits, n_layers, topology, seed=RNDSEED)

            # Then
            self.assertEqual(len(params), expected_num_params)