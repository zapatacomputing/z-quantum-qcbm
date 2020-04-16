import unittest
import numpy as np
import cirq
from pyquil import Program
from pyquil.gates import CZ, RY

from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.utils import compare_unitary

from .ansatz import ( build_qcbm_circuit_ion_trap, get_single_qubit_layer,
    get_all_topology, get_entangling_layer, generate_random_initial_params )


class TestQCBMAnsatz(unittest.TestCase):

    def test_single_qubit_layer(self):
        single_qubit_gate = "Ry"

        n_qubits = [2,3,4,10]
        
        for n in n_qubits:
            params = [x for x in range(0, n)]
            
            circ = get_single_qubit_layer(n, params, single_qubit_gate)
            self.assertEqual(circ.n_multiqubit_gates, 0)
            
            unitary = circ.to_cirq()._unitary_()
            
            #cirq
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n)]
            for i in range(0,n):
                test.append(cirq.Ry(params[i]).on(qubits[i]))
            u_cirq = test._unitary_()
            
            #pyquil
            qprogram = Program()
            for i in range(0, n):
                qprogram += RY(params[i],i)
            u_pyquil=Circuit(qprogram).to_cirq()._unitary_()

            self.assertEqual(compare_unitary(unitary, u_cirq, tol=1e-10), True)
            self.assertEqual(compare_unitary(unitary, u_pyquil, tol=1e-10), True)

    def test_all_topology(self):
        n_qubits = [2,3,4,5]
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"
        
        for n in n_qubits:
            params = [x for x in range (0,int((n*(n-1))/2))]
            
            circ = get_all_topology(n, np.array(params), static_entangler)
            unitary = circ.to_cirq()._unitary_()
            
            circ2 = get_entangling_layer(n, np.array(params), static_entangler, single_qubit_gate, topology)
            unitary2 = circ2.to_cirq()._unitary_()
            
            #cirq
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n)]
            k=0
            for i in range(0,n-1):
                for j in range(i+1,n):
                    test.append(cirq.MS(params[k]).on(qubits[i], qubits[j]))
                    k+=1
            u_cirq=test._unitary_()

            self.assertEqual(compare_unitary(unitary,u_cirq, tol=1e-10),True)
            self.assertEqual(compare_unitary(unitary2,u_cirq, tol=1e-10),True)
            self.assertEqual(compare_unitary(unitary,unitary2, tol=1e-10),True)

    def test_get_entangling_layer(self):
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"
        topology = "all"
        params = np.asarray([0,0,0,0,0,0])

        ent_layer = get_entangling_layer(n_qubits, params, static_entangler, single_qubit_gate, topology)

        for gate in ent_layer.gates:
            self.assertTrue(gate.name, "XX")

        self.assertEqual(ent_layer.gates[0].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[0].qubits[1].index, 1)
        self.assertEqual(ent_layer.gates[1].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[1].qubits[1].index, 2)
        self.assertEqual(ent_layer.gates[2].qubits[0].index, 0)
        self.assertEqual(ent_layer.gates[2].qubits[1].index, 3)
        self.assertEqual(ent_layer.gates[3].qubits[0].index, 1)
        self.assertEqual(ent_layer.gates[3].qubits[1].index, 2)
        self.assertEqual(ent_layer.gates[4].qubits[0].index, 1)
        self.assertEqual(ent_layer.gates[4].qubits[1].index, 3)
        self.assertEqual(ent_layer.gates[5].qubits[0].index, 2)
        self.assertEqual(ent_layer.gates[5].qubits[1].index, 3)

    def test_build_qcbm_circuit_iontrap (self):
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"

        for topology in ["all"]:
            entanglers_per_layer = int((n_qubits*(n_qubits-1))/2)
            n_params_per_layer = int((n_qubits*(n_qubits-1))/2)
            params = np.array([0, np.pi/2, np.pi, 2*np.pi,0, np.pi/2, np.pi, 2*np.pi,5,6,6,5,4,3])
            n_layers = int((params.shape[0]-2*n_qubits)/n_params_per_layer)

            circuit = build_qcbm_circuit_ion_trap(n_qubits,params, single_qubit_gate, static_entangler,topology)
            self.assertEqual(circuit.n_multiqubit_gates, n_layers*(entanglers_per_layer))
            self.assertEqual(len(circuit.qubits), n_qubits)

    def test_generate_random_params (self):
        n_qubits = 4
        single_qubit_gate = "Rx"
        static_entangler = "XX"

        for topology in ["all"]:
            entanglers_per_layer = int((n_qubits*(n_qubits-1))/2)
            n_params_per_layer = int((n_qubits*(n_qubits-1))/2)
            params=np.array(generate_random_initial_params(n_qubits,topology=topology))
            n_layers = int((params.shape[0]-2*n_qubits)/n_params_per_layer)

        circuit = build_qcbm_circuit_ion_trap(n_qubits,params, single_qubit_gate, static_entangler,topology)
        self.assertEqual(circuit.n_multiqubit_gates, n_layers*(entanglers_per_layer))
        self.assertEqual(len(circuit.qubits), n_qubits)