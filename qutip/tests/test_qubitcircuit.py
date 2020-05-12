# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
from numpy.testing import assert_raises, assert_, assert_allclose, run_module_suite
from qutip.qip.operations.gates import (
    gate_sequence_product, rx)
from qutip.operators import identity
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.tensor import tensor
from qutip.qobj import Qobj, ptrace
from qutip.random_objects import rand_dm
from qutip.states import fock_dm


class TestQubitCircuit:
    """
    A test class for the QuTiP functions for Circuit resolution.
    """

    def testSWAPtoCNOT(self):
        """
        SWAP to CNOT: compare unitary matrix for SWAP and product of
        resolved matrices in terms of CNOT.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("SWAP", targets=[0, 1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="CNOT")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testISWAPtoCNOT(self):
        """
        ISWAP to CNOT: compare unitary matrix for ISWAP and product of
        resolved matrices in terms of CNOT.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("ISWAP", targets=[0, 1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="CNOT")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testCSIGNtoCNOT(self):
        """
        CSIGN to CNOT: compare unitary matrix for CSIGN and product of
        resolved matrices in terms of CNOT.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("CSIGN", targets=[1], controls=[0])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="CNOT")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testCNOTtoCSIGN(self):
        """
        CNOT to CSIGN: compare unitary matrix for CNOT and product of
        resolved matrices in terms of CSIGN.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("CNOT", targets=[0], controls=[1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="CSIGN")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testCNOTtoSQRTSWAP(self):
        """
        CNOT to SQRTSWAP: compare unitary matrix for CNOT and product of
        resolved matrices in terms of SQRTSWAP.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("CNOT", targets=[0], controls=[1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="SQRTSWAP")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testCNOTtoSQRTISWAP(self):
        """
        CNOT to SQRTISWAP: compare unitary matrix for CNOT and product of
        resolved matrices in terms of SQRTISWAP.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("CNOT", targets=[0], controls=[1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="SQRTISWAP")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testCNOTtoISWAP(self):
        """
        CNOT to ISWAP: compare unitary matrix for CNOT and product of
        resolved matrices in terms of ISWAP.
        """
        qc1 = QubitCircuit(2)
        qc1.add_gate("CNOT", targets=[0], controls=[1])
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis="ISWAP")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testSNOTdecompose(self):
        """
        SNOT to rotation: compare unitary matrix for SNOT and product of
        resolved matrices in terms of rotation gates.
        """
        qc1 = QubitCircuit(1)
        qc1.add_gate("SNOT", targets=0)
        U1 = gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates()
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def testadjacentgates(self):
        """
        Adjacent Gates: compare unitary matrix for ISWAP and product of
        resolved matrices in terms of adjacent gates interaction.
        """
        qc1 = QubitCircuit(3)
        qc1.add_gate("ISWAP", targets=[0, 2])
        U1 = gate_sequence_product(qc1.propagators())
        qc0 = qc1.adjacent_gates()
        qc2 = qc0.resolve_gates(basis="ISWAP")
        U2 = gate_sequence_product(qc2.propagators())
        assert_((U1 - U2).norm() < 1e-12)

    def test_add_gate(self):
        """
        Addition of a gate object directly to a `QubitCircuit`
        """
        qc = QubitCircuit(6)
        qc.add_gate("CNOT", targets=[1], controls=[0])
        test_gate = Gate("RZ", targets=[1], arg_value=1.570796,
                         arg_label="P")
        qc.add_gate(test_gate)
        qc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
        qc.add_gate("SNOT", targets=[3])
        qc.add_gate(test_gate, index=[3])
        qc.add_1q_gate("RY", start=4, end=5, arg_value=1.570796)

        # Test explicit gate addition
        assert_(qc.gates[0].name == "CNOT")
        assert_(qc.gates[0].targets == [1])
        assert_(qc.gates[0].controls == [0])

        # Test direct gate addition
        assert_(qc.gates[1].name == test_gate.name)
        assert_(qc.gates[1].targets == test_gate.targets)
        assert_(qc.gates[1].controls == test_gate.controls)

        # Test specified position gate addition
        assert_(qc.gates[3].name == test_gate.name)
        assert_(qc.gates[3].targets == test_gate.targets)
        assert_(qc.gates[3].controls == test_gate.controls)

        # Test adding 1 qubit gate on [start, end] qubits
        assert_(qc.gates[5].name == "RY")
        assert_(qc.gates[5].targets == [4])
        assert_(qc.gates[6].name == "RY")
        assert_(qc.gates[6].targets == [5])

    def test_add_state(self):
        """
        Addition of input and output states to a circuit.
        """
        qc = QubitCircuit(3)

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        assert_(qc.input_states[0] == "0")
        assert_(qc.input_states[2] is None)
        assert_(qc.output_states[1] == "+")

        qc1 = QubitCircuit(10)

        qc1.add_state("0", targets=[2, 3, 5, 6])
        qc1.add_state("+", targets=[1, 4, 9])
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("beta", targets=[0], state_type="output")
        assert_(qc1.input_states[0] is None)

        assert_(qc1.input_states[2] == "0")
        assert_(qc1.input_states[3] == "0")
        assert_(qc1.input_states[6] == "0")
        assert_(qc1.input_states[1] == "+")
        assert_(qc1.input_states[4] == "+")

        assert_(qc1.output_states[2] is None)
        assert_(qc1.output_states[1] == "A")
        assert_(qc1.output_states[4] == "A")
        assert_(qc1.output_states[9] == "A")

        assert_(qc1.output_states[0] == "beta")

    def test_exceptions(self):
        """
        Text exceptions are thrown correctly for inadequate inputs
        """
        qc = QubitCircuit(2)
        for gate in ["X", "Y", "Z", "S", "T"]:
            assert_raises(ValueError, qc.add_gate, gate,
                          targets=[1], controls=[0])

        for gate in ["CY", "CZ", "CS", "CT"]:
            assert_raises(ValueError, qc.add_gate, gate,
                          targets=[1])
            assert_raises(ValueError, qc.add_gate, gate)


    def test_single_qubit_gates(self):
        """
        Text single qubit gates are added correctly
        """
        qc = QubitCircuit(3)

        qc.add_gate("X", targets=[0])
        qc.add_gate("CY", targets=[1], controls=[0])
        qc.add_gate("Y", targets=[2])
        qc.add_gate("CS", targets=[0], controls=[1])
        qc.add_gate("Z", targets=[1])
        qc.add_gate("CT", targets=[2], controls=[2])
        qc.add_gate("CZ", targets=[0], controls=[0])
        qc.add_gate("S", targets=[1])
        qc.add_gate("T", targets=[2])

        assert_(qc.gates[8].name == "T")
        assert_(qc.gates[7].name == "S")
        assert_(qc.gates[6].name == "CZ")
        assert_(qc.gates[5].name == "CT")
        assert_(qc.gates[4].name == "Z")
        assert_(qc.gates[3].name == "CS")
        assert_(qc.gates[2].name == "Y")
        assert_(qc.gates[1].name == "CY")
        assert_(qc.gates[0].name == "X")

        assert_(qc.gates[8].targets == [2])
        assert_(qc.gates[7].targets == [1])
        assert_(qc.gates[6].targets == [0])
        assert_(qc.gates[5].targets == [2])
        assert_(qc.gates[4].targets == [1])
        assert_(qc.gates[3].targets == [0])
        assert_(qc.gates[2].targets == [2])
        assert_(qc.gates[1].targets == [1])
        assert_(qc.gates[0].targets == [0])

        assert_(qc.gates[6].controls == [0])
        assert_(qc.gates[5].controls == [2])
        assert_(qc.gates[3].controls == [1])
        assert_(qc.gates[1].controls == [0])


    def test_reverse(self):
        """
        Reverse a quantum circuit
        """
        qc = QubitCircuit(3)

        qc.add_gate("RX", targets=[0], arg_value=3.141,
                    arg_label=r"\pi/2")
        qc.add_gate("CNOT", targets=[1], controls=[0])
        qc.add_gate("SNOT", targets=[2])
        # Keep input output same

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        qc.reverse_circuit()

        assert_(qc.gates[2].name == "SNOT")
        assert_(qc.gates[1].name == "CNOT")
        assert_(qc.gates[0].name == "RX")

        assert_(qc.input_states[0] == "0")
        assert_(qc.input_states[2] is None)
        assert_(qc.output_states[1] == "+")


    def test_user_gate(self):
        """
        User defined gate for QubitCircuit
        """
        def customer_gate1(arg_values):
            mat = np.zeros((4, 4), dtype=np.complex)
            mat[0, 0] = mat[1, 1] = 1.
            mat[2:4, 2:4] = rx(arg_values)
            return Qobj(mat, dims=[[2, 2], [2, 2]])

        def customer_gate2():
            mat = np.array([[1., 0],
                            [0., 1.j]])
            return Qobj(mat, dims=[[2], [2]])

        qc = QubitCircuit(3)
        qc.user_gates = {"CTRLRX": customer_gate1,
                         "T1": customer_gate2}
        qc.add_gate("CTRLRX", targets=[1, 2], arg_value=np.pi/2)
        qc.add_gate("T1", targets=[1])
        props = qc.propagators()
        result1 = tensor(identity(2), customer_gate1(np.pi/2))
        assert_allclose(props[0], result1)
        result2 = tensor(identity(2), customer_gate2(), identity(2))
        assert_allclose(props[1], result2)

    def test_N_level_system(self):
        """
        Test for circuit with N-level system.
        """
        mat3 = rand_dm(3, density=1.)

        def controlled_mat3(arg_value):
            """
            A qubit control an operator acting on a 3 level system
            """
            control_value = arg_value
            dim = mat3.dims[0][0]
            return (tensor(fock_dm(2, control_value), mat3) +
                    tensor(fock_dm(2, 1 - control_value), identity(dim)))

        qc = QubitCircuit(2, dims=[3, 2])
        qc.user_gates = {"CTRLMAT3": controlled_mat3}
        qc.add_gate("CTRLMAT3", targets=[1, 0], arg_value=1)
        props = qc.propagators()
        assert_allclose(mat3, ptrace(props[0], 0) - 1)


if __name__ == "__main__":
    run_module_suite()
