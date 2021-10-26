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
import pytest
import numpy as np
from qutip.qip.operations import gates
from qutip.operators import identity
from qutip.qip.circuit import (
    QubitCircuit, Gate, Measurement, _ctrl_gates, _single_qubit_gates,
    _swap_like, _toffoli_like, _fredkin_like, _para_gates)
from qutip import tensor, Qobj, ptrace, rand_ket, fock_dm, basis, rand_dm


def _op_dist(A, B):
    return (A - B).norm()


def _teleportation_circuit():
    teleportation = QubitCircuit(3, num_cbits=2,
                                input_states=["q0", "0", "0", "c0", "c1"])

    teleportation.add_gate("SNOT", targets=[1])
    teleportation.add_gate("CNOT", targets=[2], controls=[1])
    teleportation.add_gate("CNOT", targets=[1], controls=[0])
    teleportation.add_gate("SNOT", targets=[0])
    teleportation.add_measurement("M0", targets=[0], classical_store=1)
    teleportation.add_measurement("M1", targets=[1], classical_store=0)
    teleportation.add_gate("X", targets=[2], classical_controls=[0])
    teleportation.add_gate("Z", targets=[2], classical_controls=[1])

    return teleportation


class TestQubitCircuit:
    """
    A test class for the QuTiP functions for Circuit resolution.
    """

    @pytest.mark.parametrize(["gate_from", "gate_to", "targets", "controls"], [
                        pytest.param("SWAP", "CNOT",
                                        [0, 1], None, id="SWAPtoCNOT"),
                        pytest.param("ISWAP", "CNOT",
                                        [0, 1], None, id="ISWAPtoCNOT"),
                        pytest.param("CSIGN", "CNOT",
                                        [1], [0], id="CSIGNtoCNOT"),
                        pytest.param("CNOT", "CSIGN",
                                        [0], [1], id="CNOTtoCSIGN"),
                        pytest.param("CNOT", "SQRTSWAP",
                                        [0], [1], id="CNOTtoSQRTSWAP"),
                        pytest.param("CNOT", "SQRTISWAP",
                                        [0], [1], id="CNOTtoSQRTISWAP"),
                        pytest.param("CNOT", "ISWAP",
                                        [0], [1], id="CNOTtoISWAP")])
    def testresolve(self, gate_from, gate_to, targets, controls):
        qc1 = QubitCircuit(2)
        qc1.add_gate(gate_from, targets=targets, controls=controls)
        U1 = gates.gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates(basis=gate_to)
        U2 = gates.gate_sequence_product(qc2.propagators())
        assert _op_dist(U1, U2) < 1e-12

    def testSNOTdecompose(self):
        """
        SNOT to rotation: compare unitary matrix for SNOT and product of
        resolved matrices in terms of rotation gates.
        """
        qc1 = QubitCircuit(1)
        qc1.add_gate("SNOT", targets=0)
        U1 = gates.gate_sequence_product(qc1.propagators())
        qc2 = qc1.resolve_gates()
        U2 = gates.gate_sequence_product(qc2.propagators())
        assert _op_dist(U1, U2) < 1e-12

    def testadjacentgates(self):
        """
        Adjacent Gates: compare unitary matrix for ISWAP and product of
        resolved matrices in terms of adjacent gates interaction.
        """
        qc1 = QubitCircuit(3)
        qc1.add_gate("ISWAP", targets=[0, 2])
        U1 = gates.gate_sequence_product(qc1.propagators())
        qc0 = qc1.adjacent_gates()
        qc2 = qc0.resolve_gates(basis="ISWAP")
        U2 = gates.gate_sequence_product(qc2.propagators())
        assert _op_dist(U1, U2) < 1e-12

    def test_add_gate(self):
        """
        Addition of a gate object directly to a `QubitCircuit`
        """
        qc = QubitCircuit(6)
        qc.add_gate("CNOT", targets=[1], controls=[0])
        test_gate = Gate("SWAP", targets=[1, 4])
        qc.add_gate(test_gate)
        qc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
        qc.add_gate("SNOT", targets=[3])
        qc.add_gate(test_gate, index=[3])
        qc.add_1q_gate("RY", start=4, end=5, arg_value=1.570796)

        # Test explicit gate addition
        assert qc.gates[0].name == "CNOT"
        assert qc.gates[0].targets == [1]
        assert qc.gates[0].controls == [0]

        # Test direct gate addition
        assert qc.gates[1].name == test_gate.name
        assert qc.gates[1].targets == test_gate.targets

        # Test specified position gate addition
        assert qc.gates[3].name == test_gate.name
        assert qc.gates[3].targets == test_gate.targets

        # Test adding 1 qubit gate on [start, end] qubits
        assert qc.gates[5].name == "RY"
        assert qc.gates[5].targets == [4]
        assert qc.gates[5].arg_value == 1.570796
        assert qc.gates[6].name == "RY"
        assert qc.gates[6].targets == [5]
        assert qc.gates[5].arg_value == 1.570796

        # Test Exceptions  # Global phase is not included
        for gate in _single_qubit_gates:
            if gate not in _para_gates:
                # No target
                pytest.raises(ValueError, qc.add_gate, gate, None, None)
                # Multiple targets
                pytest.raises(ValueError, qc.add_gate, gate, [0, 1, 2], None)
                # With control
                pytest.raises(ValueError, qc.add_gate, gate, [0], [1])
            else:
                # No target
                pytest.raises(ValueError, qc.add_gate, gate, None, None, 1)
                # Multiple targets
                pytest.raises(ValueError, qc.add_gate, gate, [0, 1, 2], None, 1)
                # With control
                pytest.raises(ValueError, qc.add_gate, gate, [0], [1], 1)
        for gate in _ctrl_gates:
            if gate not in _para_gates:
                # No target
                pytest.raises(ValueError, qc.add_gate, gate, None, [1])
                # No control
                pytest.raises(ValueError, qc.add_gate, gate, [0], None)
            else:
                # No target
                pytest.raises(ValueError, qc.add_gate, gate, None, [1], 1)
                # No control
                pytest.raises(ValueError, qc.add_gate, gate, [0], None, 1)
        for gate in _swap_like:
            if gate not in _para_gates:
                # Single target
                pytest.raises(ValueError, qc.add_gate, gate, [0], None)
                # With control
                pytest.raises(ValueError, qc.add_gate, gate, [0, 1], [3])
            else:
                # Single target
                pytest.raises(ValueError, qc.add_gate, gate, [0], None, 1)
                # With control
                pytest.raises(ValueError, qc.add_gate, gate, [0, 1], [3], 1)
        for gate in _fredkin_like:
            # Single target
            pytest.raises(ValueError, qc.add_gate, gate, [0], [2])
            # No control
            pytest.raises(ValueError, qc.add_gate, gate, [0, 1], None)
        for gate in _toffoli_like:
            # No target
            pytest.raises(ValueError, qc.add_gate, gate, None, [1, 2])
            # Single control
            pytest.raises(ValueError, qc.add_gate, gate, [0], [1])

    def test_add_circuit(self):
        """
        Addition of a circuit to a `QubitCircuit`
        """
        qc = QubitCircuit(6)
        qc.add_gate("CNOT", targets=[1], controls=[0])
        test_gate = Gate("SWAP", targets=[1, 4])
        qc.add_gate(test_gate)
        qc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
        qc.add_gate("SNOT", targets=[3])
        qc.add_gate(test_gate, index=[3])
        qc.add_measurement("M0", targets=[0], classical_store=[1])
        qc.add_1q_gate("RY", start=4, end=5, arg_value=1.570796)

        qc1 = QubitCircuit(6)

        qc1.add_circuit(qc)

        # Test if all gates and measurements are added
        assert len(qc1.gates) == len(qc.gates)

        for i in range(len(qc1.gates)):
            assert (qc1.gates[i].name
                    == qc.gates[i].name)
            assert (qc1.gates[i].targets
                    == qc.gates[i].targets)
            if (isinstance(qc1.gates[i], Gate) and
                    isinstance(qc.gates[i], Gate)):
                assert (qc1.gates[i].controls
                        == qc.gates[i].controls)
                assert (qc1.gates[i].classical_controls
                        == qc.gates[i].classical_controls)
            elif (isinstance(qc1.gates[i], Measurement) and
                    isinstance(qc.gates[i], Measurement)):
                assert (qc1.gates[i].classical_store
                        == qc.gates[i].classical_store)

        # Test exception when qubit out of range
        pytest.raises(NotImplementedError, qc1.add_circuit, qc, start=4)

        qc2 = QubitCircuit(8)
        qc2.add_circuit(qc, start=2)

        # Test if all gates are added
        assert len(qc2.gates) == len(qc.gates)

        # Test if the positions are correct
        for i in range(len(qc2.gates)):
            if qc.gates[i].targets is not None:
                assert (qc2.gates[i].targets[0]
                        == qc.gates[i].targets[0]+2)
            if (isinstance(qc.gates[i], Gate) and
                    qc.gates[i].controls is not None):
                assert (qc2.gates[i].controls[0]
                        == qc.gates[i].controls[0]+2)

    def test_add_state(self):
        """
        Addition of input and output states to a circuit.
        """
        qc = QubitCircuit(3)

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        assert qc.input_states[0] == "0"
        assert qc.input_states[2] is None
        assert qc.output_states[1] == "+"

        qc1 = QubitCircuit(10)

        qc1.add_state("0", targets=[2, 3, 5, 6])
        qc1.add_state("+", targets=[1, 4, 9])
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("beta", targets=[0], state_type="output")
        assert qc1.input_states[0] is None

        assert qc1.input_states[2] == "0"
        assert qc1.input_states[3] == "0"
        assert qc1.input_states[6] == "0"
        assert qc1.input_states[1] == "+"
        assert qc1.input_states[4] == "+"

        assert qc1.output_states[2] is None
        assert qc1.output_states[1] == "A"
        assert qc1.output_states[4] == "A"
        assert qc1.output_states[9] == "A"

        assert qc1.output_states[0] == "beta"

    def test_add_measurement(self):
        """
        Addition of Measurement Object to a circuit.
        """

        qc = QubitCircuit(3, num_cbits=2)

        qc.add_measurement("M0", targets=[0], classical_store=1)
        qc.add_gate("CNOT", targets=[1], controls=[0])
        qc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
        qc.add_measurement("M1", targets=[2], classical_store=0)
        qc.add_gate("SNOT", targets=[1], classical_controls=[0, 1])
        qc.add_measurement("M2", targets=[1])

        # checking correct addition of measurements
        assert qc.gates[0].targets[0] == 0
        assert qc.gates[0].classical_store == 1
        assert qc.gates[3].name == "M1"
        assert qc.gates[5].classical_store is None

        # checking if gates are added correctly with measurements
        assert qc.gates[2].name == "TOFFOLI"
        assert qc.gates[4].classical_controls == [0, 1]

    @pytest.mark.parametrize('gate', ['X', 'Y', 'Z', 'S', 'T'])
    def test_exceptions(self, gate):
        """
        Text exceptions are thrown correctly for inadequate inputs
        """
        qc = QubitCircuit(2)
        pytest.raises(ValueError, qc.add_gate, gate, targets=[1], controls=[0])

    @pytest.mark.parametrize('gate', ['CY', 'CZ', 'CS', 'CT'])
    def test_exceptions_controlled(self, gate):
        """
        Text exceptions are thrown correctly for inadequate inputs
        """
        qc = QubitCircuit(2)
        '''
        pytest.raises(ValueError, qc.add_gate, gate,
                    targets=[1], controls=[0])
        '''

        pytest.raises(ValueError, qc.add_gate, gate,
                      targets=[1])
        pytest.raises(ValueError, qc.add_gate, gate)

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

        assert qc.gates[8].name == "T"
        assert qc.gates[7].name == "S"
        assert qc.gates[6].name == "CZ"
        assert qc.gates[5].name == "CT"
        assert qc.gates[4].name == "Z"
        assert qc.gates[3].name == "CS"
        assert qc.gates[2].name == "Y"
        assert qc.gates[1].name == "CY"
        assert qc.gates[0].name == "X"

        assert qc.gates[8].targets == [2]
        assert qc.gates[7].targets == [1]
        assert qc.gates[6].targets == [0]
        assert qc.gates[5].targets == [2]
        assert qc.gates[4].targets == [1]
        assert qc.gates[3].targets == [0]
        assert qc.gates[2].targets == [2]
        assert qc.gates[1].targets == [1]
        assert qc.gates[0].targets == [0]

        assert qc.gates[6].controls == [0]
        assert qc.gates[5].controls == [2]
        assert qc.gates[3].controls == [1]
        assert qc.gates[1].controls == [0]

    def test_reverse(self):
        """
        Reverse a quantum circuit
        """
        qc = QubitCircuit(3)

        qc.add_gate("RX", targets=[0], arg_value=3.141,
                    arg_label=r"\pi/2")
        qc.add_gate("CNOT", targets=[1], controls=[0])
        qc.add_measurement("M1", targets=[1])
        qc.add_gate("SNOT", targets=[2])
        # Keep input output same

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        qc_rev = qc.reverse_circuit()

        assert qc_rev.gates[0].name == "SNOT"
        assert qc_rev.gates[1].name == "M1"
        assert qc_rev.gates[2].name == "CNOT"
        assert qc_rev.gates[3].name == "RX"

        assert qc_rev.input_states[0] == "0"
        assert qc_rev.input_states[2] is None
        assert qc_rev.output_states[1] == "+"

    def test_user_gate(self):
        """
        User defined gate for QubitCircuit
        """
        def customer_gate1(arg_values):
            mat = np.zeros((4, 4), dtype=np.complex)
            mat[0, 0] = mat[1, 1] = 1.
            mat[2:4, 2:4] = gates.rx(arg_values)
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
        np.testing.assert_allclose(props[0], result1)
        result2 = tensor(identity(2), customer_gate2(), identity(2))
        np.testing.assert_allclose(props[1], result2)

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
        np.testing.assert_allclose(mat3, ptrace(props[0], 0) - 1)

    @pytest.mark.repeat(10)
    def test_run_teleportation(self):
        """
        Test circuit run and mid-circuit measurement functionality
        by repeating the teleportation circuit on multiple random kets
        """

        teleportation = _teleportation_circuit()

        state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))
        initial_measurement = Measurement("start", targets=[0])
        _, initial_probabilities = initial_measurement.measurement_comp_basis(state)

        state_final, probability = teleportation.run(state)

        final_measurement = Measurement("start", targets=[2])
        _, final_probabilities = final_measurement.measurement_comp_basis(state_final)

        np.testing.assert_allclose(initial_probabilities, final_probabilities)

    def test_runstatistics_teleportation(self):
        """
        Test circuit run_statistics on teleportation circuit
        """

        teleportation = _teleportation_circuit()
        final_measurement = Measurement("start", targets=[2])
        initial_measurement = Measurement("start", targets=[0])

        state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))
        _, initial_probabilities = initial_measurement.measurement_comp_basis(state)

        states, probabilites = teleportation.run_statistics(state)

        for i, state in enumerate(states):
            state_final = state
            prob = probabilites[i]
            _, final_probabilities = final_measurement.measurement_comp_basis(state_final)
            np.testing.assert_allclose(initial_probabilities,
                                        final_probabilities)
            assert prob == pytest.approx(0.25, abs=1e-7)


if __name__ == "__main__":
    run_module_suite()
