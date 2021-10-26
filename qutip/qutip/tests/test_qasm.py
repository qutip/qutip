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
from pathlib import Path

from qutip.qip.qasm import read_qasm, circuit_to_qasm_str
from qutip.qip.circuit import Measurement, QubitCircuit
from qutip import tensor, rand_ket, basis, rand_dm, identity
from qutip.qip.operations.gates import cnot, ry


@pytest.mark.parametrize(["filename", "error", "error_message"], [
    pytest.param("command_error.qasm", SyntaxError,
                 "QASM: post is not a valid QASM command."),
    pytest.param("bracket_error.qasm", SyntaxError,
                 "QASM: incorrect bracket formatting"),
    pytest.param("qasm_error.qasm", SyntaxError,
                 "QASM: File does not contain QASM 2.0 header")])
def test_qasm_errors(filename, error, error_message):
    filepath = Path(__file__).parent / 'qasm_files' / filename
    with pytest.raises(error) as exc_info:
        read_qasm(filepath)
    assert error_message in str(exc_info.value)


def check_gate_defn(gate, gate_name, targets, controls=None,
                    classical_controls=None, control_value=None):
    assert gate.name == gate_name
    assert gate.targets == targets
    assert gate.controls == controls
    assert gate.classical_controls == classical_controls
    assert gate.control_value == control_value


def check_measurement_defn(gate, gate_name, targets, classical_store):
    assert gate.name == gate_name
    assert gate.targets == targets
    assert gate.classical_store == classical_store


def test_qasm_addcircuit():
    filename = "test_add.qasm"
    filepath = Path(__file__).parent / 'qasm_files' / filename
    qc = read_qasm(filepath)
    assert qc.N == 2
    assert qc.num_cbits == 2
    check_gate_defn(qc.gates[0], "X", [1])
    check_gate_defn(qc.gates[1], "SNOT", [0])
    check_gate_defn(qc.gates[2], "SNOT", [1])
    check_gate_defn(qc.gates[3], "CNOT", [1], [0])
    check_gate_defn(qc.gates[4], "SNOT", [0])
    check_gate_defn(qc.gates[5], "SNOT", [1])
    check_gate_defn(qc.gates[6], "SNOT", [0], None, [0, 1], 0)
    check_measurement_defn(qc.gates[7], "M", [0], 0)
    check_measurement_defn(qc.gates[8], "M", [1], 1)


def test_custom_gates():
    filename = "test_custom_gates.qasm"
    filepath = Path(__file__).parent / 'qasm_files' / filename
    qc = read_qasm(filepath)
    unitaries = qc.propagators()
    assert (unitaries[0] - unitaries[1]).norm() < 1e-12
    ry_cx = cnot() * tensor(identity(2), ry(np.pi/2))
    assert (unitaries[2] - ry_cx).norm() < 1e-12


def test_qasm_teleportation():
    filename = "teleportation.qasm"
    filepath = Path(__file__).parent / 'qasm_files' / filename
    teleportation = read_qasm(filepath)
    final_measurement = Measurement("start", targets=[2])
    initial_measurement = Measurement("start", targets=[0])

    state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))
    _, initial_probabilities = initial_measurement.measurement_comp_basis(state)

    states, probabilites = teleportation.run_statistics(state)

    for i, state in enumerate(states):
        final = state
        prob = probabilites[i]
        _, final_probabilities = final_measurement.measurement_comp_basis(final)
        np.testing.assert_allclose(initial_probabilities,
                                   final_probabilities)
        assert prob == pytest.approx(0.25, abs=1e-7)


def test_qasm_str():
    expected_qasm_str = ('// QASM 2.0 file generated by QuTiP\n\nOPENQASM 2.0;'
                         '\ninclude "qelib1.inc";\n\nqreg q[2];\ncreg c[1];\n\n'
                         'x q[0];\nmeasure q[1] -> c[0]\n')
    simple_qc = QubitCircuit(2, num_cbits=1)
    simple_qc.add_gate("X", targets=[0])
    simple_qc.add_measurement("M", targets=[1], classical_store=0)
    assert circuit_to_qasm_str(simple_qc) == expected_qasm_str


def test_export_import():
    qc = QubitCircuit(3)
    qc.add_gate("CRY", targets=1, controls=0, arg_value=np.pi)
    qc.add_gate("CRX", targets=1, controls=0, arg_value=np.pi)
    qc.add_gate("CRZ", targets=1, controls=0, arg_value=np.pi)
    qc.add_gate("CNOT", targets=1, controls=0)
    qc.add_gate("TOFFOLI", targets=2, controls=[0, 1])
    # qc.add_gate("SQRTNOT", targets=0)
    qc.add_gate("CS", targets=1, controls=0)
    qc.add_gate("CT", targets=1, controls=0)
    qc.add_gate("SWAP", targets=[0, 1])
    qc.add_gate("QASMU", targets=[0], arg_value=[np.pi, np.pi, np.pi])
    qc.add_gate("RX", targets=[0], arg_value=np.pi)
    qc.add_gate("RY", targets=[0], arg_value=np.pi)
    qc.add_gate("RZ", targets=[0], arg_value=np.pi)
    qc.add_gate("SNOT", targets=[0])
    qc.add_gate("X", targets=[0])
    qc.add_gate("Y", targets=[0])
    qc.add_gate("Z", targets=[0])
    qc.add_gate("S", targets=[0])
    qc.add_gate("T", targets=[0])
    # qc.add_gate("CSIGN", targets=[0], controls=[1])

    read_qc = read_qasm(circuit_to_qasm_str(qc), strmode=True)

    props = qc.propagators()
    read_props = read_qc.propagators()

    for u0, u1 in zip(props, read_props):
        assert (u0 - u1).norm() < 1e-12
