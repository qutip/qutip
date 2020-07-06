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
from qutip.qip.qasm import read_qasm
import numpy as np
from pathlib import Path


@pytest.mark.parametrize(["filename", "error", "error_regex"], [
    pytest.param("command_error.qasm", SyntaxError,
                 "QASM: post is not a valid QASM command."),
    pytest.param("bracket_error.qasm", SyntaxError,
                 "QASM: incorrect bracket formatting"),
    pytest.param("qasm_error.qasm", SyntaxError,
                 "QASM: File does not contain QASM 2.0 header")])
def test_qasm_errors(filename, error, error_regex):
    filepath = Path(__file__).parent
    np.testing.assert_raises_regex(error, error_regex, read_qasm,
                                   "{}/qasm_files/{}".format(filepath, filename))


def check_gate_defn(gate, gate_name, targets, controls=None, classical_controls=None, control_value=None):
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
    filepath = Path(__file__).parent
    file = "{}/qasm_files/test_add.qasm".format(filepath)
    qc = read_qasm(file)
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
