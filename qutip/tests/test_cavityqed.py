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

import warnings
import numpy as np
import pytest
import qutip
from qutip.qip.circuit import Gate
from qutip.qip.operations.gates import gate_sequence_product
from qutip.qip.device.cavityqed import DispersiveCavityQED

_tol = 1e-2

_iswap = Gate("ISWAP", targets=[0, 1])
_sqrt_iswap = Gate("SQRTISWAP", targets=[0, 1])
_rz = Gate("RZ", targets=[1], arg_value=np.pi/2, arg_label=r"\pi/2")
_rx = Gate("RX", targets=[0], arg_value=np.pi/2, arg_label=r"\pi/2")


@pytest.mark.parametrize("gates", [
    pytest.param([_iswap], id="ISWAP"),
    pytest.param([_sqrt_iswap], id="SQRTISWAP", marks=pytest.mark.skip),
    pytest.param([_iswap, _rz, _rx], id="ISWAP RZ RX"),
])
def test_device_against_gate_sequence(gates):
    n_qubits = 3
    circuit = qutip.qip.circuit.QubitCircuit(n_qubits)
    for gate in gates:
        circuit.add_gate(gate)
    U_ideal = gate_sequence_product(circuit.propagators())

    device = DispersiveCavityQED(n_qubits, correct_global_phase=True)
    U_physical = gate_sequence_product(device.run(circuit))
    assert (U_ideal - U_physical).norm() < _tol


def test_analytical_evolution():
    n_qubits = 3
    circuit = qutip.qip.circuit.QubitCircuit(n_qubits)
    for gate in [_iswap, _rz, _rx]:
        circuit.add_gate(gate)
    state = qutip.rand_ket(2**n_qubits)
    state.dims = [[2]*n_qubits, [1]*n_qubits]
    ideal = gate_sequence_product([state] + circuit.propagators())
    device = DispersiveCavityQED(n_qubits, correct_global_phase=True)
    operators = device.run_state(init_state=state, qc=circuit, analytical=True)
    result = gate_sequence_product(operators)
    assert abs(qutip.metrics.fidelity(result, ideal) - 1) < _tol


def test_numerical_evolution():
    n_qubits = 3
    circuit = qutip.qip.circuit.QubitCircuit(n_qubits)
    circuit.add_gate("RX", targets=[0], arg_value=np.pi/2)
    circuit.add_gate("CNOT", targets=[0], controls=[1])
    circuit.add_gate("ISWAP", targets=[2, 1])
    circuit.add_gate("CNOT", targets=[0], controls=[2])
    with warnings.catch_warnings(record=True):
        device = DispersiveCavityQED(n_qubits, g=0.1)
    device.load_circuit(circuit)

    state = qutip.rand_ket(2**n_qubits)
    state.dims = [[2]*n_qubits, [1]*n_qubits]
    target = gate_sequence_product([state] + circuit.propagators())
    extra = qutip.basis(10, 0)
    options = qutip.Options(store_final_state=True, nsteps=50_000)
    result = device.run_state(init_state=qutip.tensor(extra, state),
                              analytical=False,
                              options=options)
    assert _tol > abs(1 - qutip.metrics.fidelity(result.final_state,
                                                 qutip.tensor(extra, target)))
