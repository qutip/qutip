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
from qutip.operators import identity, tensor
from qutip.qip.circuit import QubitCircuit
from qutip import rand_ket, basis


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

def test_runstatistics_teleportation():
    """
    Test circuit run_statistics on teleportation circuit
    """

    teleportation = _teleportation_circuit()
    state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))

    states, probabilites = teleportation.run_statistics(state)
    print(states)

if __name__ == "__main__":
    test_runstatistics_teleportation()
