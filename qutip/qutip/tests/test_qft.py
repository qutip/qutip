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

from numpy.testing import assert_, assert_equal, assert_string_equal, run_module_suite
from qutip.qip.algorithms.qft import qft, qft_steps, qft_gate_sequence
from qutip.qip.operations.gates import gate_sequence_product


class TestQFT:
    """
    A test class for the QuTiP functions for QFT
    """

    def testQFTComparison(self):
        """
        qft: compare qft and product of qft steps
        """
        for N in range(1, 5):
            U1 = qft(N)
            U2 = gate_sequence_product(qft_steps(N))
            assert_((U1 - U2).norm() < 1e-12)

    def testQFTGateSequenceNoSwapping(self):
        """
        qft: Inspect key properties of gate sequences of length N,
        with swapping disabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=False)
            assert_equal(circuit.N, N)

            totsize = N * (N + 1) / 2
            assert_equal(len(circuit.gates), totsize)

            snots = sum(g.name == "SNOT" for g in circuit.gates)
            assert_equal(snots, N)

            phases = sum(g.name == "CPHASE" for g in circuit.gates)
            assert_equal(phases, N * (N - 1) / 2)

    def testQFTGateSequenceWithSwapping(self):
        """
        qft: Inspect swap gates added to gate sequences if
        swapping is enabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=True)

            phases = int(N * (N + 1) / 2)
            swaps = int(N // 2)
            assert_equal(len(circuit.gates), phases + swaps)

            for i in range(phases, phases + swaps):
                assert_string_equal(circuit.gates[i].name, "SWAP")

if __name__ == "__main__":
    run_module_suite()
