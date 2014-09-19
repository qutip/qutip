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

from numpy.testing import assert_, run_module_suite
from qutip.qip.gates import gate_sequence_product
from qutip.qip.circuit import QubitCircuit


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


if __name__ == "__main__":
    run_module_suite()
