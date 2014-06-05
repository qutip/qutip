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
from numpy.testing import assert_, run_module_suite
from qutip.qip.qft import qft, qft_steps
from qutip.qip.gates import *
from qutip.qip.circuit import *


class TestQubitCircuit:
    """
    A test class for the QuTiP functions for Circuit resolution
    """

    def testSWAPtoCNOT(self):
        """
        SWAP to CNOT: compare unitary matrix for SWAP and product of 
        resolved matrices in terms of CNOT
        """
        qc = QubitCircuit(2)
        qc.add_gate("SWAP", targets=[0, 1])
        U0 = gate_sequence_product(qc.unitary_matrix())
        qc.resolved_gates(basis="CNOT")
        U1 = gate_sequence_product(qc.unitary_matrix(resolved=True))
        assert_((U0 - U1).norm() < 1e-12)
        
    def testCNOTtoCSIGN(self):
        """
        CNOT to CSIGN: compare unitary matrix for CNOT and product of 
        resolved matrices in terms of CSIGN
        """
        qc = QubitCircuit(2)
        qc.add_gate("CNOT", targets=[0], controls=[1])
        U0 = gate_sequence_product(qc.unitary_matrix())
        qc.resolved_gates(basis="CSIGN")
        U1 = gate_sequence_product(qc.unitary_matrix(resolved=True))
        assert_((U0 - U1).norm() < 1e-12)
    
    def testCNOTtoSQRTSWAP(self):
        """
        CNOT to SQRTSWAP: compare unitary matrix for CNOT and product of 
        resolved matrices in terms of SQRTSWAP
        """
        qc = QubitCircuit(2)
        qc.add_gate("CNOT", targets=[0], controls=[1])
        U0 = gate_sequence_product(qc.unitary_matrix())
        qc.resolved_gates(basis="SQRTSWAP")
        U1 = gate_sequence_product(qc.unitary_matrix(resolved=True))
        assert_((U0 - U1).norm() < 1e-12)
   
    def testCNOTtoISWAP(self):
        """
        CNOT to ISWAP: compare unitary matrix for CNOT and product of 
        resolved matrices in terms of ISWAP
        """
        qc = QubitCircuit(2)
        qc.add_gate("CNOT", targets=[0], controls=[1])
        U0 = gate_sequence_product(qc.unitary_matrix())
        qc.resolved_gates(basis="ISWAP")
        U1 = gate_sequence_product(qc.unitary_matrix(resolved=True))
        assert_((U0 - U1).norm() < 1e-12)
   
if __name__ == "__main__":
    run_module_suite()
