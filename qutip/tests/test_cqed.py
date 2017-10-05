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
from qutip.qip.gates import gate_sequence_product
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.cqed import DispersivecQED


class Testcqed:
    """
    A test class for the QuTiP functions for physical implementation of
    resonator-qubit models.
    """

    def test_dispersivecqed_ISWAP(self):
        """
        Dispersive cQED Setup: compare unitary matrix for ISWAP and propogator
        matrix of the implemented physical model.
        """
        N = 3

        qc1 = QubitCircuit(N)
        qc1.add_gate("ISWAP", targets=[0, 1])
        U_ideal = gate_sequence_product(qc1.propagators())

        p = DispersivecQED(N, correct_global_phase=True)
        U_list = p.run(qc1)
        U_physical = gate_sequence_product(U_list)

        print((U_ideal - U_physical).norm())
        assert_((U_ideal - U_physical).norm() < 1e-2)

    def skip_dispersivecqed_SQRTISWAP(self):
        """
        Dispersive cQED Setup: compare unitary matrix for SQRTISWAP and
        propogator matrix of the implemented physical model.
        """
        N = 3

        qc1 = QubitCircuit(N)
        qc1.add_gate("SQRTISWAP", targets=[0, 1])
        U_ideal = gate_sequence_product(qc1.propagators())

        p = DispersivecQED(N, correct_global_phase=True)
        U_list = p.run(qc1)
        U_physical = gate_sequence_product(U_list)

        print((U_ideal - U_physical).norm())
        assert_((U_ideal - U_physical).norm() < 1e-4)

    def test_dispersivecqed_combination(self):
        """
        Dispersive cQED Setup: compare unitary matrix for ISWAP, SQRTISWAP,
        RX and RY gates and the propogator matrix of the implemented physical
        model.
        """
        N = 3

        qc1 = QubitCircuit(N)
        qc1.add_gate("ISWAP", targets=[0, 1])
        qc1.add_gate("RZ", arg_value=np.pi/2, arg_label=r"\pi/2", targets=[1])
        qc1.add_gate("RX", arg_value=np.pi/2, arg_label=r"\pi/2", targets=[0])
        U_ideal = gate_sequence_product(qc1.propagators())

        p = DispersivecQED(N, correct_global_phase=True)
        U_list = p.run(qc1)
        U_physical = gate_sequence_product(U_list)

        print((U_ideal - U_physical).norm())
        assert_((U_ideal - U_physical).norm() < 1e-2)


if __name__ == "__main__":
    run_module_suite()
