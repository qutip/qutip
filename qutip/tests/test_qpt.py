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
import scipy.linalg as la

from qutip import (spre, spost, qeye, sigmax, sigmay, sigmaz, qpt)
from qutip.qip.gates import snot, cnot


def test_qpt_snot():
    "quantum process tomography for snot gate"

    U_psi = snot()
    U_rho = spre(U_psi) * spost(U_psi.dag())
    N = 1
    op_basis = [[qeye(2), sigmax(), 1j * sigmay(), sigmaz()] for i in range(N)]
    # op_label = [["i", "x", "y", "z"] for i in range(N)]
    chi1 = qpt(U_rho, op_basis)

    chi2 = np.zeros((2 ** (2 * N), 2 ** (2 * N)), dtype=complex)
    chi2[1, 1] = chi2[1, 3] = chi2[3, 1] = chi2[3, 3] = 0.5

    assert_(la.norm(chi2 - chi1) < 1e-8)


def test_qpt_cnot():
    "quantum process tomography for cnot gate"

    U_psi = cnot()
    U_rho = spre(U_psi) * spost(U_psi.dag())
    N = 2
    op_basis = [[qeye(2), sigmax(), 1j * sigmay(), sigmaz()] for i in range(N)]
    # op_label = [["i", "x", "y", "z"] for i in range(N)]
    chi1 = qpt(U_rho, op_basis)

    chi2 = np.zeros((2 ** (2 * N), 2 ** (2 * N)), dtype=complex)
    chi2[0, 0] = chi2[0, 1] = chi2[1, 0] = chi2[1, 1] = 0.25

    chi2[12, 0] = chi2[12, 1] = 0.25
    chi2[13, 0] = chi2[13, 1] = -0.25

    chi2[0, 12] = chi2[1, 12] = 0.25
    chi2[0, 13] = chi2[1, 13] = -0.25

    chi2[12, 12] = chi2[13, 13] = 0.25
    chi2[13, 12] = chi2[12, 13] = -0.25

    assert_(la.norm(chi2 - chi1) < 1e-8)

if __name__ == "__main__":
    run_module_suite()
