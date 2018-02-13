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

# disable the progress bar
import os

from qutip import sigmax, sigmay, sigmaz, sigmam, qeye
from qutip import qobj, basis, expect
from qutip import sesolve
from qutip.solver import Options

os.environ['QUTIP_GRAPHICS'] = "NO"


class TestSchrodingerEqSolve:
    """
    A test class for the QuTiP Schrodinger Eq. solver
    """

    def qubit_integrate(self, tlist, psi0, epsilon, delta, e_ops=[]):

        H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()

        return sesolve(H, psi0, tlist, e_ops)

    def test_01_1_state_with_const_H(self):
        "sesolve: state with const H"
        tol = 5e-3
        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 5, 200)
        e_ops = [sigmax(), sigmay(), sigmaz()]

        output = self.qubit_integrate(tlist, psi0, epsilon, delta, e_ops)
        sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist)
        sz_analytic = np.cos(2 * np.pi * tlist)

        assert_(max(abs(sx - sx_analytic)) < tol,
                msg="expect X not matching analytic")
        assert_(max(abs(sy - sy_analytic)) < tol,
                msg="expect Y not matching analytic")
        assert_(max(abs(sz - sz_analytic)) < tol,
                msg="expect Z not matching analytic")

    def test_01_1_unitary_with_const_H(self):
        "sesolve: unitary operator with const H"
        tol = 5e-3
        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        psi0 = basis(2, 0)        # initial state
        U0 = qeye(2)                # initital operator
        tlist = np.linspace(0, 5, 200)

        output = self.qubit_integrate(tlist, U0, epsilon, delta)
        sx = [expect(sigmax(), U*psi0) for U in output.states]
        sy = [expect(sigmay(), U*psi0) for U in output.states]
        sz = [expect(sigmaz(), U*psi0) for U in output.states]

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist)
        sz_analytic = np.cos(2 * np.pi * tlist)

        assert_(max(abs(sx - sx_analytic)) < tol,
                msg="expect X not matching analytic")
        assert_(max(abs(sy - sy_analytic)) < tol,
                msg="expect Y not matching analytic")
        assert_(max(abs(sz - sz_analytic)) < tol,
                msg="expect Z not matching analytic")

if __name__ == "__main__":
    run_module_suite()
