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

from qutip.operators import (num, destroy,
                             sigmax, sigmay, sigmaz, sigmam, sigmap)
from qutip.states import fock, fock_dm
from qutip.expect import expect
from qutip.mesolve import mesolve
from qutip.random_objects import rand_herm, rand_ket


class TestExpect:
    """
    A test class for the QuTiP function for calculating expectation values.
    """

    def testOperatorKet(self):
        """
        expect: operator and ket
        """
        N = 10
        op_N = num(N)
        op_a = destroy(N)
        for n in range(N):
            e = expect(op_N, fock(N, n))
            assert_(e == n)
            assert_(type(e) == float)
            e = expect(op_a, fock(N, n))
            assert_(e == 0)
            assert_(type(e) == complex)

    def testOperatorKetRand(self):
        """
        expect: rand operator & rand ket
        """
        for kk in range(20):
            N = 20
            H = rand_herm(N, 0.2)
            psi = rand_ket(N,0.3)
            out = expect(H,psi)
            ans = (psi.dag()*H*psi).tr()
            assert_(np.abs(out-ans) < 1e-14)
    
            G = rand_herm(N, 0.1)
            out = expect(H+1j*G,psi)
            ans = (psi.dag()*(H+1j*G)*psi).tr()
            assert_(np.abs(out-ans) < 1e-14)
    
    def testOperatorDensityMatrix(self):
        """
        expect: operator and density matrix
        """
        N = 10
        op_N = num(N)
        op_a = destroy(N)
        for n in range(N):
            e = expect(op_N, fock_dm(N, n))
            assert_(e == n)
            assert_(type(e) == float)
            e = expect(op_a, fock_dm(N, n))
            assert_(e == 0)
            assert_(type(e) == complex)

    def testOperatorStateList(self):
        """
        expect: operator and state list
        """
        N = 10
        op = num(N)

        res = expect(op, [fock(N, n) for n in range(N)])
        assert_(all(res == range(N)))
        assert_(isinstance(res, np.ndarray) and res.dtype == np.float64)

        res = expect(op, [fock_dm(N, n) for n in range(N)])
        assert_(all(res == range(N)))
        assert_(isinstance(res, np.ndarray) and res.dtype == np.float64)

        op = destroy(N)

        res = expect(op, [fock(N, n) for n in range(N)])
        assert_(all(res == np.zeros(N)))
        assert_(isinstance(res, np.ndarray) and res.dtype == np.complex128)

        res = expect(op, [fock_dm(N, n) for n in range(N)])
        assert_(all(res == np.zeros(N)))
        assert_(isinstance(res, np.ndarray) and res.dtype == np.complex128)

    def testOperatorListState(self):
        """
        expect: operator list and state
        """
        res = expect([sigmax(), sigmay(), sigmaz()], fock(2, 0))
        assert_(len(res) == 3)
        assert_(all(abs(res - [0, 0, 1]) < 1e-12))

        res = expect([sigmax(), sigmay(), sigmaz()], fock_dm(2, 1))
        assert_(len(res) == 3)
        assert_(all(abs(res - [0, 0, -1]) < 1e-12))

    def testOperatorListStateList(self):
        """
        expect: operator list and state list
        """
        operators = [sigmax(), sigmay(), sigmaz(), sigmam(), sigmap()]
        states = [fock(2, 0), fock(2, 1), fock_dm(2, 0), fock_dm(2, 1)]
        res = expect(operators, states)

        assert_(len(res) == len(operators))

        for r_idx, r in enumerate(res):

            assert_(isinstance(r, np.ndarray))

            if operators[r_idx].isherm:
                assert_(r.dtype == np.float64)
            else:
                assert_(r.dtype == np.complex128)

            for s_idx, s in enumerate(states):
                assert_(r[s_idx] == expect(operators[r_idx], states[s_idx]))

    def testExpectSolverCompatibility(self):
        """
        expect: operator list and state list
        """
        c_ops = [0.0001 * sigmaz()]
        e_ops = [sigmax(), sigmay(), sigmaz(), sigmam(), sigmap()]
        times = np.linspace(0, 10, 100)

        res1 = mesolve(sigmax(), fock(2, 0), times, c_ops, e_ops)
        res2 = mesolve(sigmax(), fock(2, 0), times, c_ops, [])

        e1 = res1.expect
        e2 = expect(e_ops, res2.states)

        assert_(len(e1) == len(e2))

        for n in range(len(e1)):
            assert_(len(e1[n]) == len(e2[n]))
            assert_(isinstance(e1[n], np.ndarray))
            assert_(isinstance(e2[n], np.ndarray))
            assert_(e1[n].dtype == e2[n].dtype)
            assert_(all(abs(e1[n] - e2[n]) < 1e-12))


if __name__ == "__main__":
    run_module_suite()
