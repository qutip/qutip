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
import numpy as np
from numpy.testing import assert_, run_module_suite
import qutip
from qutip import (expect, destroy, coherent, coherent_dm, thermal_dm,
                   fock_dm, triplet_states)


@pytest.mark.parametrize("size, n", [(2, 0), (2, 1), (100, 99)])
def test_basis_simple(size, n):
    qobj = qutip.basis(size, n)
    numpy = np.zeros((size, 1), dtype=complex)
    numpy[n, 0] = 1
    assert np.array_equal(qobj.full(), numpy)


@pytest.mark.parametrize("to_test", [qutip.basis, qutip.fock, qutip.fock_dm])
@pytest.mark.parametrize("size, n", [([2, 2], [0, 1]), ([2, 3, 4], [1, 2, 0])])
def test_implicit_tensor_basis_like(to_test, size, n):
    implicit = to_test(size, n)
    explicit = qutip.tensor(*[to_test(ss, nn) for ss, nn in zip(size, n)])
    assert implicit == explicit


@pytest.mark.parametrize("size, n, m", [
        ([2, 2], [0, 0], [1, 1]),
        ([2, 3, 4], [1, 2, 0], [0, 1, 3]),
    ])
def test_implicit_tensor_projection(size, n, m):
    implicit = qutip.projection(size, n, m)
    explicit = qutip.tensor(*[qutip.projection(ss, nn, mm)
                              for ss, nn, mm in zip(size, n, m)])
    assert implicit == explicit


class TestStates:
    """
    A test class for the QuTiP functions for generating quantum states
    """

    def testCoherentState(self):
        """
        states: coherent state
        """
        N = 10
        alpha = 0.5
        c1 = coherent(N, alpha) # displacement method
        c2 = coherent(7, alpha, offset=3) # analytic method
        assert_(abs(expect(destroy(N), c1) - alpha) < 1e-10)
        assert_((c1[3:]-c2).norm() < 1e-7)


    def testCoherentDensityMatrix(self):
        """
        states: coherent density matrix
        """
        N = 10

        rho = coherent_dm(N, 1)

        # make sure rho has trace close to 1.0
        assert_(abs(rho.tr() - 1.0) < 1e-12)

    def testThermalDensityMatrix(self):
        """
        states: thermal density matrix
        """
        N = 40

        rho = thermal_dm(N, 1)

        # make sure rho has trace close to 1.0
        assert_(abs(rho.tr() - 1.0) < 1e-12)

    def testFockDensityMatrix(self):
        """
        states: Fock density matrix
        """
        N = 10
        for i in range(N):
            rho = fock_dm(N, i)
            # make sure rho has trace close to 1.0
            assert_(abs(rho.tr() - 1.0) < 1e-12)
            assert_(rho.data[i, i] == 1.0)

    def testTripletStateNorm(self):
        """
        Test the states returned by function triplet_states are normalized.
        """
        for triplet in triplet_states():
            assert_(abs(triplet.norm() - 1.) < 1e-12)


if __name__ == "__main__":
    run_module_suite()
