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

from qutip import sigmax, sigmay, sigmaz, tensor, destroy, qeye
from numpy import amax
from numpy.testing import assert_equal, run_module_suite
import scipy


def test_diagHamiltonian1():
    """
    Diagonalization of random two-level system
    """

    H = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()

    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)


def test_diagHamiltonian2():
    """
    Diagonalization of composite systems
    """

    H1 = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()
    H2 = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()

    H = tensor(H1, H2)

    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)

    N1 = 10
    N2 = 2

    a1 = tensor(destroy(N1), qeye(N2))
    a2 = tensor(qeye(N1), destroy(N2))
    H = scipy.rand() * a1.dag() * a1 + scipy.rand() * a2.dag() * a2 + \
        scipy.rand() * (a1 + a1.dag()) * (a2 + a2.dag())
    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)


if __name__ == "__main__":
    run_module_suite()
