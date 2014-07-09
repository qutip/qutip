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

from numpy import linspace, sum, abs
from numpy.testing import assert_, run_module_suite, assert_equal
from qutip import *
from scipy.special import laguerre
from numpy.random import rand


def test_wigner_coherent():
    "wigner: test wigner function calculation for coherent states"
    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 20
    beta = rand() + rand() * 1.0j
    psi = coherent(N, beta)

    # calculate the wigner function using qutip and analytic formula
    W_qutip = wigner(psi, xvec, yvec, g=2)
    W_analytic = 2 / pi * exp(-2 * abs(a - beta) ** 2)

    # check difference
    assert_(sum(abs(W_qutip - W_analytic) ** 2) < 1e-4)

    # check normalization
    assert_(sum(W_qutip) * dx * dy - 1.0 < 1e-8)
    assert_(sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_fock():
    "wigner: test wigner function calculation for Fock states"

    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in [2, 3, 4, 5, 6]:

        psi = fock(N, n)

        # calculate the wigner function using qutip and analytic formula
        W_qutip = wigner(psi, xvec, yvec, g=2)
        W_analytic = 2 / pi * (-1) ** n * exp(-2 * abs(a) ** 2) * polyval(
            laguerre(n), 4 * abs(a) ** 2)

        # check difference
        assert_(sum(abs(W_qutip - W_analytic)) < 1e-4)

        # check normalization
        assert_(sum(W_qutip) * dx * dy - 1.0 < 1e-8)
        assert_(sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_dm():
    "wigner: compare wigner methods for random density matrices"

    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        rho = rand_dm(N, 0.5 + rand() / 2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(rho, xvec, yvec, g=2)
        W_qutip2 = wigner(rho, xvec, yvec, g=2, method='laguerre')

        # check difference
        assert_(sum(abs(W_qutip1 - W_qutip1)) < 1e-4)

        # check normalization
        assert_(sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert_(sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_ket():
    "wigner: compare wigner methods for random state vectors"

    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        psi = rand_ket(N, 0.5 + rand() / 2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(psi, xvec, yvec, g=2)
        W_qutip2 = wigner(psi, xvec, yvec, g=2, method='laguerre')

        # check difference
        assert_(sum(abs(W_qutip1 - W_qutip1)) < 1e-4)

        # check normalization
        assert_(sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert_(sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_fft_comparse_ket():
    "Wigner: Compare Wigner fft and iterative for rand. ket"
    N = 20
    xvec = linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_ket(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_fft_comparse_dm():
    "Wigner: Compare Wigner fft and iterative for rand. dm"
    N = 20
    xvec = linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(sum(abs(Wdiff)) < 1e-7, True)

if __name__ == "__main__":
    run_module_suite()
