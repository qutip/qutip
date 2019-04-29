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
from scipy.special import laguerre
from numpy.random import rand
from numpy.testing import assert_, run_module_suite, assert_equal

from qutip.states import coherent, fock, ket, bell_state
from qutip.wigner import wigner, wigner_transform, _parity
from qutip.random_objects import rand_dm, rand_ket


def test_wigner_bell1_su2parity():
    """wigner: testing the SU2 parity of the first Bell state.
    """
    psi = bell_state('00')

    steps = 25
    theta = np.tile(np.linspace(0, np.pi, steps), 2).reshape(2, steps)
    phi = np.tile(np.linspace(0, 2 * np.pi, steps), 2).reshape(2, steps)
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = np.real(((1 + np.sqrt(3)
                                            * np.cos(theta[0, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           + 3 * (np.sin(theta[0, t])
                                           * np.exp(-1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(-1j * phi[1, p])
                                           + np.sin(theta[0, t])
                                           * np.exp(1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(1j * phi[1, p]))
                                           + (1 - np.sqrt(3)
                                           * np.cos(theta[0, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[1, t]))) / 8.)

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert_(np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


def test_wigner_bell4_su2parity():
    """wigner: testing the SU2 parity of the fourth Bell state.
    """
    psi = bell_state('11')

    steps = 100
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = -0.5

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert_(np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


def test_wigner_bell4_fullparity():
    """wigner: testing the parity of the fourth Bell state using the parity of
    the full space.
    """
    psi = bell_state('11')

    steps = 100
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = -0.30901699

    print("wigner anal: ", wigner_analyt)
    wigner_theo = wigner_transform(psi, 0.5, True, steps, slicearray)

    print("wigner theo: ", wigner_theo)
    assert_(np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-4)


def test_parity():
    """wigner: testing the parity function.
    """
    j = 0.5
    assert_(_parity(2, j)[0, 0] - (1 - np.sqrt(3)) / 2. < 1e-11)
    assert_(_parity(2, j)[0, 1] < 1e-11)
    assert_(_parity(2, j)[1, 1] - (1 + np.sqrt(3)) / 2. < 1e-11)
    assert_(_parity(2, j)[1, 0] < 1e-11)


def test_wigner_pure_su2():
    """wigner: testing the SU2 wigner transformation of a pure state.
    """
    psi = (ket([1]))
    steps = 100
    theta = np.linspace(0, np.pi, steps)
    phi = np.linspace(0, 2 * np.pi, steps)
    theta = theta[None, :]
    phi = phi[None, :]
    slicearray = ['l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = (1 + np.sqrt(3) * np.cos(theta[0, t])) / 2.

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert_(np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


def test_wigner_ghz_su2parity():
    """wigner: testing the SU2 wigner transformation of the GHZ state.
    """
    psi = (ket([0, 0, 0]) + ket([1, 1, 1])) / np.sqrt(2)

    steps = 100
    N = 3
    theta = np.tile(np.linspace(0, np.pi, steps), N).reshape(N, steps)
    phi = np.tile(np.linspace(0, 2 * np.pi, steps), N).reshape(N, steps)
    slicearray = ['l', 'l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = np.real(((1 + np.sqrt(3)*np.cos(theta[0, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[2, t]))
                                           + 3**(3 / 2) * (np.sin(theta[0, t])
                                           * np.exp(-1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(-1j * phi[1, p])
                                           * np.sin(theta[2, t])
                                           * np.exp(-1j * phi[2, p])
                                           + np.sin(theta[0, t])
                                           * np.exp(1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(1j * phi[1, p])
                                           * np.sin(theta[2, t])
                                           * np.exp(1j * phi[2, p]))
                                           + (1 - np.sqrt(3)
                                           * np.cos(theta[0, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[2, t]))) / 16.)

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert_(np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


def test_angle_slicing():
    """wigner: tests angle slicing.
    """
    psi1 = bell_state('00')
    psi2 = bell_state('01')
    psi3 = bell_state('10')
    psi4 = bell_state('11')

    steps = 100
    j = 0.5

    wigner1 = wigner_transform(psi1, j, False, steps, ['l', 'l'])
    wigner2 = wigner_transform(psi2, j, False, steps, ['l', 'z'])
    wigner3 = wigner_transform(psi3, j, False, steps, ['l', 'x'])
    wigner4 = wigner_transform(psi4, j, False, steps, ['l', 'y'])

    assert_(np.sum(np.abs(wigner2 - wigner1)) < 1e-11)
    assert_(np.sum(np.abs(wigner3 - wigner2)) < 1e-11)
    assert_(np.sum(np.abs(wigner4 - wigner3)) < 1e-11)
    assert_(np.sum(np.abs(wigner4 - wigner1)) < 1e-11)


def test_wigner_coherent():
    "wigner: test wigner function calculation for coherent states"
    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 20
    beta = rand() + rand() * 1.0j
    psi = coherent(N, beta)

    # calculate the wigner function using qutip and analytic formula
    W_qutip = wigner(psi, xvec, yvec, g=2)
    W_qutip_cl = wigner(psi, xvec, yvec, g=2, method='clenshaw')
    W_analytic = 2 / np.pi * np.exp(-2 * abs(a - beta) ** 2)

    # check difference
    assert_(np.sum(abs(W_qutip - W_analytic) ** 2) < 1e-4)
    assert_(np.sum(abs(W_qutip_cl - W_analytic) ** 2) < 1e-4)

    # check normalization
    assert_(np.sum(W_qutip) * dx * dy - 1.0 < 1e-8)
    assert_(np.sum(W_qutip_cl) * dx * dy - 1.0 < 1e-8)
    assert_(np.sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_fock():
    "wigner: test wigner function calculation for Fock states"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in [2, 3, 4, 5, 6]:

        psi = fock(N, n)

        # calculate the wigner function using qutip and analytic formula
        W_qutip = wigner(psi, xvec, yvec, g=2)
        W_qutip_cl = wigner(psi, xvec, yvec, g=2, method='clenshaw')
        W_qutip_sparse = wigner(psi, xvec, yvec, g=2, sparse=True, method='clenshaw')
        W_analytic = 2 / np.pi * (-1) ** n * \
            np.exp(-2 * abs(a) ** 2) * np.polyval(laguerre(n), 4 * abs(a) ** 2)

        # check difference
        assert_(np.sum(abs(W_qutip - W_analytic)) < 1e-4)
        assert_(np.sum(abs(W_qutip_cl - W_analytic)) < 1e-4)
        assert_(np.sum(abs(W_qutip_sparse - W_analytic)) < 1e-4)

        # check normalization
        assert_(np.sum(W_qutip) * dx * dy - 1.0 < 1e-8)
        assert_(np.sum(W_qutip_cl) * dx * dy - 1.0 < 1e-8)
        assert_(np.sum(W_qutip_sparse) * dx * dy - 1.0 < 1e-8)
        assert_(np.sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_dm():
    "wigner: compare wigner methods for random density matrices"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    # a = X + 1j * Y  # consistent with g=2 option to wigner function

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
        assert_(np.sum(abs(W_qutip1 - W_qutip1)) < 1e-4)

        # check normalization
        assert_(np.sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert_(np.sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_ket():
    "wigner: compare wigner methods for random state vectors"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    # a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        psi = rand_ket(N, 0.5 + rand() / 2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(psi, xvec, yvec, g=2)
        W_qutip2 = wigner(psi, xvec, yvec, g=2, sparse=True)

        # check difference
        assert_(np.sum(abs(W_qutip1 - W_qutip2)) < 1e-4)

        # check normalization
        assert_(np.sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert_(np.sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_fft_comparse_ket():
    "Wigner: Compare Wigner fft and iterative for rand. ket"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_ket(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_fft_comparse_dm():
    "Wigner: Compare Wigner fft and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_clenshaw_iter_dm():
    "Wigner: Compare Wigner clenshaw and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wclen = wigner(rho, xvec, xvec, method='clenshaw')
        W = wigner(rho, xvec, xvec, method='iterative')

        Wdiff = abs(W - Wclen)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_clenshaw_sp_iter_dm():
    "Wigner: Compare Wigner sparse clenshaw and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wclen = wigner(rho, xvec, xvec, method='clenshaw', sparse=True)
        W = wigner(rho, xvec, xvec, method='iterative')

        Wdiff = abs(W - Wclen)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


if __name__ == "__main__":
    run_module_suite()
