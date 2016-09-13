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

from qutip import (sigmax, sigmay, sigmaz, sigmam, identity, basis,
                   mesolve, brmesolve, destroy, ket2dm, expect, tensor, fock)


def testTLS():
    """
    brmesolve: simple qubit
    """

    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam()]
    a_ops = [sigmax()]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                         spectra_cb=[lambda w: gamma * (w >= 0)])

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def testCOPS():
    """
    brmesolve: c_ops alone
    """

    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam()]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, [], e_ops,
                         spectra_cb=[], c_ops=c_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def testCOPSwithAOPS():
    """
    brmesolve: c_ops with a_ops
    """

    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam(), np.sqrt(gamma) * sigmaz()]
    c_ops_brme = [np.sqrt(gamma) * sigmaz()]
    a_ops = [sigmax()]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                         spectra_cb=[lambda w: gamma * (w >= 0)],
                         c_ops=c_ops_brme)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def testHOZeroTemperature():
    """
    brmesolve: harmonic oscillator, zero temperature
    """

    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    c_ops = [np.sqrt(kappa) * a]
    a_ops = [a + a.dag()]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                         spectra_cb=[lambda w: kappa * (w >= 0)])

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def testHOFiniteTemperature():
    """
    brmesolve: harmonic oscillator, finite temperature
    """

    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15
    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    n_th = 1.5
    w_th = w0/np.log(1 + 1/n_th)

    def S_w(w):
        if w >= 0:
            return (n_th + 1) * kappa
        else:
            return (n_th + 1) * kappa * np.exp(w / w_th)

    c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag()]
    a_ops = [a + a.dag()]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops, [S_w])

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def testHOFiniteTemperatureStates():
    """
    brmesolve: harmonic oscillator, finite temperature, states
    """

    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.25
    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    n_th = 1.5
    w_th = w0/np.log(1 + 1/n_th)

    def S_w(w):
        if w >= 0:
            return (n_th + 1) * kappa
        else:
            return (n_th + 1) * kappa * np.exp(w / w_th)

    c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag()]
    a_ops = [a + a.dag()]
    e_ops = []

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops, [S_w])

    n_me = expect(a.dag() * a, res_me.states)
    n_brme = expect(a.dag() * a, res_brme.states)

    diff = abs(n_me - n_brme).max()
    assert_(diff < 1e-2)


def testJCZeroTemperature():
    """
    brmesolve: Jaynes-Cummings model, zero temperature
    """

    N = 10
    a = tensor(destroy(N), identity(2))
    sm = tensor(identity(N), destroy(2))
    psi0 = ket2dm(tensor(basis(N, 1), basis(2, 0)))
    a_ops = [(a + a.dag())]
    e_ops = [a.dag() * a, sm.dag() * sm]

    w0 = 1.0 * 2 * np.pi
    g = 0.05 * 2 * np.pi
    kappa = 0.05
    times = np.linspace(0, 2 * 2 * np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0 * a.dag() * a + w0 * sm.dag() * sm + \
        g * (a + a.dag()) * (sm + sm.dag())

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                         spectra_cb=[lambda w: kappa * (w >= 0)])

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 5e-2)  # accept 5% error


def testTDPhononInitialization():
    """
    brmesolve: TLS model finite temperature, phonon-assisted initialization
    """

    # this test models phonon-assisted initialization with a quantum dot, such
    # as in P.-L. Ardelt, L. Hanschke, K. A. Fischer, et al.
    # Phys. Rev. B Condens. Matter 90, 241404 (2014).

    def pulse_shape(t):
        # gaussian pulse shape, ~10ps pulse
        return np.exp(-(t - 24) ** 2 / 105)

    def phonon_spectrum(w):
        # this spectrum is a displaced gaussian multiplied by w**3, which
        # models coupling to LA phonons. The electron and hole interactions
        # contribute constructively.
        import numpy

        # fitting parameters ae/ah
        ah = 1.9e-9  # m
        ae = 3.5e-9  # m
        # GaAs material parameters
        De = 7
        Dh = -3.5
        v = 5110  # m/s
        rho_m = 5370  # kg/m^3
        hbar = 1.05457173e-34  # Js
        T = 4.2  # Kelvin, temperature

        J = 1.6 * 1e-13 * w ** 3 / (
            4 * numpy.pi ** 2 * rho_m * hbar * v ** 5) * \
            (De * numpy.exp(-(w * 1e12 * ae / (2 * v)) ** 2) -
             Dh * numpy.exp(-(w * 1e12 * ah / (2 * v)) ** 2)) ** 2

        # results in peak at ~3THz angular frequency, w in THz
        # correct temperature dependence - 'negative' frequency components
        # correspond to absorption vs emission.
        if w > 0:
            JT_p = J * (1 + numpy.exp(-w * 0.6582119 / (T * 0.086173)) /
                        (1 - numpy.exp(-w * 0.6582119 / (T * 0.086173))))
            return JT_p
        elif w < 0:
            JT_m = -J * numpy.exp(w * 0.6582119 / (T * 0.086173)) / \
                   (1 - numpy.exp(w * 0.6582119 / (T * 0.086173)))
            return JT_m
        else:
            return 0

    tlist = np.linspace(0, 50, 40)  # tmax a bit over 2x FWHM

    # define system operators
    sm = destroy(2)
    H0 = -0.35 * sm.dag() * sm
    HI = sm + sm.dag()
    H = [H0, [HI, pulse_shape(tlist)]]

    # the
    a_ops = [sm.dag() * sm]
    spectra_cb = [phonon_spectrum]

    state0 = fock(2, 0)
    e_ops = [sm.dag() * sm]

    # test with constant and time-dependent c_ops (included for test purposes but
    # doesn't have physical meaning to the system)
    c_ops_0 = [0.1 * sm, [sm.dag() * sm, pulse_shape(tlist)]]
    result_1 = brmesolve(H, state0, tlist, a_ops,
                         e_ops=e_ops, c_ops=c_ops_0, spectra_cb=spectra_cb)

    # test with time-dependent c_ops only (included for test purposes but
    # doesn't have physical meaning to the system)
    c_ops_1 = [[sm.dag() * sm, pulse_shape(tlist)]]
    result_2 = brmesolve(H, state0, tlist, a_ops,
                         e_ops=e_ops, c_ops=c_ops_1, spectra_cb=spectra_cb)

    # test with no c_ops -> this is the model that has physical meaning
    c_ops_2 = []
    result_3 = brmesolve(H, state0, tlist, a_ops,
                         e_ops=e_ops, c_ops=c_ops_2, spectra_cb=spectra_cb)
    results = np.array([result_1.expect[0][-1], result_2.expect[0][-1],
                        result_3.expect[0][-1]])

    print(results)
    expected_results = np.array([0.552, 0.633, 0.861])
    max_diff = max(abs(results - expected_results))
    assert_(max_diff < 1e-3)  # accept only a very small error


if __name__ == "__main__":
    run_module_suite()
