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

from functools import partial
from numpy import allclose, linspace, mean, ones
from numpy.testing import assert_, run_module_suite

from qutip import *


class TestBRMESolve:
    """
    Test for the Bloch-Redfield master equation.
    """

    def testTLS(self):
        "brmesolve: qubit"

        delta = 0.0 * 2 * pi
        epsilon = 0.5 * 2 * pi
        gamma = 0.25
        times = linspace(0, 10, 100)
        H = delta/2 * sigmax() + epsilon/2 * sigmaz()
        psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
        c_ops = [sqrt(gamma) * sigmam()]
        a_ops = [sigmax()]
        e_ops = [sigmax(), sigmay(), sigmaz()]
        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                             spectra_cb=[lambda w: gamma * (w >= 0)])

        for idx, e in enumerate(e_ops):
            diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
            assert_(diff < 1e-2)

    def testHOZeroTemperature(self):
        "brmesolve: harmonic oscillator, zero temperature"

        N = 10
        w0 = 1.0 * 2 * pi
        g = 0.05 * w0
        kappa = 0.15

        times = linspace(0, 25, 1000)
        a = destroy(N)
        H = w0 * a.dag() * a + g * (a + a.dag())
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

        c_ops = [sqrt(kappa) * a]
        a_ops = [a + a.dag()]
        e_ops = [a.dag() * a, a + a.dag()]

        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                             spectra_cb=[lambda w: kappa * (w >= 0)])

        for idx, e in enumerate(e_ops):
            diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
            assert_(diff < 1e-2)

    def testHOFiniteTemperature(self):
        "brmesolve: harmonic oscillator, finite temperature"

        N = 10
        w0 = 1.0 * 2 * pi
        g = 0.05 * w0
        kappa = 0.15
        times = linspace(0, 25, 1000)
        a = destroy(N)
        H = w0 * a.dag() * a + g * (a + a.dag())
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

        n_th = 1.5
        w_th = w0/log(1 + 1/n_th)

        def S_w(w):
            if w >= 0:
                return (n_th + 1) * kappa
            else:
                return (n_th + 1) * kappa * exp(w / w_th)

        c_ops = [sqrt(kappa * (n_th + 1)) * a, sqrt(kappa * n_th) * a.dag()]
        a_ops = [a + a.dag()]
        e_ops = [a.dag() * a, a + a.dag()]

        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops, [S_w])

        for idx, e in enumerate(e_ops):
            diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
            assert_(diff < 1e-2)

    def testHOFiniteTemperatureStates(self):
        "brmesolve: harmonic oscillator, finite temperature, states"

        N = 10
        w0 = 1.0 * 2 * pi
        g = 0.05 * w0
        kappa = 0.25
        times = linspace(0, 25, 1000)
        a = destroy(N)
        H = w0 * a.dag() * a + g * (a + a.dag())
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

        n_th = 1.5
        w_th = w0/log(1 + 1/n_th)

        def S_w(w):
            if w >= 0:
                return (n_th + 1) * kappa
            else:
                return (n_th + 1) * kappa * exp(w / w_th)

        c_ops = [sqrt(kappa * (n_th + 1)) * a, sqrt(kappa * n_th) * a.dag()]
        a_ops = [a + a.dag()]
        e_ops = []

        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops, [S_w])

        n_me = expect(a.dag() * a, res_me.states)
        n_brme = expect(a.dag() * a, res_brme.states)

        diff = abs(n_me - n_brme).max()
        assert_(diff < 1e-2)

    def testJCZeroTemperature(self):
        "brmesolve: Jaynes-Cummings model, zero temperature"

        N = 10
        a = tensor(destroy(N), identity(2))
        sm = tensor(identity(N), destroy(2))
        psi0 = ket2dm(tensor(basis(N, 1), basis(2, 0)))
        a_ops = [(a + a.dag())]
        e_ops = [a.dag() * a, sm.dag() * sm]

        w0 = 1.0 * 2 * pi
        g = 0.05 * 2 * pi
        kappa = 0.05
        times = linspace(0, 2 * 2 * pi / g, 1000)

        c_ops = [sqrt(kappa) * a]
        H = w0 * a.dag() * a + w0 * sm.dag() * sm + \
            g * (a + a.dag()) * (sm + sm.dag())

        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                             spectra_cb=[lambda w: kappa * (w >= 0)])

        for idx, e in enumerate(e_ops):
            diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
            assert_(diff < 5e-2)  # accept 5% error


if __name__ == "__main__":
    run_module_suite()
