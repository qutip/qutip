# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

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
                                spectra_cb=[lambda w : gamma * (w >= 0)])

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
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N,0)).unit())

        c_ops = [sqrt(kappa) * a]
        a_ops = [a + a.dag()]
        e_ops = [a.dag() * a, a + a.dag()]

        res_me = mesolve(H, psi0, times, c_ops, e_ops)
        res_brme = brmesolve(H, psi0, times, a_ops, e_ops,
                                spectra_cb=[lambda w : kappa * (w >= 0)])

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
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N,0)).unit())

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
        psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N,0)).unit())

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
                                spectra_cb=[lambda w : kappa * (w >= 0)])

        for idx, e in enumerate(e_ops):
            diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
            assert_(diff < 5e-2)  # accept 5% error


if __name__ == "__main__":
    run_module_suite()
