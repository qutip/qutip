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
import scipy
import time
from numpy.testing import assert_, run_module_suite

from qutip import *


def test_compare_solvers_coherent_state():
    "correlation: comparing me and es for oscillator in coherent initial state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1 * (1 + n_th)) * a, sqrt(G1 * n_th) * a.dag()]
    rho0 = coherent_dm(N, sqrt(4.0))

    taulist = np.linspace(0, 5.0, 100)
    corr1 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_steadystate():
    "correlation: comparing me and es for oscillator in steady state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1 * (1 + n_th)) * a, sqrt(G1 * n_th) * a.dag()]

    taulist = np.linspace(0, 5.0, 100)
    corr1 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_spectrum():
    "correlation: compare spectrum from eseries and fft methods"

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * pi
    g = 0.1 * 2 * pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [sqrt(kappa * (1 + n_th)) * a,
             sqrt(kappa * n_th) * a.dag(),
             sqrt(gamma) * sm]

    tlist = np.linspace(0, 100, 2500)
    corr = correlation_ss(H, tlist, c_ops, a.dag(), a)
    wlist1, spec1 = spectrum_correlation_fft(tlist, corr)
    spec2 = spectrum_ss(H, wlist1, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)


def test_spectrum():
    "correlation: compare spectrum from eseries and pseudo-inverse methods"

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * pi
    g = 0.1 * 2 * pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [sqrt(kappa * (1 + n_th)) * a,
             sqrt(kappa * n_th) * a.dag(),
             sqrt(gamma) * sm]

    wlist = 2 * pi * np.linspace(0.5, 1.5, 100)
    spec1 = spectrum_ss(H, wlist, c_ops, a.dag(), a)
    spec2 = spectrum_pi(H, wlist, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)

if __name__ == "__main__":
    run_module_suite()
