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
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

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
    "correlation: compare spectrum obtained for eseries and fft methods"

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

if __name__ == "__main__":
    run_module_suite()
