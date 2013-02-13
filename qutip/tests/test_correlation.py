# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import scipy
import time
from numpy.testing import assert_, run_module_suite

from qutip import *

def _func(x):
    time.sleep(scipy.rand() * 0.25) # random delay
    return x**2


def test_compare_solvers_coherent_state():
    "comparing me and es for oscillator in coherent initial state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1*(1+n_th)) * a, sqrt(G1*n_th) * a.dag()]
    rho0 = coherent_dm(N, sqrt(4.0))

    taulist = linspace(0, 5.0, 100)
    corr1 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="es")

    print "norm =", norm(corr1-corr2)

    assert_(norm(corr1-corr2) < 1e-3)


def test_compare_solvers_steadystate():
    "comparing me and es for oscillator in steady state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1*(1+n_th)) * a, sqrt(G1*n_th) * a.dag()]

    taulist = linspace(0, 5.0, 100)
    corr1 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="es")

    print "norm =", norm(corr1-corr2)

    assert_(norm(corr1-corr2) < 1e-3)

if __name__ == "__main__":
    run_module_suite()
