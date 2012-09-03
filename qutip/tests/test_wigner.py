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
# Copyright (C) 2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from numpy.testing import assert_, run_module_suite
from qutip import *
from scipy.special import laguerre

def test_wigner_coherent():
    "wigner: test wigner function calculation for coherent states"
    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X,Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1]-xvec[0]
    dy = yvec[1]-yvec[0]

    N = 20
    beta = rand() + rand() * 1.0j
    psi = coherent(N, beta)


    # calculate the wigner function using qutip and analytic formula
    W_qutip = wigner(psi, xvec, yvec, g=2)
    W_analytic = 2/pi * exp(-2*abs(a-beta)**2)

    # check difference
    assert_(sum(abs(W_qutip - W_analytic)**2) < 1e-4)

    # check normalization
    assert_(sum(W_qutip)    * dx * dy - 1.0 < 1e-8)
    assert_(sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_fock():
    "wigner: test wigner function calculation for Fock states"

    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X,Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1]-xvec[0]
    dy = yvec[1]-yvec[0]

    N = 15

    for n in [2,3,4,5,6]:

        psi = fock(N, n)

        # calculate the wigner function using qutip and analytic formula
        W_qutip = wigner(psi, xvec, yvec, g=2)
        W_analytic = 2/pi * (-1)**n * exp(-2*abs(a)**2) * polyval(laguerre(n), 4*abs(a)**2)

        # check difference
        assert_(sum(abs(W_qutip - W_analytic)) < 1e-4)

        # check normalization
        assert_(sum(W_qutip)    * dx * dy - 1.0 < 1e-8)
        assert_(sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_dm():
    "wigner: compare wigner methods for random density matrices"

    xvec = linspace(-5.0, 5.0, 100)
    yvec = xvec

    X,Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1]-xvec[0]
    dy = yvec[1]-yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        rho = rand_dm(N, 0.5 + rand()/2)

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

    X,Y = meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1]-xvec[0]
    dy = yvec[1]-yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        psi = rand_ket(N, 0.5 + rand()/2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(psi, xvec, yvec, g=2)
        W_qutip2 = wigner(psi, xvec, yvec, g=2, method='laguerre')

        # check difference
        assert_(sum(abs(W_qutip1 - W_qutip1)) < 1e-4)

        # check normalization
        assert_(sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert_(sum(W_qutip2) * dx * dy - 1.0 < 1e-8)



if __name__ == "__main__":
    run_module_suite()

