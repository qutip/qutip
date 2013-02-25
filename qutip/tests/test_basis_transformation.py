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

import sys
from qutip import *
from numpy.testing import assert_equal, assert_, run_module_suite
import scipy


def test_Transformation1():
    "Transform 2-level to eigenbasis and back"
    H1 = scipy.rand() * sigmax() + scipy.rand() * sigmay() + \
        scipy.rand() * sigmaz()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation2():
    "Transform 10-level real-values to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation3():
    "Transform 10-level to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5 - scipy.rand(N, N)) + 1j * (0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation4():
    "Transform 10-level imag to eigenbasis and back"
    N = 10
    H1 = Qobj(1j * (0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation5():
    "Consistency between transformations of kets and denstity matrices"

    N = 4
    psi0 = rand_ket(N)

    # generate a random basis
    evals, rand_basis = rand_dm(N, density=1).eigenstates()

    rho1 = ket2dm(psi0).transform(rand_basis, True)
    rho2 = ket2dm(psi0.transform(rand_basis, True))

    assert_((rho1 - rho2).norm() < 1e-6)


if __name__ == "__main__":
    run_module_suite()
