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
from numpy import amax
from numpy.testing import assert_equal, run_module_suite
import scipy


def test_diagHamiltonian1():
    """
    Diagonalization of random two-level system
    """

    H = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()

    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)


def test_diagHamiltonian2():
    """
    Diagonalization of composite systems
    """

    H1 = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()
    H2 = scipy.rand() * sigmax() + scipy.rand() * sigmay() +\
        scipy.rand() * sigmaz()

    H = tensor(H1, H2)

    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)

    N1 = 10
    N2 = 2

    a1 = tensor(destroy(N1), qeye(N2))
    a2 = tensor(qeye(N1), destroy(N2))
    H = scipy.rand() * a1.dag() * a1 + scipy.rand() * a2.dag() * a2 + \
        scipy.rand() * (a1 + a1.dag()) * (a2 + a2.dag())
    evals, ekets = H.eigenstates()

    for n in range(len(evals)):
        # assert that max(H * ket - e * ket) is small
        assert_equal(amax(
            abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10, True)


if __name__ == "__main__":
    run_module_suite()
