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
"""
Unit tests for QuTiP partial transpose functions.
"""

import numpy
from numpy.testing import assert_, assert_equal, run_module_suite

from qutip import *
from qutip.partial_transpose import _partial_transpose_reference


def test_partial_transpose_bipartite():
    """partial transpose of bipartite systems"""

    rho = Qobj(arange(16).reshape(4, 4), dims=[[2, 2], [2, 2]])

    # no transpose
    rho_pt = partial_transpose(rho, [0, 0])
    assert_(numpy.abs(numpy.max(rho_pt.full() - rho.full())) < 1e-12)

    # partial transpose subsystem 1
    rho_pt = partial_transpose(rho, [1, 0])
    rho_pt_expected = array([[0, 1,  8,  9],
                             [4, 5, 12, 13],
                             [2, 3, 10, 11],
                             [6, 7, 14, 15]])
    assert_(numpy.abs(numpy.max(rho_pt.full() - rho_pt_expected)) < 1e-12)

    # partial transpose subsystem 2
    rho_pt = partial_transpose(rho, [0, 1])
    rho_pt_expected = array([[0, 4, 2, 6],
                             [1, 5, 3, 7],
                             [8, 12, 10, 14],
                             [9, 13, 11, 15]])
    assert_(numpy.abs(numpy.max(rho_pt.full() - rho_pt_expected)) < 1e-12)

    # full transpose
    rho_pt = partial_transpose(rho, [1, 1])
    assert_(numpy.abs(numpy.max(rho_pt.full() - rho.trans().full())) < 1e-12)


def test_partial_transpose_comparison():
    """partial transpose: comparing sparse and dense implementations"""

    N = 10
    rho = tensor(rand_dm(N, density=0.5), rand_dm(N, density=0.5))

    # partial transpose of system 1
    rho_pt1 = partial_transpose(rho, [1, 0], method="dense")
    rho_pt2 = partial_transpose(rho, [1, 0], method="sparse")
    numpy.abs(numpy.max(rho_pt1.full() - rho_pt1.full())) < 1e-12

    # partial transpose of system 2
    rho_pt1 = partial_transpose(rho, [0, 1], method="dense")
    rho_pt2 = partial_transpose(rho, [0, 1], method="sparse")
    numpy.abs(numpy.max(rho_pt1.full() - rho_pt1.full())) < 1e-12


def test_partial_transpose_randomized():
    """partial transpose: randomized tests on tripartite system"""

    rho = tensor(rand_dm(2, density=1),
                 rand_dm(2, density=1),
                 rand_dm(2, density=1))

    mask = numpy.random.randint(2, size=3)

    rho_pt_ref = _partial_transpose_reference(rho, mask)

    rho_pt1 = partial_transpose(rho, mask, method="dense")
    numpy.abs(numpy.max(rho_pt1.full() - rho_pt_ref.full())) < 1e-12

    rho_pt2 = partial_transpose(rho, mask, method="sparse")
    numpy.abs(numpy.max(rho_pt2.full() - rho_pt_ref.full())) < 1e-12


if __name__ == "__main__":
    run_module_suite()
