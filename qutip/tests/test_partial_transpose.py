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
"""
Unit tests for QuTiP partial transpose functions.
"""

import numpy as np
from numpy.testing import assert_, run_module_suite

from qutip import Qobj, partial_transpose, tensor, rand_dm
from qutip.partial_transpose import _partial_transpose_reference


def test_partial_transpose_bipartite():
    """partial transpose of bipartite systems"""

    rho = Qobj(np.arange(16).reshape(4, 4), dims=[[2, 2], [2, 2]])

    # no transpose
    rho_pt = partial_transpose(rho, [0, 0])
    assert_(np.abs(np.max(rho_pt.full() - rho.full())) < 1e-12)

    # partial transpose subsystem 1
    rho_pt = partial_transpose(rho, [1, 0])
    rho_pt_expected = np.array([[0, 1,  8,  9],
                                [4, 5, 12, 13],
                                [2, 3, 10, 11],
                                [6, 7, 14, 15]])
    assert_(np.abs(np.max(rho_pt.full() - rho_pt_expected)) < 1e-12)

    # partial transpose subsystem 2
    rho_pt = partial_transpose(rho, [0, 1])
    rho_pt_expected = np.array([[0, 4, 2, 6],
                                [1, 5, 3, 7],
                                [8, 12, 10, 14],
                                [9, 13, 11, 15]])
    assert_(np.abs(np.max(rho_pt.full() - rho_pt_expected)) < 1e-12)

    # full transpose
    rho_pt = partial_transpose(rho, [1, 1])
    assert_(np.abs(np.max(rho_pt.full() - rho.trans().full())) < 1e-12)


def test_partial_transpose_comparison():
    """partial transpose: comparing sparse and dense implementations"""

    N = 10
    rho = tensor(rand_dm(N, density=0.5), rand_dm(N, density=0.5))

    # partial transpose of system 1
    rho_pt1 = partial_transpose(rho, [1, 0], method="dense")
    rho_pt2 = partial_transpose(rho, [1, 0], method="sparse")
    np.abs(np.max(rho_pt1.full() - rho_pt1.full())) < 1e-12

    # partial transpose of system 2
    rho_pt1 = partial_transpose(rho, [0, 1], method="dense")
    rho_pt2 = partial_transpose(rho, [0, 1], method="sparse")
    np.abs(np.max(rho_pt1.full() - rho_pt2.full())) < 1e-12


def test_partial_transpose_randomized():
    """partial transpose: randomized tests on tripartite system"""

    rho = tensor(rand_dm(2, density=1),
                 rand_dm(2, density=1),
                 rand_dm(2, density=1))

    mask = np.random.randint(2, size=3)

    rho_pt_ref = _partial_transpose_reference(rho, mask)

    rho_pt1 = partial_transpose(rho, mask, method="dense")
    np.abs(np.max(rho_pt1.full() - rho_pt_ref.full())) < 1e-12

    rho_pt2 = partial_transpose(rho, mask, method="sparse")
    np.abs(np.max(rho_pt2.full() - rho_pt_ref.full())) < 1e-12


if __name__ == "__main__":
    run_module_suite()
