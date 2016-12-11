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
from numpy.testing import assert_equal, run_module_suite

from qutip import (jmat, basis, destroy, create, displace, qeye, 
                    num, squeeze, charge, tunneling)


def test_jmat_12():
    "Spin 1/2 operators"
    spinhalf = jmat(1 / 2.)

    paulix = np.array([[0.0 + 0.j, 0.5 + 0.j], [0.5 + 0.j, 0.0 + 0.j]])
    pauliy = np.array([[0. + 0.j, 0. - 0.5j], [0. + 0.5j, 0. + 0.j]])
    pauliz = np.array([[0.5 + 0.j, 0.0 + 0.j], [0.0 + 0.j, -0.5 + 0.j]])
    sigmap = np.array([[0. + 0.j, 1. + 0.j], [0. + 0.j, 0. + 0.j]])
    sigmam = np.array([[0. + 0.j, 0. + 0.j], [1. + 0.j, 0. + 0.j]])

    assert_equal(np.allclose(spinhalf[0].full(), paulix), True)
    assert_equal(np.allclose(spinhalf[1].full(), pauliy), True)
    assert_equal(np.allclose(spinhalf[2].full(), pauliz), True)
    assert_equal(np.allclose(jmat(1 / 2., '+').full(), sigmap), True)
    assert_equal(np.allclose(jmat(1 / 2., '-').full(), sigmam), True)


def test_jmat_32():
    "Spin 3/2 operators"
    spin32 = jmat(3 / 2.)

    paulix32 = np.array(
        [[0.0000000 + 0.j, 0.8660254 + 0.j, 0.0000000 + 0.j, 0.0000000 + 0.j],
         [0.8660254 + 0.j, 0.0000000 + 0.j, 1.0000000 + 0.j, 0.0000000 + 0.j],
         [0.0000000 + 0.j, 1.0000000 + 0.j, 0.0000000 + 0.j, 0.8660254 + 0.j],
         [0.0000000 + 0.j, 0.0000000 + 0.j, 0.8660254 + 0.j, 0.0000000 + 0.j]])

    pauliy32 = np.array(
        [[0. + 0.j, 0. - 0.8660254j, 0. + 0.j, 0. + 0.j],
         [0. + 0.8660254j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
         [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. - 0.8660254j],
         [0. + 0.j, 0. + 0.j, 0. + 0.8660254j, 0. + 0.j]])

    pauliz32 = np.array([[1.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.0 + 0.j, -0.5 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -1.5 + 0.j]])

    assert_equal(np.allclose(spin32[0].full(), paulix32), True)
    assert_equal(np.allclose(spin32[1].full(), pauliy32), True)
    assert_equal(np.allclose(spin32[2].full(), pauliz32), True)


def test_jmat_42():
    "Spin 2 operators"
    spin42 = jmat(4 / 2., '+')
    assert_equal(spin42.dims == [[5], [5]], True)


def test_jmat_52():
    "Spin 5/2 operators"
    spin52 = jmat(5 / 2., '+')
    assert_equal(spin52.shape == (6, 6), True)


def test_destroy():
    "Destruction operator"
    b4 = basis(5, 4)
    d5 = destroy(5)
    test1 = d5 * b4
    assert_equal(np.allclose(test1.full(), 2.0 * basis(5, 3).full()), True)
    d3 = destroy(3)
    matrix3 = np.array(
        [[0.00000000 + 0.j, 1.00000000 + 0.j, 0.00000000 + 0.j],
         [0.00000000 + 0.j, 0.00000000 + 0.j, 1.41421356 + 0.j],
         [0.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j]])

    assert_equal(np.allclose(matrix3, d3.full()), True)


def test_create():
    "Creation operator"
    b3 = basis(5, 3)
    c5 = create(5)
    test1 = c5 * b3
    assert_equal(np.allclose(test1.full(), 2.0 * basis(5, 4).full()), True)
    c3 = create(3)
    matrix3 = np.array(
        [[0.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j],
         [1.00000000 + 0.j, 0.00000000 + 0.j, 0.00000000 + 0.j],
         [0.00000000 + 0.j, 1.41421356 + 0.j, 0.00000000 + 0.j]])

    assert_equal(np.allclose(matrix3, c3.full()), True)


def test_qeye():
    "Identity operator"
    eye3 = qeye(5)
    assert_equal(np.allclose(eye3.full(), np.eye(5, dtype=complex)), True)


def test_qeye_dims():
    "Identity operator"
    eye24 = qeye([2, 3, 4])
    assert_equal(np.allclose(eye24.full(), np.eye(24, dtype=complex)), True)
    assert_equal(eye24.dims, [[2, 3, 4], [2, 3, 4]])


def test_num():
    "Number operator"
    n5 = num(5)
    assert_equal(
        np.allclose(n5.full(),
                    np.diag([0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j])),
        True)


def test_squeeze():
    "Squeezing operator"
    sq = squeeze(4, 0.1 + 0.1j)
    sqmatrix = np.array([[0.99500417 + 0.j, 0.00000000 + 0.j,
                          0.07059289 - 0.07059289j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, 0.98503746 + 0.j,
                          0.00000000 + 0.j, 0.12186303 - 0.12186303j],
                         [-0.07059289 - 0.07059289j, 0.00000000 + 0.j,
                          0.99500417 + 0.j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, -0.12186303 - 0.12186303j,
                          0.00000000 + 0.j, 0.98503746 + 0.j]])

    assert_equal(np.allclose(sq.full(), sqmatrix), True)


def test_displace():
    "Displacement operator"
    dp = displace(4, 0.25)
    dpmatrix = np.array(
        [[0.96923323 + 0.j, -0.24230859 + 0.j, 0.04282883 + 0.j, -
          0.00626025 + 0.j],
         [0.24230859 + 0.j, 0.90866411 + 0.j, -0.33183303 +
          0.j, 0.07418172 + 0.j],
         [0.04282883 + 0.j, 0.33183303 + 0.j, 0.84809499 +
          0.j, -0.41083747 + 0.j],
         [0.00626025 + 0.j, 0.07418172 + 0.j, 0.41083747 + 0.j,
          0.90866411 + 0.j]])

    assert_equal(np.allclose(dp.full(), dpmatrix), True)


def test_charge():
    "Charge operator"
    N = 5
    M = - np.random.randint(N)
    ch = charge(N,M)
    ch_matrix = np.diag(np.arange(M,N+1))
    assert_equal(np.allclose(ch.full(), ch_matrix), True)


def test_tunneling():
    "Tunneling operator"
    N = 5
    tn = tunneling(2*N+1)
    tn_matrix = np.diag(np.ones(2*N),k=-1) + np.diag(np.ones(2*N),k=1)
    assert_equal(np.allclose(tn.full(), tn_matrix), True) 
    
    tn = tunneling(2*N+1,2)
    tn_matrix = np.diag(np.ones(2*N-1),k=-2) + np.diag(np.ones(2*N-1),k=2)
    assert_equal(np.allclose(tn.full(), tn_matrix), True)


if __name__ == "__main__":
    run_module_suite()
