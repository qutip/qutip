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

import numbers
import numpy as np
import pytest
import qutip


N = 5


def test_jmat_12():
    spinhalf = qutip.jmat(1 / 2.)

    paulix = np.array([[0.0 + 0.j, 0.5 + 0.j], [0.5 + 0.j, 0.0 + 0.j]])
    pauliy = np.array([[0. + 0.j, 0. - 0.5j], [0. + 0.5j, 0. + 0.j]])
    pauliz = np.array([[0.5 + 0.j, 0.0 + 0.j], [0.0 + 0.j, -0.5 + 0.j]])
    sigmap = np.array([[0. + 0.j, 1. + 0.j], [0. + 0.j, 0. + 0.j]])
    sigmam = np.array([[0. + 0.j, 0. + 0.j], [1. + 0.j, 0. + 0.j]])

    np.testing.assert_allclose(spinhalf[0].full(), paulix)
    np.testing.assert_allclose(spinhalf[1].full(), pauliy)
    np.testing.assert_allclose(spinhalf[2].full(), pauliz)
    np.testing.assert_allclose(qutip.jmat(1 / 2., '+').full(), sigmap)
    np.testing.assert_allclose(qutip.jmat(1 / 2., '-').full(), sigmam)

    np.testing.assert_allclose(qutip.spin_Jx(1 / 2.).full(), paulix)
    np.testing.assert_allclose(qutip.spin_Jy(1 / 2.).full(), pauliy)
    np.testing.assert_allclose(qutip.spin_Jz(1 / 2.).full(), pauliz)
    np.testing.assert_allclose(qutip.spin_Jp(1 / 2.).full(), sigmap)
    np.testing.assert_allclose(qutip.spin_Jm(1 / 2.).full(), sigmam)

    np.testing.assert_allclose(qutip.sigmax().full(), paulix * 2)
    np.testing.assert_allclose(qutip.sigmay().full(), pauliy * 2)
    np.testing.assert_allclose(qutip.sigmaz().full(), pauliz * 2)
    np.testing.assert_allclose(qutip.sigmap().full(), sigmap)
    np.testing.assert_allclose(qutip.sigmam().full(), sigmam)


def test_jmat_32():
    spin32 = qutip.jmat(3 / 2.)

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

    np.testing.assert_allclose(spin32[0].full(), paulix32)
    np.testing.assert_allclose(spin32[1].full(), pauliy32)
    np.testing.assert_allclose(spin32[2].full(), pauliz32)


@pytest.mark.parametrize(['spin', 'N'], [
    pytest.param(3/2., 4, id="1.5"),
    pytest.param(5/2., 6, id="2.5"),
    pytest.param(3.0, 7, id="3.0"),
])
def test_jmat_dims(spin, N):
    spin_mat = qutip.jmat(spin, '+')
    assert spin_mat.dims == [[N], [N]]
    assert spin_mat.shape == (N, N)


@pytest.mark.parametrize(['oper_func', 'diag', 'offset', 'args'], [
    pytest.param(qutip.destroy, np.arange(1, N)**0.5, 1, (), id="destroy"),
    pytest.param(qutip.destroy, np.arange(6, N+5)**0.5, 1, (5,),
                 id="destroy_offset"),
    pytest.param(qutip.create, np.arange(1, N)**0.5, -1, (), id="create"),
    pytest.param(qutip.create, np.arange(6, N+5)**0.5, -1, (5,),
                 id="create_offset"),
    pytest.param(qutip.num, np.arange(N), 0, (), id="num"),
    pytest.param(qutip.num, np.arange(5, N+5), 0, (5,), id="num_offset"),
    pytest.param(qutip.charge, np.arange(-N, N+1), 0, (), id="charge"),
    pytest.param(qutip.charge, np.arange(2, N+1)/3, 0, (2, 1/3),
                 id="charge_args"),
])
def test_diagonal_oper(oper_func, diag, offset, args):
    oper = oper_func(N, *args)
    assert oper == qutip.Qobj(np.diag(diag, offset))


@pytest.mark.parametrize(['function', 'message'], [
    (qutip.qeye, "All dimensions must be integers >= 0"),
    (qutip.destroy, "Hilbert space dimension must be integer value"),
    (qutip.create, "Hilbert space dimension must be integer value"),
], ids=["qeye", "destroy", "create"])
def test_diagonal_raise(function, message):
    with pytest.raises(ValueError) as e:
        function(2.5)
    assert str(e.value) == message


@pytest.mark.parametrize("to_test", [qutip.qzero, qutip.qeye, qutip.identity])
@pytest.mark.parametrize("dimensions", [
        2,
        [2],
        [2, 3, 4],
        1,
        [1],
        [1, 1],
    ])
def test_implicit_tensor_creation(to_test, dimensions):
    implicit = to_test(dimensions)
    if isinstance(dimensions, numbers.Integral):
        dimensions = [dimensions]
    assert implicit.dims == [dimensions, dimensions]


@pytest.mark.parametrize("to_test", [qutip.qzero, qutip.qeye, qutip.identity])
def test_super_operator_creation(to_test):
    size = 2
    implicit = to_test([[size], [size]])
    explicit = qutip.to_super(to_test(size))
    assert implicit == explicit


@pytest.mark.parametrize(["to_test", "factor", "phase"], [
    (qutip.position, 1, 1),
    (qutip.momentum, 1j, -1),
])
def test_bidiagonal(to_test, factor, phase):
    N = 5
    operator = to_test(N)
    expected = (np.diag((np.arange(1, N) / 2)**0.5, k=-1) +
                np.diag((np.arange(1, N) / 2)**0.5, k=1) * phase) * factor
    np.testing.assert_allclose(operator.full(), expected)


def test_squeeze():
    sq = qutip.squeeze(4, 0.1 + 0.1j)
    sqmatrix = np.array([[0.99500417 + 0.j, 0.00000000 + 0.j,
                          0.07059289 - 0.07059289j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, 0.98503746 + 0.j,
                          0.00000000 + 0.j, 0.12186303 - 0.12186303j],
                         [-0.07059289 - 0.07059289j, 0.00000000 + 0.j,
                          0.99500417 + 0.j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, -0.12186303 - 0.12186303j,
                          0.00000000 + 0.j, 0.98503746 + 0.j]])
    np.testing.assert_allclose(sq.full(), sqmatrix, atol=1e-8)


def test_squeezing():
    squeeze = qutip.squeeze(4, 0.1 + 0.1j)
    a = qutip.destroy(4)
    squeezing = qutip.squeezing(a, a, 0.1 + 0.1j)
    assert squeeze == squeezing


def test_displace():
    dp = qutip.displace(4, 0.25)
    dpmatrix = np.array(
        [[0.96923323 + 0.j, -0.24230859 + 0.j, 0.04282883 + 0.j, -
          0.00626025 + 0.j],
         [0.24230859 + 0.j, 0.90866411 + 0.j, -0.33183303 +
          0.j, 0.07418172 + 0.j],
         [0.04282883 + 0.j, 0.33183303 + 0.j, 0.84809499 +
          0.j, -0.41083747 + 0.j],
         [0.00626025 + 0.j, 0.07418172 + 0.j, 0.41083747 + 0.j,
          0.90866411 + 0.j]])
    np.testing.assert_allclose(dp.full(), dpmatrix, atol=1e-8)


@pytest.mark.parametrize("shift", [1, 2])
def test_tunneling(shift):
    N = 10
    tn = qutip.tunneling(N, shift)
    tn_matrix = np.diag(np.ones(N - shift), k=shift)
    tn_matrix += tn_matrix.T
    np.testing.assert_allclose(tn.full(), tn_matrix)


def test_commutator():
    A = qutip.qeye(N)
    B = qutip.destroy(N)
    assert qutip.commutator(A, B) == qutip.qzero(N)

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    assert qutip.commutator(sx, sy) / 2 == (qutip.sigmaz() * 1j)
