# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Copyright 2020 United States Government as represented by the Administrator
#    of the National Aeronautics and Space Administration. All Rights Reserved.
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

import pytest
import itertools
import numpy as np
import qutip
from qutip.qip.operations import gates
from qutip.qip.operations.local_operations import local_multiply_dense
from qutip.core import data


class Test_local_multiply_dense:
    # these tests are designed to check the `local_multiply_dense` function.
    # we test left and right multiplication on ket, oper, and bra types.
    # we test explicitly 1,2, and 3-local operations.
    # we also test operations on qudit types.
    # for each test, we use each backend (einsum and a vectorized approach).
    # typically, we compare against results using `expand_operator`.

    @pytest.mark.parametrize('n', (1, 2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_left_multiply_matrix(self, n, backend):
        d = 2 ** n
        # array elements 1 through 2^2n
        unique_arr = qutip.Qobj(np.array(range(1, 1 + d**2)).reshape(d, d),
                                dims=[[2]*n, [2]*n])
        op = -1j * np.pi * 0.25 * (qutip.sigmax() + qutip.sigmaz()).expm()

        for t in range(n):
            actual = local_multiply_dense(op, unique_arr, targets=t,
                                          right=False, backend=backend)
            actual_arr = actual.full()

            full_op = gates.expand_operator(op, n, targets=t)
            expected = full_op * unique_arr
            expected_arr = expected.full()

            np.testing.assert_array_almost_equal(actual_arr, expected_arr)

    @pytest.mark.parametrize('n', (1, 2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_right_multiply_matrix(self, n, backend):
        d = 2 ** n
        # array elements 1 through 2^2n
        unique_arr = qutip.Qobj(np.array(range(1, 1 + d**2)).reshape(d, d),
                                dims=[[2]*n, [2]*n])
        op = -1j * np.pi * 0.25 * (qutip.sigmax() + qutip.sigmaz()).expm()

        for t in range(n):
            actual = local_multiply_dense(op, unique_arr, targets=t, right=True,
                                          backend=backend)

            full_op = gates.expand_operator(op, n, targets=t)
            expected = unique_arr * full_op
            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('n', (1, 2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_multiply_vector(self, n, backend):
        d = 2 ** n
        # array elements 1 through 2^n
        unique_vec = qutip.Qobj(np.array(range(1, 1 + d)).reshape(d, 1),
                                dims=[[2]*n, [1]*n])
        op = -1j * np.pi * 0.25 * (qutip.sigmax() + qutip.sigmaz()).expm()

        for t in range(n):
            actual = local_multiply_dense(op, unique_vec, targets=t,
                                          backend=backend)

            full_op = gates.expand_operator(op, n, targets=t)
            expected = full_op * unique_vec
            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('n', (1, 2))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_bra(self, n, backend):
        bra = qutip.tensor([qutip.ghz_state(1).dag()] * n)

        # full rank local op
        op = qutip.Qobj(np.array(range(1, 5)).reshape(2, 2))

        for t in range(n):
            actual = local_multiply_dense(op, bra, targets=t, right=True,
                                          backend=backend)
            expected = bra * gates.expand_operator(op, n, targets=t)

            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('n', (2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_two_local_multiply_matrix(self, n, backend):
        d = 2 ** n
        # array elements 1 through 2^2n
        unique_arr = qutip.Qobj(np.array(range(1, 1 + d**2)).reshape(d, d),
                                dims=[[2]*n, [2]*n])
        # two full rank matrices
        exp1 = (-1j * 0.25 * np.pi * qutip.sigmax()).expm()
        exp2 = (-1j * 0.25 * np.pi * qutip.sigmay()).expm()
        op = qutip.tensor(exp1, exp2)

        for t1, t2 in itertools.product(range(n), range(n)):
            if t1 == t2:
                continue
            full_op = gates.expand_operator(op, n, targets=[t1, t2])

            # left multiply
            expected = full_op * unique_arr
            actual = local_multiply_dense(op, unique_arr, targets=[t1, t2],
                                          right=False, backend=backend)

            np.testing.assert_array_almost_equal(actual.full(), expected.full())

            # right multiply
            expected = unique_arr * full_op
            actual = local_multiply_dense(op, unique_arr, targets=[t1, t2],
                                          right=True)
            np.testing.assert_array_almost_equal(actual.full(), expected.full())


    @pytest.mark.parametrize('n', (2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_two_local_multiply_vector(self, n, backend):
        d = 2 ** n
        # array elements 1 through 2^n
        unique_vec = qutip.Qobj(np.array(range(1, 1 + d)).reshape(d, 1),
                                dims=[[2]*n, [1]*n])
        op = qutip.tensor(qutip.sigmax(), qutip.sigmaz())

        for t1, t2 in itertools.product(range(n), range(n)):
            if t1 == t2:
                continue
            actual = local_multiply_dense(op, unique_vec, targets=[t1, t2],
                                          right=False, backend=backend)

            full_op = gates.expand_operator(op, n, targets=[t1, t2])
            expected = full_op * unique_vec
            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_three_local(self, backend):
        # this tests 3-local, left and right multiplication, on
        # a ket, operator, and bra
        n = 6
        d = 2 ** n
        v_dims = [[2] * n, [1] * n]
        # unnormalized state with unique elements
        psi = qutip.Qobj(np.array(range(1, 1 + d)), type='ket', dims=v_dims)
        rho = qutip.ket2dm(psi)

        # build 3-local operator
        exp1 = (-1j * 0.25 * np.pi * qutip.sigmax()).expm()
        exp2 = (-1j * 0.25 * np.pi * qutip.sigmay()).expm()
        exp3 = (-1j * 0.25 * np.pi * qutip.sigmaz()).expm()
        op = qutip.tensor(exp1, exp2, exp3)

        for targets in itertools.product(*[range(n)] * 3):
            if len(set(targets)) != 3:
                continue
            full_op = gates.expand_operator(op, n, targets=targets)

            for state in [psi, rho]:
                # left multiplication
                expected = full_op * state
                actual = local_multiply_dense(op, state, targets=targets,
                                              backend=backend)

                np.testing.assert_array_almost_equal(actual.full(), expected.full())

                # right multiplication
                # take dagger to convert ket to bra
                expected = state.dag() * full_op
                actual = local_multiply_dense(op, state.dag(), targets=targets,
                                              right=True)

                np.testing.assert_array_almost_equal(actual.full(),
                                                     expected.full())

    @pytest.mark.parametrize('d', (3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_qudit_vec(self, d, backend):
        d_vec = qutip.Qobj(np.array(range(1, 1 + d)).reshape(d, 1),
                           dims=[[d], [1]])

        # d x d operation
        op = qutip.Qobj(np.array(range(d**2)).reshape(d, d))

        actual = local_multiply_dense(op, d_vec, targets=0, right=False,
                                      backend=backend)
        expected = op * d_vec
        np.testing.assert_array_almost_equal(actual.full(), expected.full())

        # now test 1-local on system of 2 qudits.
        d2_vec = qutip.tensor(d_vec, d_vec)
        for t in range(2):
            full_op = gates.expand_operator(op, 2, targets=t, dims=[d, d])

            actual = local_multiply_dense(op, d2_vec, targets=t)
            expected = full_op * d2_vec
            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('d', (3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_qudit_mat(self, d, backend):
        d_mat = qutip.Qobj(np.array(range(1, 1 + d ** 2)).reshape(d, d),
                           dims=[[d], [d]])

        # d x d operation
        op = qutip.Qobj(np.array(range(d**2)).reshape(d, d))

        # left mult
        actual = local_multiply_dense(op, d_mat, targets=0, right=False,
                                      backend=backend)
        expected = op * d_mat
        np.testing.assert_array_almost_equal(actual.full(), expected.full())

        # right mult
        actual = local_multiply_dense(op, d_mat, targets=0, right=True,
                                      backend=backend)
        expected = d_mat * op
        np.testing.assert_array_almost_equal(actual.full(), expected.full())

    @pytest.mark.parametrize('n', (2, 3, 4))
    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_different_dims(self, n, backend):
        qubit = qutip.Qobj(np.array(range(1, 3)))
        qutrit = qutip.Qobj(np.array(range(4, 7)))
        qu4it = qutip.Qobj(np.array(range(8, 12)))
        vecs = [qubit, qutrit, qu4it]

        qut4it_op = np.array(range(16)).reshape(4, 4)
        qutrit_op = qutip.Qobj(qut4it_op[0:3, 0: 3])
        qubit_op = qutip.Qobj(qut4it_op[0: 2, 0: 2])
        qut4it_op = qutip.Qobj(qut4it_op)
        ops = [qubit_op, qutrit_op, qut4it_op]

        for ind in itertools.product(*[range(3)] * n):
            vec = qutip.tensor([vecs[i] for i in ind])
            mat = qutip.ket2dm(vec)
            for t in range(n):
                op = ops[ind[t]]
                actual = local_multiply_dense(op, vec, targets=t,
                                              backend=backend)
                U = gates.expand_operator(op, n, targets=t, dims=vec.dims[0])

                # test for vector multiplication
                expected = U * vec
                np.testing.assert_array_almost_equal(actual.full(),
                                                     expected.full())

                # now test left multiplication on matrix
                actual = local_multiply_dense(op, mat, targets=t,
                                              backend=backend)
                expected = U * mat
                np.testing.assert_array_almost_equal(actual.full(),
                                                     expected.full())

                # now test right multiplication on matrix
                actual = local_multiply_dense(op, mat, targets=t, right=True,
                                              backend=backend)
                expected = mat * U
                np.testing.assert_array_almost_equal(actual.full(),
                                                     expected.full())

    @pytest.mark.parametrize('backend', ('einsum', 'vectorize'))
    def test_superop_mult(self, backend):
        # test computing, in one go, L * M * R, as (L ⊗ R^T) |M>>
        # where L, R act on same qubit of a multi-qubit system
        # Here M, L, R are chosen arbitrarily
        n = 4
        d = 2 ** n
        M = np.array(range(d ** 2), dtype='complex').reshape(d, d)
        M += 1j * np.array(range(d ** 2, 2 * d ** 2)).reshape(d, d)
        M = qutip.Qobj(M, dims=[[2] * n, [2] * n])
        L = np.array(range(4)) + 1j * np.array(range(4, 8))
        L = L.reshape(2, 2)
        R = np.array(range(-4, 0)) + 1j * np.array(range(-10, -6))
        R = R.reshape(2, 2)

        for t in range(n):
            expected = local_multiply_dense(L, M, targets=t, backend=backend)
            expected = local_multiply_dense(R, expected, targets=t, right=True,
                                            backend=backend)

            LR = np.kron(L, R.T)
            actual = local_multiply_dense(LR, M, targets=[t, t + n],
                                          backend=backend)

            np.testing.assert_array_almost_equal(actual.full(), expected.full())

    def test_data_types(self):
        psi_sp = qutip.ghz_state(1)
        psi_dense = psi_sp.to(data.Dense)

        op = (-0.25 * np.pi * 1j * qutip.sigmax()).expm()

        out_sp = local_multiply_dense(op, psi_sp, targets=0)
        out_dense = local_multiply_dense(op, psi_dense, targets=0)

        np.testing.assert_array_almost_equal(out_sp.full(), out_dense.full())
        np.testing.assert_array_almost_equal(out_sp.full(),
                                             (op * psi_sp).full())

        if not isinstance(out_sp.data, data.CSR):
            raise TypeError()

        if not isinstance(out_dense.data, data.Dense):
            raise TypeError()