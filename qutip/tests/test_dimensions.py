# -*- coding: utf-8 -*-
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

from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.dimensions import (
    type_from_dims,
    flatten, enumerate_flat, deep_remove, unflatten,
    dims_idxs_to_tensor_idxs, dims_to_tensor_shape,
    collapse_dims_oper, collapse_dims_super
)
from qutip.qobj import Qobj


def test_flatten():
    l = [[[0], 1], 2]
    assert_equal(flatten(l), [0, 1, 2])


def test_enumerate_flat():
    l = [[[10], [20, 30]], 40]
    labels = enumerate_flat(l)
    assert_equal(labels, [[[0], [1, 2]], 3])


def test_deep_remove():
    l = [[[0], 1], 2]
    l = deep_remove(l, 1)
    assert_equal(l, [[[0]], 2])

    # Harder case...
    l = [[[[0, 1, 2]], [3, 4], [5], [6, 7]]]
    l = deep_remove(l, 0, 5)
    assert l == [[[[1, 2]], [3, 4], [], [6, 7]]]


def test_unflatten():
    l = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]
    labels = enumerate_flat(l)
    assert unflatten(flatten(l), labels) == l


def test_dims_idxs_to_tensor_idxs():
    # Dims for a superoperator acting on linear operators on C^2 x C^3.
    dims = [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]
    # Should swap the input and output subspaces of the left and right dims.
    assert_equal(
        dims_idxs_to_tensor_idxs(dims, list(range(len(flatten(dims))))),
        [2, 3, 0, 1, 6, 7, 4, 5]
    )
    # TODO: more cases (oper-ket, oper-bra, and preserves
    #       non-vectorized qobjs).


def test_dims_to_tensor_shape():
    # Dims for a superoperator:
    #     L(L(C⁰ × C¹, C² × C³), L(C³ × C⁴, C⁵ × C⁶)),
    # where L(X, Y) is a linear operator from X to Y (dims [Y, X]).
    in_dims  = [[2, 3], [0, 1]]
    out_dims = [[3, 4], [5, 6]]
    dims = [out_dims, in_dims]

    # To make the expected shape, we want the left and right spaces to each
    # be flipped, then the whole thing flattened.
    shape = (5, 6, 3, 4, 0, 1, 2, 3)

    assert_equal(
        dims_to_tensor_shape(dims),
        shape
    )
    # TODO: more cases (oper-ket, oper-bra, and preserves
    #       non-vectorized qobjs).


def test_type_from_dims():
    def dims_case(dims, expected_type, enforce_square=True):
        actual_type = type_from_dims(dims, enforce_square=enforce_square)

        assert_equal(
            actual_type,
            expected_type,
            "Expected {} to be type='{}', but was type='{}'.".format(
                dims, expected_type, actual_type
            )
        )

    def qobj_case(qobj):
        assert_equal(type_from_dims(qobj.dims), qobj.type)

    dims_case([[2], [2]], 'oper')
    dims_case([[2, 3], [2, 3]], 'oper')
    dims_case([[2], [3]], 'other')
    dims_case([[2], [3]], 'oper', False)

    dims_case([[2], [1]], 'ket')
    dims_case([[1], [2]], 'bra')

    dims_case([[[2, 3], [2, 3]], [1]], 'operator-ket')
    dims_case([[1], [[2, 3], [2, 3]]], 'operator-bra')

    dims_case([[[3], [3]], [[2, 3], [2, 3]]], 'other')
    dims_case([[[3], [3]], [[2, 3], [2, 3]]], 'super', False)

    dims_case([[[2], [3, 3]], [[3], [2, 3]]], 'other')

    ## Qobj CASES ##

    N = int(np.ceil(10.0 * np.random.random())) + 5

    ket_data = np.random.random((N, 1))
    ket_qobj = Qobj(ket_data)
    qobj_case(ket_qobj)

    bra_data = np.random.random((1, N))
    bra_qobj = Qobj(bra_data)
    qobj_case(bra_qobj)

    oper_data = np.random.random((N, N))
    oper_qobj = Qobj(oper_data)
    qobj_case(oper_qobj)

    N = 9
    super_data = np.random.random((N, N))
    super_qobj = Qobj(super_data, dims=[[[3]], [[3]]])
    qobj_case(super_qobj)

def test_collapse():
    # ket-type
    # assert_equal(collapse_dims_oper([[1], [3]]), [[1], [3]], err_msg='ket-type, trivial')
    # assert_equal(collapse_dims_oper([[1], [2, 3]]), [[1], [6]], err_msg='ket-type, bipartite')
    # # bra-type
    # assert_equal(collapse_dims_oper([[2], [1]]), [[2], [1]], err_msg='bra-type, trivial')
    # assert_equal(collapse_dims_oper([[2, 3], [1]]), [[6], [1]], err_msg='bra-type, bipartite')
    # # oper-type
    # assert_equal(collapse_dims_oper([[2, 3], [2, 3]]), [[6], [6]], err_msg='oper-type, trivial')
    # assert_equal(collapse_dims_oper([[2, 3], [4, 5]]), [[6], [20]], err_msg='oper-type, bipartite')
    # # operator-ket-type
    # assert_equal(collapse_dims_super([[[1]], [[2, 3], [2, 3]]]), [[[1]], [[6], [6]]])
    # operator-bra-type
    assert_equal(collapse_dims_super([[[2, 3], [2, 3]], [[1]]]), [[[6], [6]], [[1]]])
    # super-type
    assert_equal(collapse_dims_super([[[2, 3], [2, 3]], [[2, 3], [2, 3]]]), [[[6], [6]], [[6], [6]]])
