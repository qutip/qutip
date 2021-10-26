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

import pytest
import collections
import qutip
from qutip.dimensions import (
    type_from_dims, flatten, unflatten, enumerate_flat, deep_remove, deep_map,
    dims_idxs_to_tensor_idxs, dims_to_tensor_shape, dims_to_tensor_perm,
    collapse_dims_super, collapse_dims_oper)

_v = "vector"
_vo = "vectorized_oper"


@pytest.mark.parametrize(["rank", "actual_type", "scalar"], [
    pytest.param([1], _v, True, id="scalar"),
    pytest.param([1, 1], _v, True, id="tensor scalar"),
    pytest.param([[1]], _vo, True, id="nested scalar"),
    pytest.param([[1], [1]], _vo, True, id="nested tensor scalar 1"),
    pytest.param([[1, 1]], _vo, True, id="nested tensor scalar 2"),
    pytest.param([2], _v, False, id="vector"),
    pytest.param([2, 3], _v, False, id="tensor vector"),
    pytest.param([1, 2, 3], _v, False, id="tensor vector with 1d subspace 1"),
    pytest.param([2, 1, 1], _v, False, id="tensor vector with 1d subspace 2"),
    pytest.param([[2]], _vo, False, id="vectorised operator"),
    pytest.param([[2, 3]], _vo, False, id="vector tensor operator"),
    pytest.param([[1, 3]], _vo, False, id="vector tensor operator with 1d"),
])
@pytest.mark.parametrize("test_type", ["scalar", _v, _vo])
def test_rank_type_detection(rank, actual_type, scalar, test_type):
    """
    Test the rank detection tests `is_scalar`, `is_vector` and
    `is_vectorized_oper` for a range of different test cases.  These tests are
    designed to be called on individual elements of the two-element `dims`
    parameter of `Qobj`s, so they're testing the row-rank and column-rank.

    It's possible to be both a scalar and something else, but "vector" and
    "vectorized_oper" are mutually exclusive.

    These functions aren't properly specified for improper dimension setups, so
    there are no tests for those.
    """
    expected = scalar if test_type == "scalar" else (actual_type == test_type)
    function = getattr(qutip.dimensions, "is_" + test_type)
    assert function(rank) == expected


@pytest.mark.parametrize(["base", "flat"], [
    pytest.param([[[0], 1], 2], [0, 1, 2], id="standard"),
    pytest.param([1, 2, [3, [4]], [5, 6], [7, [[[[[[[8]]]]]]]]],
                 [1, 2, 3, 4, 5, 6, 7, 8], id="standard"),
    pytest.param([1, 2, 3], [1, 2, 3], id="already flat"),
    pytest.param([], [], id="empty list"),
    pytest.param([[], [], [[[], [], []]]], [], id="nested empty lists"),
])
class TestFlattenUnflatten:
    def test_flatten(self, base, flat):
        assert flatten(base) == flat

    def test_unflatten(self, base, flat):
        labels = enumerate_flat(base)
        assert unflatten(flat, labels) == base


@pytest.mark.parametrize(["base", "expected"], [
    pytest.param([1, 2, 3], [0, 1, 2], id="flat"),
    pytest.param([[1], [2], [3]], [[0], [1], [2]], id="nested"),
    pytest.param([[[1], [2, 3]], 4], [[[0], [1, 2]], 3], id="nested"),
    pytest.param([], [], id="empty"),
    pytest.param([[], [], [[[], [], []]]],
                 [[], [], [[[], [], []]]], id="nested empty lists"),
])
def test_enumerate_flat(base, expected):
    assert enumerate_flat(base) == expected


@pytest.mark.parametrize(["base", "to_remove", "expected"], [
    pytest.param([[[0], 1], 2], (1,), [[[0]], 2], id="simple"),
    pytest.param([[[[0, 1, 2]], [3, 4], [5], [6, 7]]], (0, 5),
                 [[[[1, 2]], [3, 4], [], [6, 7]]], id="harder"),
    pytest.param([1, 2, 3], (), [1, 2, 3], id="no-op"),
    pytest.param([], (), [], id="empty"),
])
def test_deep_remove(base, to_remove, expected):
    assert deep_remove(base, *to_remove) == expected


@pytest.mark.parametrize("mapping", [
    pytest.param(lambda x: x*2, id="(x -> 2x)"),
    pytest.param(lambda x: [x], id="(x -> [x])"),
])
@pytest.mark.parametrize("base", [
    pytest.param([[[0], 1], 2], id="standard"),
    pytest.param([1, 2, [3, [4]], [7, [[[[[[[8]]]]]]]]], id="standard"),
    pytest.param([1, 2, 3], id="flat"),
    pytest.param([], id="empty list"),
    pytest.param([[], [], [[[], [], []]]], id="nested empty lists"),
])
def test_deep_map(base, mapping):
    """
    Test the deep mapping.  To simplify generation of edge-cases, this tests
    against an equivalent (but slower) operation of flattening and unflattening
    the list.  We can get false negatives if the `flatten` or `unflatten`
    functions are broken, but other tests should catch those.
    """
    # This function might not need to be public, and consequently might not
    # need to be tested here.
    labels = enumerate_flat(base)
    expected = unflatten([mapping(x) for x in flatten(base)], labels)
    assert deep_map(mapping, base) == expected


_Indices = collections.namedtuple('_Indices', ['base', 'permutation', 'shape'])


@pytest.mark.parametrize("indices", [
    pytest.param(_Indices([[2], [1]], [0, 1], (2, 1)), id="ket preserved"),
    pytest.param(_Indices([[2, 3], [1, 1]], [0, 1, 2, 3], (2, 3, 1, 1)),
                 id="tensor-ket preserved"),
    pytest.param(_Indices([[1], [2]], [0, 1], (1, 2)), id="bra preserved"),
    pytest.param(_Indices([[1, 1], [2, 2]], [0, 1, 2, 3], (1, 1, 2, 2)),
                 id="tensor-bra preserved"),
    pytest.param(_Indices([[2], [3]], [0, 1], (2, 3)), id="oper preserved"),
    pytest.param(_Indices([[2, 3], [1, 0]], [0, 1, 2, 3], (2, 3, 1, 0)),
                 id="tensor-oper preserved"),
    pytest.param(_Indices([[[2, 4], [6, 8]], [[1, 3], [5, 7]]],
                          [2, 3, 0, 1, 6, 7, 4, 5],
                          (6, 8, 2, 4, 5, 7, 1, 3)),
                 id="super-oper"),
    pytest.param(_Indices([[[2, 4], [6, 8]], [1]],
                          [0, 1, 2, 3, 4], (2, 4, 6, 8, 1)),
                 id="operator-ket"),
    pytest.param(_Indices([[1], [[2, 4], [6, 8]]],
                          [0, 1, 2, 3, 4], (1, 2, 4, 6, 8)),
                 id="operator-bra"),
])
class TestSuperOperatorDimsModification:
    def test_dims_to_tensor_perm(self, indices):
        # This function might not need to be public, and consequently might not
        # need to be tested here.
        assert dims_to_tensor_perm(indices.base) == indices.permutation

    def test_dims_idxs_to_tensor_idxs(self, indices):
        test_indices = list(range(len(flatten(indices.base))))
        assert (dims_idxs_to_tensor_idxs(indices.base, test_indices)
                == indices.permutation)

    def test_dims_to_tensor_shape(self, indices):
        assert dims_to_tensor_shape(indices.base) == indices.shape


class TestTypeFromDims:
    @pytest.mark.parametrize(["base", "expected", "enforce_square"], [
        pytest.param([[2], [2]], 'oper', True),
        pytest.param([[2, 3], [2, 3]], 'oper', True),
        pytest.param([[2], [3]], 'other', True),
        pytest.param([[2], [3]], 'oper', False),
        pytest.param([[2], [1]], 'ket', True),
        pytest.param([[1], [2]], 'bra', True),
        pytest.param([[[2, 3], [2, 3]], [1]], 'operator-ket', True),
        pytest.param([[1], [[2, 3], [2, 3]]], 'operator-bra', True),
        pytest.param([[[3], [3]], [[2, 3], [2, 3]]], 'other', True),
        pytest.param([[[3], [3]], [[2, 3], [2, 3]]], 'super', False),
        pytest.param([[[2], [3, 3]], [[3], [2, 3]]], 'other', True),
    ])
    def test_type_from_dims(self, base, expected, enforce_square):
        assert type_from_dims(base, enforce_square=enforce_square) == expected

    @pytest.mark.parametrize("qobj", [
        pytest.param(qutip.rand_ket(10), id='ket'),
        pytest.param(qutip.rand_ket(10).dag(), id='bra'),
        pytest.param(qutip.rand_dm(10), id='oper'),
        pytest.param(qutip.to_super(qutip.rand_dm(10)), id='super'),
    ])
    def test_qobj_dims_match_qobj(self, qobj):
        assert type_from_dims(qobj.dims) == qobj.type


class TestCollapseDims:
    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[1], [3]], [[1], [3]], id="ket trivial"),
        pytest.param([[1, 1], [2, 3]], [[1], [6]], id="ket tensor"),
        pytest.param([[2], [1]], [[2], [1]], id="bra trivial"),
        pytest.param([[2, 3], [1, 1]], [[6], [1]], id="bra tensor"),
        pytest.param([[5], [5]], [[5], [5]], id="oper trivial"),
        pytest.param([[2, 3], [2, 3]], [[6], [6]], id="oper tensor"),
    ])
    def test_oper(self, base, expected):
        assert collapse_dims_oper(base) == expected

    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[[1]], [[2, 3], [2, 3]]],
                     [[[1]], [[6], [6]]], id="operator-ket"),
        pytest.param([[[2, 3], [2, 3]], [[1]]],
                     [[[6], [6]], [[1]]], id="operator-bra"),
        pytest.param([[[2, 3], [2, 3]], [[2, 3], [2, 3]]],
                     [[[6], [6]], [[6], [6]]], id="super"),
    ])
    def test_super(self, base, expected):
        assert collapse_dims_super(base) == expected
