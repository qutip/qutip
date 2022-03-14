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
from qutip.core.dimensions import (
    flatten, unflatten, enumerate_flat, deep_remove, deep_map,
    dims_idxs_to_tensor_idxs, dims_to_tensor_shape, dims_to_tensor_perm,
    collapse_dims_super, collapse_dims_oper, Dimensions
)


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
    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[2], [2]], 'oper'),
        pytest.param([[2, 3], [2, 3]], 'oper'),
        pytest.param([[2], [3]], 'oper'),
        pytest.param([[2], [1]], 'ket'),
        pytest.param([[1], [2]], 'bra'),
        pytest.param([[[2, 3], [2, 3]], [1]], 'operator-ket'),
        pytest.param([[1], [[2, 3], [2, 3]]], 'operator-bra'),
        pytest.param([[[3], [3]], [[2, 3], [2, 3]]], 'super'),
    ])
    def test_Dimensions_type(self, base, expected):
        assert Dimensions(base).type == expected


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
