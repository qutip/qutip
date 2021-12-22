# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project
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

import itertools

import numpy as np
import pytest

import qutip
from qutip.core import data as _data


def expected(qobj, sel):
    if qobj.isbra or qobj.isket:
        qobj = qobj.proj()
    sel = sorted(sel)
    dims = [[x for i, x in enumerate(qobj.dims[0]) if i in sel]]*2
    new_shape = (np.prod(dims[0]),) * 2
    out = qobj.full()
    before, after = 1, qobj.shape[0]
    for i, dim in enumerate(qobj.dims[0]):
        after //= dim
        if i in sel:
            before = before * dim
            continue
        tmp_dims = (before, dim, after) * 2
        out = np.einsum('aibcid->abcd', out.reshape(tmp_dims))
    return qutip.Qobj(out.reshape(new_shape), dims=dims)


@pytest.fixture(params=[_data.CSR, _data.Dense], ids=['CSR', 'Dense'])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['dm', 'ket'])
def dm(request):
    return request.param


@pytest.fixture
def state(dtype, dm):
    dims = [2, 3, 4]
    state = qutip.rand_ket(np.prod(dims), dims=[dims, [1]*len(dims)])
    if dm:
        state = state.proj()
    return state.to(dtype)


def test_ptrace_noncompound_rand(dtype, dm):
    """Test `A.ptrace(0) == A` when `A` is in a non-tensored Hilbert space."""
    for _ in range(5):
        state = qutip.rand_ket(5)
        if dm:
            state = state.proj()
        state = state.to(dtype)
        assert state.ptrace(0) == (state if dm else state.proj())


@pytest.mark.parametrize('pair', list(itertools.combinations(range(3), 2)))
def test_ptrace_unsorted_selection_subset(state, pair):
    """
    Regression test for gh-1325.  ptrace should work the same independently of
    the order of the input; no transposition in done in the trace operation.
    """
    # pair is always sorted.
    state_ordered = state.ptrace(pair)
    state_reversed = state.ptrace(pair[::-1])
    assert state_ordered.dims == state_reversed.dims
    assert state_ordered == state_reversed


@pytest.mark.parametrize('permutation', list(itertools.permutations(range(3))))
def test_ptrace_unsorted_selection_all(state, permutation):
    state_ptraced = state.ptrace(permutation)
    if state.isket:
        state = state.proj()
    assert state.dims == state_ptraced.dims
    assert state == state_ptraced


@pytest.mark.parametrize(['selection', 'exception'], [
    pytest.param(4, IndexError, id='too big'),
    pytest.param(-1, IndexError, id='too small'),
    pytest.param([0, 0], ValueError, id='duplicate'),
    # 'too many' may throw either from duplication or invalid index.
    pytest.param([0, 1, 2, 3], Exception, id='too many'),
])
def test_ptrace_fails_on_invalid_input(state, selection, exception):
    with pytest.raises(exception):
        state.ptrace(selection)


def test_ptrace_rand(dtype):
    'ptrace : randomized tests'
    for _ in range(5):
        A = qutip.tensor(
            qutip.rand_ket(5), qutip.rand_ket(2), qutip.rand_ket(3),
        ).to(dtype)
        for sel in ([2, 1], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.tensor(
            qutip.rand_dm(2), qutip.thermal_dm(10, 1), qutip.rand_unitary(3),
        ).to(dtype)
        for sel in ([1, 2], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.tensor(
            qutip.rand_ket(2), qutip.rand_ket(2), qutip.rand_ket(2),
            qutip.rand_ket(2), qutip.rand_ket(2), qutip.rand_ket(2),
        ).to(dtype)
        for sel in ([3, 2], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.rand_dm(64, 0.5, dims=[[4, 4, 4], [4, 4, 4]]).to(dtype)
        for sel in ([], [0], [1], [0, 2]):
            assert A.ptrace(sel) == expected(A, sel)
