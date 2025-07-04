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
    new_shape = (np.prod(dims[0], dtype=int),) * 2
    if not dims[0]:
        dims = None
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
    state = qutip.rand_ket(dims)
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


@pytest.mark.parametrize('dims, sel',
                         [
                             ([5, 2, 3], [2, 1]),
                             ([5, 2, 3], [0, 2]),
                             ([5, 2, 3], [0, 1]),
                             ([2]*6, [3, 2]),
                             ([2]*6, [0, 2]),
                             ([2]*6, [0, 1]),
                         ])
def test_ptrace_rand_ket(dtype, dims, sel):
    A = qutip.rand_ket(dims)
    assert A.ptrace(sel) == expected(A, sel)


@pytest.mark.parametrize('sel', [[], [0, 1, 2], [0], [1], [1, 0], [0, 2]],
                         ids=['trace_all',
                              'trace_none',
                              'trace_one',
                              'trace_one_2',
                              'trace_multiple',
                              'trace_multiple_not_sorted',
                              ])
def test_ptrace_rand_dm(dtype, sel):
    A = qutip.rand_dm([4, 4, 4], density=0.5).to(dtype)
    assert A.ptrace(sel) == expected(A, sel)


@pytest.mark.parametrize('sel', [[], [0, 1, 2], [0], [1], [1, 0], [0, 2]],
                         ids=['trace_all',
                              'trace_none',
                              'trace_one',
                              'trace_one_2',
                              'trace_multiple',
                              'trace_multiple_not_sorted',
                              ])
def test_ptrace_operator(dtype, sel):
    A = qutip.tensor(
        qutip.rand_dm(2), qutip.thermal_dm(10, 1), qutip.rand_unitary(3),
    ).to(dtype)
    assert A.ptrace(sel) == expected(A, sel)

@pytest.mark.parametrize('dims, sel',
                         [
                             ([5, 2, 3], [2, 1]),
                             ([5, 2, 3], [0, 2]),
                             ([5, 2, 3], [0, 1]),
                             ([5, 2, 3], [0]),
                             ([2]*6, []),
                             ([2]*6, [3]),
                             ([2]*6, [0, 2]),
                             ([2]*6, [0, 1, 4]),
                             ([2]*6, [0, 1, 2, 3, 4, 5]),
                         ])
def test_ptrace_ket_specialization_matches_old_implementation(dtype, dims, sel):
    """Kets have a different implementation for ptrace. 
       Test that this specialization gives the same result 
       as the non-specialized version.
    """
    A = qutip.rand_ket(dims)
    assert A.ptrace(sel) == A.proj().ptrace(sel)