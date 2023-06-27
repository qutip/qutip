import itertools

import numpy as np
from numpy.testing import assert_
import pytest

from qutip import *
from qutip.legacy.ptrace import _ptrace as _pt


@pytest.fixture(params=[True, False], ids=['sparse', 'dense'])
def sparse(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['dm', 'ket'])
def dm(request):
    return request.param


@pytest.fixture
def state(dm):
    dims = [2, 3, 4]
    state = qutip.rand_ket(np.prod(dims), dims=[dims, [1]*len(dims)])
    if dm:
        state = state.proj()
    return state


def test_ptrace_noncompound_rand(sparse, dm):
    """Test `A.ptrace(0) == A` when `A` is in a non-tensored Hilbert space."""
    for _ in range(5):
        state = qutip.rand_ket(5)
        if dm:
            state = state.proj()
        if state.isket:
            target = state.proj()
        else:
            target = state
        assert state.ptrace(0, sparse=sparse) == target


@pytest.mark.parametrize('pair', list(itertools.combinations(range(3), 2)))
def test_ptrace_unsorted_selection_subset(state, sparse, pair):
    """
    Regression test for gh-1325.  ptrace should work the same independently of
    the order of the input; no transposition in done in the trace operation.
    """
    # pair is always sorted.
    state_ordered = state.ptrace(pair, sparse=sparse)
    state_reversed = state.ptrace(pair[::-1], sparse=sparse)
    assert state_ordered.dims == state_reversed.dims
    assert state_ordered == state_reversed


def test_ptrace_ket():
    ket_1 = qutip.rand_ket(3)
    ket_2 = qutip.rand_ket(4)
    ket = qutip.tensor(ket_1, ket_2)
    assert ket.ptrace([0, 1]) == ket.proj()
    assert ket.ptrace([0]) == ket_1.proj() * ket_2.norm()


@pytest.mark.parametrize('permutation', list(itertools.permutations(range(3))))
def test_ptrace_unsorted_selection_all(state, sparse, permutation):
    state_ptraced = state.ptrace(permutation, sparse=sparse)
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
def test_ptrace_fails_on_invalid_input(state, sparse, selection, exception):
    with pytest.raises(exception):
        state.ptrace(selection, sparse=sparse)


def test_ptrace_rand():
    'ptrace : randomized tests, sparse'
    for k in range(5):
        A = tensor(rand_ket(5), rand_ket(2), rand_ket(3))
        B = A.ptrace([1,2], True)
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], True)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], True)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = tensor(rand_dm(2), thermal_dm(10,1), rand_unitary(3))
        B = A.ptrace([1,2], True)
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], True)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], True)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = tensor(rand_ket(2),rand_ket(2),rand_ket(2),
                    rand_ket(2),rand_ket(2),rand_ket(2))
        B = A.ptrace([3,2], True)
        bdat,bd,bs = _pt(A, [3,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], True)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], True)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = rand_dm(64,0.5,dims=[[4,4,4],[4,4,4]])
        B = A.ptrace([0], True)
        bdat,bd,bs = _pt(A, [0])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([1], True)
        bdat,bd,bs = _pt(A, [1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], True)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)


def test_ptrace_rand():
    'ptrace : randomized tests, dense'
    for k in range(5):
        A = tensor(rand_ket(5), rand_ket(2), rand_ket(3))
        B = A.ptrace([1,2], False)
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], False)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], False)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = tensor(rand_dm(2), thermal_dm(10,1), rand_unitary(3))
        B = A.ptrace([1,2], False)
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], False)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], False)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = tensor(rand_ket(2),rand_ket(2),rand_ket(2),
                    rand_ket(2),rand_ket(2),rand_ket(2))
        B = A.ptrace([3,2], False)
        bdat,bd,bs = _pt(A, [3,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], False)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,1], False)
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

    for k in range(5):
        A = rand_dm(64,0.5,dims=[[4,4,4],[4,4,4]])
        B = A.ptrace([0], False)
        bdat,bd,bs = _pt(A, [0])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([1], False)
        bdat,bd,bs = _pt(A, [1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)

        B = A.ptrace([0,2], False)
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
