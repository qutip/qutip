import pytest
import numpy as np
import scipy.sparse
import qutip
from qutip.fastsparse import fast_csr_matrix
from qutip.cy.checks import (_test_sorting, _test_coo2csr_inplace_struct,
                             _test_csr2coo_struct, _test_coo2csr_struct)
from qutip.random_objects import rand_jacobi_rotation


def _unsorted_csr(N, density=0.5):
    M = scipy.sparse.diags(np.arange(N), 0, dtype=complex, format='csr')
    nvals = N**2 * density
    while M.nnz < 0.95*nvals:
        M = rand_jacobi_rotation(M)
    M = M.tocsr()
    return fast_csr_matrix((M.data, M.indices, M.indptr), shape=M.shape)


def sparse_arrays_equal(a, b):
    return not (a != b).data.any()


@pytest.mark.repeat(20)
def test_coo2csr_struct():
    "Cython structs : COO to CSR"
    A = qutip.rand_dm(5, 0.5).data
    assert sparse_arrays_equal(A, _test_coo2csr_struct(A.tocoo()))


@pytest.mark.repeat(20)
def test_indices_sort():
    "Cython structs : sort CSR indices inplace"
    A = _unsorted_csr(10, 0.25)
    B = A.copy()
    B.sort_indices()
    _test_sorting(A)
    assert np.all(A.data == B.data)
    assert np.all(A.indices == B.indices)


@pytest.mark.repeat(20)
def test_coo2csr_inplace_nosort():
    "Cython structs : COO to CSR inplace (no sort)"
    A = qutip.rand_dm(5, 0.5).data
    B = _test_coo2csr_inplace_struct(A.tocoo(), sorted=0)
    assert sparse_arrays_equal(A, B)


@pytest.mark.repeat(20)
def test_coo2csr_inplace_sort():
    "Cython structs : COO to CSR inplace (sorted)"
    A = qutip.rand_dm(5, 0.5).data
    B = _test_coo2csr_inplace_struct(A.tocoo(), sorted=1)
    assert sparse_arrays_equal(A, B)


@pytest.mark.repeat(20)
def test_csr2coo():
    "Cython structs : CSR to COO"
    A = qutip.rand_dm(5, 0.5).data
    B = A.tocoo()
    C = _test_csr2coo_struct(A)
    assert sparse_arrays_equal(B, C)
