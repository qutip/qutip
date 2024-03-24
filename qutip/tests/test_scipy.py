import numpy as np
import scipy
import pytest

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    import scipy.sparse
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.double)
    out = scipy.sparse.linalg.eigs(matrix, 4)

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    import scipy.sparse
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.complex128)
    out = scipy.sparse.linalg.eigs(matrix, 4)

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_svd(N):
    import scipy.sparse
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.complex128)
    out = scipy.sparse.linalg.svds(matrix, 4)
