import pytest
import numpy as np
import scipy.linalg
import scipy.sparse

import qutip
if qutip.settings.has_mkl:
    from qutip._mkl.spsolve import mkl_splu, mkl_spsolve

pytestmark = [
    pytest.mark.skipif(not qutip.settings.has_mkl,
                       reason='MKL extensions not found.'),
]


class Test_spsolve:
    def test_single_rhs_vector_real(self):
        Adense = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [0, 0, 1]])
        As = scipy.sparse.csr_matrix(Adense)
        np.random.seed(1234)
        x = np.random.randn(3)
        b = As * x
        x2 = mkl_spsolve(As, b, verbose=True)
        np.testing.assert_allclose(x, x2)

    def test_single_rhs_vector_complex(self):
        A = qutip.rand_herm(10, density=0.8, dtype='csr')
        x = qutip.rand_ket(10).full()
        b = A.full() @ x
        y = mkl_spsolve(A.data.as_scipy(), b, verbose=True)
        np.testing.assert_allclose(x, y)

    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_multi_rhs_vector(self, dtype):
        M = np.array([
            [1, 0, 2],
            [0, 0, 3],
            [-4, 5, 6],
        ], dtype=dtype)
        sM = scipy.sparse.csr_matrix(M)
        N = np.array([
            [3, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ], dtype=dtype)
        sX = mkl_spsolve(sM, N, verbose=True)
        X = scipy.linalg.solve(M, N)
        np.testing.assert_allclose(X, sX)

    def test_rhs_shape_is_maintained(self):
        A = scipy.sparse.csr_matrix(np.array([
            [1, 0, 2],
            [0, 0, 3],
            [-4, 5, 6],
        ], dtype=np.complex128))
        b = np.array([0, 2, 0], dtype=np.complex128)
        out = mkl_spsolve(A, b, verbose=True)
        assert b.shape == out.shape

        b = np.array([0, 2, 0], dtype=np.complex128).reshape((3, 1))
        out = mkl_spsolve(A, b, verbose=True)
        assert b.shape == out.shape

    def test_sparse_rhs(self):
        A = scipy.sparse.csr_matrix([
            [1, 2, 0],
            [0, 3, 0],
            [0, 0, 5],
        ])
        b = scipy.sparse.csr_matrix([
            [0, 1],
            [1, 0],
            [0, 0],
        ])
        x = mkl_spsolve(A, b, verbose=True)
        ans = np.array([[-0.66666667, 1],
                        [0.33333333, 0],
                        [0, 0]])
        np.testing.assert_allclose(x.toarray(), ans)

    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_symmetric_solver(self, dtype):
        A = qutip.rand_herm(10, distribution="eigen",
                            eigenvalues=np.arange(1, 11),
                            dtype='csr').data.as_scipy()
        if dtype == np.float64:
            A = A.real
        x = np.ones(10, dtype=dtype)
        b = A.dot(x)
        y = mkl_spsolve(A, b, hermitian=1, verbose=True)
        np.testing.assert_allclose(x, y)


class Test_splu:
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_repeated_rhs_solve(self, dtype):
        M = np.array([
            [1, 0, 2],
            [0, 0, 3],
            [-4, 5, 6],
        ], dtype=dtype)
        sM = scipy.sparse.csr_matrix(M)
        N = np.array([
            [3, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ], dtype=dtype)
        test_X = np.zeros((3, 3), dtype=dtype)
        lu = mkl_splu(sM, verbose=True)
        for k in range(3):
            test_X[:, k] = lu.solve(N[:, k])
        lu.delete()
        expected_X = scipy.linalg.solve(M, N)
        np.testing.assert_allclose(test_X, expected_X)
