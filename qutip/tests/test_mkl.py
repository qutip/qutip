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

def _nonhermitian_sparse(n, seed):
    """Random complex, non-Hermitian sparse matrix."""
    rng = np.random.default_rng(seed)
    A = scipy.sparse.random_array(
        (n, n), density=0.3, rng=rng, dtype=np.complex128,
        data_sampler=lambda size: rng.standard_normal(size)
                                  + 1j * rng.standard_normal(size),
    )
    return scipy.sparse.csr_array(A)

class Test_spsolve_nonhermitian:
    def test_complex_nonhermitian_single_rhs(self):
        A = scipy.sparse.csr_array(np.array([
            [2 + 1j, 0, 1 - 3j],
            [4j, 1, 0],
            [0, -2 + 1j, 3],
        ], dtype=np.complex128))
        # Ensure non-hermitian
        assert (A.toarray() != A.toarray().conj().T).any()
        rng = np.random.default_rng(1234)
        x = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        b = A @ x
        np.testing.assert_allclose(x, mkl_spsolve(A, b, verbose=True))

    @pytest.mark.parametrize("k", [None, 1, 4])
    def test_random_sparse_nonhermitian_multi_rhs(self, k):
        """Test single- and multi-RHS with large non-hermitian sparse matrix.
        The case of flat-array shape is tested too."""
        A = _nonhermitian_sparse(30, seed=42)
        rng = np.random.default_rng(7)
        shape = (30,) if k is None else (30, k)
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        b = A @ x
        y = mkl_spsolve(A, b, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(x, y, atol=1e-10)

    def test_rand_unitary_nonhermitian(self):
        """Non-Hermitian, complex, perfectly conditioned: tests matrix_type 13."""
        A = qutip.rand_unitary(32, density=0.2, seed=1, dtype='csr').data.as_scipy()
        A = scipy.sparse.csr_array(A)
        rng = np.random.default_rng(11)
        x = rng.standard_normal(32) + 1j * rng.standard_normal(32)
        np.testing.assert_allclose(x, mkl_spsolve(A, A @ x, verbose=True), atol=1e-12)

    def test_liouvillian(self):
        N = 6
        a = qutip.destroy(N)
        H = a.dag() * a + 0.3 * (a + a.dag())
        L = qutip.liouvillian(H, [0.2 * a, 0.05 * a.dag()])
        Ls = scipy.sparse.csr_array(L.to("csr").data.as_scipy())
        assert (Ls.toarray() != Ls.toarray().conj().T).any()
        # L is singular by construction; shift so the system has a unique solution
        Ls = Ls + scipy.sparse.eye_array(N**2, format="csr")
        rng = np.random.default_rng(5)
        x = rng.standard_normal(N**2) + 1j * rng.standard_normal(N**2)
        b = Ls @ x
        np.testing.assert_allclose(x, mkl_spsolve(Ls, b, verbose=True), atol=1e-10)


    def test_structurally_unsymmetric(self):
        A = scipy.sparse.csr_array(np.triu(np.arange(1, 26, dtype=np.float64).reshape(5, 5)))
        x = np.arange(1, 6, dtype=np.float64)
        np.testing.assert_allclose(x, mkl_spsolve(A, A @ x, verbose=True))


    def test_sparse_rhs_nonhermitian(self):
        A = scipy.sparse.csr_array(np.array([
            [1, 2 + 1j, 0],
            [0, 3, 1j],
            [4, 0, 5],
        ], dtype=np.complex128))
        b = scipy.sparse.csr_array(np.array([[0, 1], [1, 0], [0, 2]], dtype=np.complex128))
        x = mkl_spsolve(A, b, verbose=True)
        assert scipy.sparse.issparse(x)
        np.testing.assert_allclose(x.toarray(),
                                   scipy.linalg.solve(A.toarray(), b.toarray()), atol=1e-12)

    def test_nonnormal_residual(self):
        n = 20
        A = np.eye(n) + 2.0 * np.eye(n, k=1)   # non-normal Jordan-like block
        A = scipy.sparse.csr_array(A.astype(np.complex128))
        b = np.ones(n, dtype=np.complex128)
        x = mkl_spsolve(A, b, verbose=True)
        assert np.linalg.norm(A @ x - b) <= 1e-10 * np.linalg.norm(b)

    def test_repeated_rhs_solve_nonhermitian(self):
        A = _nonhermitian_sparse(12, seed=99)
        rng = np.random.default_rng(3)
        N = rng.standard_normal((12, 3)) + 1j * rng.standard_normal((12, 3))
        lu = mkl_splu(A, verbose=True)
        X = np.zeros((12, 3), dtype=np.complex128)
        for k in range(3):
            X[:, k] = lu.solve(N[:, k])
        lu.delete()
        np.testing.assert_allclose(X, scipy.linalg.solve(A.toarray(), N), atol=1e-10)

    def test_rand_stochastic_real_unsymmetric(self):
        """Real non-Hermitian: matrix_type 11, larger size"""
        A = qutip.rand_stochastic(32, density=0.2, seed=2, dtype='csr').data.as_scipy()
        A = scipy.sparse.csr_array(A).real          # drop the all-zero imaginary part
        x = np.arange(1, 33, dtype=np.float64)
        np.testing.assert_allclose(x, mkl_spsolve(A, A @ x, verbose=True), atol=1e-10)

class Test_spsolve:
    def test_single_rhs_vector_real(self):
        Adense = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [0, 0, 1]])
        As = scipy.sparse.csr_matrix(Adense)
        rng = np.random.default_rng(seed=1234)
        x = rng.standard_normal(3)
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
