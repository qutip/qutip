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

@pytest.fixture(params=[
    pytest.param(True, id="real-positive"),
    pytest.param(False, id="real-indefinite"),
])
def real_symmetric_matrix(request):
    """Return a real symmetric CSR matrix and its solver flag."""
    posdef = request.param
    diagonal = [4., 3., 2.] if posdef else [2., -2., 3.]
    upper = [1., 1.]
    A = (
        np.diag(np.array(diagonal, dtype=np.float64))
        + np.diag(upper, k=1)
        + np.diag(upper, k=-1)
    )

    eigenvalues = np.linalg.eigvalsh(A)
    if posdef:
        assert eigenvalues.min() > 0
    else:
        assert eigenvalues.min() < 0 < eigenvalues.max()

    return scipy.sparse.csr_array(A), posdef

class Test_spsolve:
    # Already tests hermitian=False path. Adense has assymetric sparsity pattern; tests general real solve
    # TODO: document which Pardiso type is tested. Do we have a way to verify which matrix type was selected by the solver?
    def test_single_rhs_vector_real(self, random_generator):
        Adense = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [0, 0, 1]])
        As = scipy.sparse.csr_matrix(Adense)
        x = random_generator.standard_normal(3)
        b = As * x
        x2 = mkl_spsolve(As, b, verbose=True)
        np.testing.assert_allclose(x, x2)
    # General complex solve on Hermitian, positive definite, with a column RHS
    # TODO: which Pardiso type?
    # TODO: actually, rand_herm returns positive semi-definite: do we care about possible zero eigenvalues for testing the solver?
    def test_single_rhs_vector_complex(self, random_generator):
        A = qutip.rand_herm(
            10, density=0.8, distribution='pos_def',
            seed=random_generator, dtype='csr'
        )
        x = qutip.rand_ket(10, seed=random_generator).full()
        b = A.full() @ x
        y = mkl_spsolve(A.data.as_scipy(), b, verbose=True)
        np.testing.assert_allclose(x, y)
    # Tests real/complex multiple-RHS, contains zero-column case
    # Note: complex dtype contains only real-valued entries
    # TODO: add shape assertions
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
    # General complex vector and column RHS shape preservation
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

    def test_real_symmetric_solver(self, real_symmetric_matrix):
        A, posdef = real_symmetric_matrix
        x = np.array([1., -2., 3.], dtype=A.dtype)
        b = A @ x
        y = mkl_spsolve(A, b, hermitian=True, posdef=posdef, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(A @ y, b, rtol=1e-10, atol=1e-12)

    def test_complex_hermitian_solver(self, complex_hermitian_matrix):
        A, posdef = complex_hermitian_matrix
        x = np.array([1. + 0.5j, -2. - 1j, 3. + 2j], dtype=A.dtype)
        b = A @ x
        y = mkl_spsolve(A, b, hermitian=True, posdef=posdef, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(A @ y, b, rtol=1e-10, atol=1e-12)
    # In contrast to existing tests for testing hermitian=False, this test also contains complex input -> adds more numerical coverage (non-Hermitian tests test real-valued matrices with complex dtype passed)
    def test_complex_nonhermitian_single_rhs(self, random_generator):
        A = scipy.sparse.csr_array(np.array([
            [2 + 1j, 0, 1 - 3j],
            [4j, 1, 0],
            [0, -2 + 1j, 3],
        ], dtype=np.complex128))
        # Ensure non-hermitian
        assert (A.toarray() != A.toarray().conj().T).any()
        x = (random_generator.standard_normal(3)
             + 1j * random_generator.standard_normal(3))
        b = A @ x
        np.testing.assert_allclose(x, mkl_spsolve(A, b, verbose=True))
    # TODO: merge this test into the other multi-RHS test that misses complex values
    @pytest.mark.parametrize("k", [None, 1, 4])
    def test_sparse_nonhermitian_multi_rhs(self, k):
        """Test RHS shapes with a well-conditioned non-Hermitian matrix."""
        A = np.array([
            [4, 1.2 + 0.3j, 0],
            [1 - 0.2j, 5, 1],
            [0, 1, 6],
        ], dtype=np.complex128)
        assert not np.allclose(A, A.conj().T)
        rhs = np.array([
            [3, 0, 1, 2],
            [0, 2, 0, 1],
            [0, 0, 1, 4],
        ], dtype=np.complex128)
        b = rhs[:, 0] if k is None else rhs[:, :k]
        expected = scipy.linalg.solve(A, b)
        A = scipy.sparse.csr_array(A)
        y = mkl_spsolve(A, b, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(y, expected, atol=1e-10)
    # Small integration test
    def test_liouvillian(self, random_generator):
        N = 6
        a = qutip.destroy(N)
        H = a.dag() * a + 0.3 * (a + a.dag())
        L = qutip.liouvillian(H, [0.2 * a, 0.05 * a.dag()])
        Ls = scipy.sparse.csr_array(L.to("csr").data.as_scipy())
        assert (Ls.toarray() != Ls.toarray().conj().T).any()
        # Add shift to L to have a unique solution
        Ls = Ls + scipy.sparse.eye_array(N**2, format="csr")
        x = (random_generator.standard_normal(N**2)
             + 1j * random_generator.standard_normal(N**2))
        b = Ls @ x
        np.testing.assert_allclose(x, mkl_spsolve(Ls, b, verbose=True), atol=1e-10)
    # TODO: merge into a shared sparse-RHS test
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


class Test_splu:
    # Idea here: reuse real/complex factorisation across vector RHS solvers;
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
