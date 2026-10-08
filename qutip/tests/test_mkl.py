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
    pytest.param((np.float64, True), id="real-posdef"),
    pytest.param((np.float64, False), id="real-indefinite"),
    pytest.param((np.complex128, True), id="complex-posdef"),
    pytest.param((np.complex128, False), id="complex-indefinite"),
])
def hermitian_matrix(request):
    """Return a symmetric/Hermitian CSR matrix and its solver flag."""
    dtype, posdef = request.param
    is_complex = np.issubdtype(dtype, np.complexfloating)
    diagonal = [4., 4., 3.] if posdef else [2., -2., 3.]
    upper = [1. + 1.j, 1.j] if is_complex else [1., 1.]
    A = (
        np.diag(np.array(diagonal, dtype=dtype))
        + np.diag(upper, k=1)
        + np.diag(np.conjugate(upper), k=-1)
    )
    eigenvalues = np.linalg.eigvalsh(A)
    if posdef:
        assert eigenvalues.min() > 0
    else:
        assert eigenvalues.min() < 0 < eigenvalues.max()
    return scipy.sparse.csr_array(A), posdef


class Test_spsolve:
    # PARDISO mtype=11: real nonsymmetric
    def test_real_nonsymmetric_vector_rhs(self, random_generator):
        Adense = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [0, 0, 1]])
        As = scipy.sparse.csr_matrix(Adense)
        x = random_generator.standard_normal(3)
        b = As * x
        x2 = mkl_spsolve(As, b, hermitian=False, posdef=False, verbose=True)
        np.testing.assert_allclose(x, x2)

    # PARDISO mtype=4: complex Hermitian positive-definite
    def test_complex_hermitian_posdef_column_rhs(self, random_generator):
        A = qutip.rand_herm(
            10, density=0.8, distribution='pos_def',
            seed=random_generator, dtype='csr'
        )
        x = qutip.rand_ket(10, seed=random_generator).full()
        b = A.full() @ x
        y = mkl_spsolve(
            A.data.as_scipy(), b, hermitian=True, posdef=True, verbose=True
        )
        np.testing.assert_allclose(x, y)

    # PARDISO mtype=11 (real) and mtype=13 (complex), with multiple RHS
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_nonsymmetric_multiple_rhs(self, dtype):
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
        sX = mkl_spsolve(sM, N, hermitian=False, posdef=False, verbose=True)
        X = scipy.linalg.solve(M, N)
        np.testing.assert_allclose(X, sX)

    # PARDISO mtype=11: integer inputs are converted to floats
    def test_real_nonsymmetric_integer_sparse_rhs(self):
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
        x = mkl_spsolve(A, b, hermitian=False, posdef=False, verbose=True)
        ans = np.array([[-0.66666667, 1],
                        [0.33333333, 0],
                        [0, 0]])
        np.testing.assert_allclose(x.toarray(), ans)

    # PARDISO mtype=2/-2 (real symmetric positive definite/indefinite) and
    # mtype=4/-4 (complex Hermitian positive definite/indefinite)
    def test_hermitian_vector_rhs(self, hermitian_matrix):
        A, posdef = hermitian_matrix
        x = np.array([1., -2., 3.], dtype=A.dtype)
        if np.issubdtype(A.dtype, np.complexfloating):
            x += np.array([0.5j, -1j, 2j])
        b = A @ x
        y = mkl_spsolve(A, b, hermitian=True, posdef=posdef, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12)

    # PARDISO mtype=13 with complex matrix and RHS entries
    @pytest.mark.parametrize("k", [
        pytest.param(None, id="vector"),
        pytest.param(1, id="column"),
        pytest.param(4, id="multiple"),
    ])
    def test_complex_nonhermitian_rhs(self, k):
        A = np.array([
            [4 + 1j, 1.2 + 0.3j, 0],
            [1 - 0.2j, 5, 1],
            [0, 1, 6],
        ], dtype=np.complex128)
        assert not np.allclose(A, A.conj().T)
        rhs = np.array([
            [3 + 1j, 0, 1, 2 - 1j],
            [0, 2 - 1j, 0, 1],
            [1j, 0, 1 + 2j, 4],
        ], dtype=np.complex128)
        b = rhs[:, 0] if k is None else rhs[:, :k]
        expected = scipy.linalg.solve(A, b)
        y = mkl_spsolve(
            scipy.sparse.csr_array(A), b,
            hermitian=False, posdef=False, verbose=True,
        )
        assert y.shape == b.shape
        np.testing.assert_allclose(y, expected, atol=1e-10)

    # PARDISO mtype=13: complex nonsymmetric, with sparse multiple RHS
    def test_complex_nonhermitian_sparse_rhs(self):
        A = scipy.sparse.csr_array(np.array([
            [1, 2 + 1j, 0],
            [0, 3, 1j],
            [4, 0, 5],
        ], dtype=np.complex128))
        b = scipy.sparse.csr_array(np.array([
            [0, 1], [1, 0], [0, 2],
        ], dtype=np.complex128))
        x = mkl_spsolve(A, b, hermitian=False, posdef=False, verbose=True)
        assert scipy.sparse.issparse(x)
        assert x.shape == b.shape
        np.testing.assert_allclose(
            x.toarray(), scipy.linalg.solve(A.toarray(), b.toarray()),
            atol=1e-12,
        )


class Test_splu:
    # PARDISO mtype=11 (real) and mtype=13 (complex): reuse the general
    # factorization for repeated vector RHS solves
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_nonsymmetric_repeated_vector_rhs(self, dtype):
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
        lu = mkl_splu(sM, hermitian=False, posdef=False, verbose=True)
        for k in range(3):
            test_X[:, k] = lu.solve(N[:, k])
        lu.delete()
        expected_X = scipy.linalg.solve(M, N)
        np.testing.assert_allclose(test_X, expected_X)
