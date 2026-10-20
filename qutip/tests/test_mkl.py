import pytest
import numpy as np
import scipy.linalg
import scipy.sparse
import qutip

if not qutip.settings.has_mkl:
    pytest.skip(
        "MKL extensions not found.",
        allow_module_level=True,
    )

if qutip.settings.has_mkl:
    from qutip._mkl.spsolve import mkl_splu, mkl_spsolve, _mkl_matrix_type

@pytest.fixture(params=[
    pytest.param((False, None), id="rhs-dense-vector"),
    pytest.param((False, 1), id="rhs-dense-column"),
    pytest.param((False, 4), id="rhs-dense-multiple"),
    pytest.param((True, 1), id="rhs-sparse-column"),
    pytest.param((True, 4), id="rhs-sparse-multiple"),
])
def rhs(request):
    is_sparse, columns = request.param

    def make_rhs(dtype):
        b = np.array([
                [3, 0, 1, 0],
                [0, 2, 0, 0],
                [1, 0, 2, 0],
        ], dtype=dtype)
        if np.issubdtype(type, np.complexfloating):
            b[0, 0] += 1j
            b[2, 2] -+ 2j

        b = b[:, 0] if columns is None else b[:, :columns]
        return scipy.sparse.csr_array(b) if is_sparse else b

    return make_rhs

@pytest.fixture(params=[
    pytest.param(
        (dtype, posdef),
        id=f"{name}-mtype={_mkl_matrix_type(dtype, True, posdef)}",
    )
    for dtype, name in [
        (np.float64, "real"),
        (np.complex128, "complex"),
    ]
    for posdef in [True, False]
])
def symmetric_matrix(request):
    """Return a symmetric/Hermitian CSR matrix and its solver flags."""
    dtype, posdef = request.param
    hermitian = True
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
    return scipy.sparse.csr_array(A), hermitian, posdef

@pytest.fixture(params=[
    pytest.param(
        dtype,
        id=f"{name}-mtype={_mkl_matrix_type(dtype, False, False)}",
    )
    for dtype, name in [
        (np.float64, "real"),
        (np.complex128, "complex"),
    ]
])
def nonsymmetric_matrix(request):
    """Return a nonsymmetric CSR matrix and its solver flags."""
    dtype = request.param
    hermitian = False
    posdef = False

    A = np.array([
        [1, 0, 2],
        [0, 0, 3],
        [-4, 5, 6],
    ], dtype=dtype)
    assert not np.allclose(A, A.conj().T)
    return scipy.sparse.csr_array(A), hermitian, posdef

class Test_spsolve_nonsymmetric:

    def test_nonsymmetric(self, nonsymmetric_matrix, rhs):
        A, hermitian, posdef = nonsymmetric_matrix
        b = rhs(A.dtype)
        x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
        assert x.shape == b.shape
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b
        expected = scipy.linalg.solve(A.toarray(), b_dense)
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    """
    # TODO: no case when A is dense. Add one to test pydiso's conversion to sparse
     
    # PARDISO mtype=11: real nonsymmetric
    # A - sparse, real-valued
    # b - sparse/dense? , real-valued
    def test_real_nonsymmetric_vector_rhs(self, random_generator):
        Adense = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [0, 0, 1]])
        As = scipy.sparse.csr_matrix(Adense)
        x = random_generator.standard_normal(3)
        b = As * x
        x2 = mkl_spsolve(As, b, hermitian=False, posdef=False, verbose=False)
        np.testing.assert_allclose(x, x2)
    
    # PARDISO mtype=11 (real) and mtype=13 (complex), with multiple RHS
    # A - sparse, real/complex
    # b - dense, real/complex
    def test_nonsymmetric_multiple_rhs(self, nonsymmetric_matrix):
        A, hermitian, posdef = nonsymmetric_matrix
        N = np.array([
            [3, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ], dtype=A.dtype)
        sX = mkl_spsolve(A, N, hermitian=hermitian, posdef=posdef, verbose=False)
        X = scipy.linalg.solve(A.toarray(), N)
        np.testing.assert_allclose(X, sX)
    
    # PARDISO mtype=11
    # A - real-valued, sparse
    # RHS: sparse, real valued matrix
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
        x = mkl_spsolve(A, b, hermitian=False, posdef=False, verbose=False)
        ans = np.array([[-0.66666667, 1],
                        [0.33333333, 0],
                        [0, 0]])
        np.testing.assert_allclose(x.toarray(), ans)
    
    # PARDISO mtype=13: complex nonsymmetric, with sparse multiple RHS
    # A - complex, sparse
    # RHS - complex matrix, sparse
    def test_complex_nonhermitian_sparse_rhs(self):
        A = scipy.sparse.csr_array(np.array([
            [1, 2 + 1j, 0],
            [0, 3, 1j],
            [4, 0, 5],
        ], dtype=np.complex128))
        b = scipy.sparse.csr_array(np.array([
            [0, 1], [1, 0], [0, 2],
        ], dtype=np.complex128))
        x = mkl_spsolve(A, b, hermitian=False, posdef=False, verbose=False)
        assert scipy.sparse.issparse(x)
        assert x.shape == b.shape
        np.testing.assert_allclose(
            x.toarray(), scipy.linalg.solve(A.toarray(), b.toarray()),
            atol=1e-12,
        )
    
    # PARDISO mtype=13 with complex matrix and RHS entries
    # A - complex and dense, sparse
    # RHS is parameterised to be column-vector, row-vector, and matrix
    # RHS is dense, complex-valued
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
            hermitian=False, posdef=False, verbose=False,
        )
        assert y.shape == b.shape
        np.testing.assert_allclose(y, expected, atol=1e-10)
"""
class Test_spsolve_symmetric:
    
    def test_symmetric(self, symmetric_matrix, rhs):
        A, hermitian, posdef = symmetric_matrix
        b = rhs(A.dtype)
        x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
        assert x.shape == b.shape
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b
        expected = scipy.linalg.solve(A.toarray(), b_dense)
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

"""
    # PARDISO mtype=4: complex Hermitian positive-definite
    # A - complex-valued, sparse
    # RHS - dense, complex valued, column vector
    def test_complex_hermitian_posdef_column_rhs(self, random_generator):
        A = qutip.rand_herm(
            10, density=0.8, distribution='pos_def',
            seed=random_generator, dtype='csr'
        )
        x = qutip.rand_ket(10, seed=random_generator).full()
        b = A.full() @ x
        y = mkl_spsolve(
            A.data.as_scipy(), b, hermitian=True, posdef=True, verbose=False
        )
        np.testing.assert_allclose(x, y)

    # PARDISO mtype=2/-2 (real symmetric positive definite/indefinite) and
    # mtype=4/-4 (complex Hermitian positive definite/indefinite)
    def test_hermitian_vector_rhs(self, symmetric_matrix):
        A, hermitian, posdef = symmetric_matrix
        x = np.array([1., -2., 3.], dtype=A.dtype)
        if np.issubdtype(A.dtype, np.complexfloating):
            x += np.array([0.5j, -1j, 2j])
        b = A @ x
        y = mkl_spsolve(A, b, hermitian=True, posdef=posdef, verbose=True)
        assert y.shape == b.shape
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12)

class Test_splu:
    # PARDISO mtype=11 (real) and mtype=13 (complex): reuse the general
    # factorization for repeated vector RHS solves
    def test_nonsymmetric_repeated_vector_rhs(self, nonsymmetric_matrix):
        A, hermitian, posdef = nonsymmetric_matrix
        N = np.array([
            [3, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ], dtype=A.dtype)
        test_X = np.zeros((3, 3), dtype=A.dtype)
        lu = mkl_splu(A, hermitian=hermitian, posdef=posdef, verbose=False)
        for k in range(3):
            test_X[:, k] = lu.solve(N[:, k])
        lu.delete()
        expected_X = scipy.linalg.solve(A.toarray(), N)
        np.testing.assert_allclose(test_X, expected_X)
"""
