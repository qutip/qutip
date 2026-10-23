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

@pytest.fixture(
    params=[
        pytest.param((False, None), id="rhs-dense-vector"),
        pytest.param((False, 1), id="rhs-dense-column"),
        pytest.param((False, 4), id="rhs-dense-multiple"),
        pytest.param((True, 1), id="rhs-sparse-column"),
        pytest.param((True, 4), id="rhs-sparse-multiple"),
    ]
)
def rhs(request):
    """Return a right-hand side of a linear system of different types and structures:
    1. Dense, row vector (1D array), real-valued
    2. Dense, row vector (1D array), complex-valued
    3. Sparse, row vector (1D array), real-valued
    4. Sparse, row vector (1D array), complex-valued
    5. Dense, column vector (2D array), real-valued
    6. Dense, column vector (2D array), complex-valued
    7. Sparse, column vector (2D array), real-valued
    8. Sparse, column vector (2D array), complex-valued
    9. Dense, matrix (2D array), real-valued
    10. Dense, matrix (2D array), complex-valued
    11. Sparse, matrix (2D array), real-valued
    12. Sparse, matrix (2D array), complex-valued
    """
    is_sparse, columns = request.param

    def make_rhs(dtype):
        b = np.array(
            [
                [3, 0, 1, 0],
                [0, 2, 0, 0],
                [1, 0, 2, 0],
            ],
            dtype=dtype,
        )
        if np.issubdtype(dtype, np.complexfloating):
            b[0, 0] += 1j
            b[2, 2] - +2j

        b = b[:, 0] if columns is None else b[:, :columns]
        return scipy.sparse.csr_array(b) if is_sparse else b

    return make_rhs

@pytest.fixture(params=[np.float64, np.complex128], ids=["real", "complex"])
def nonsymmetric_matrix_random(request, random_generator):
    """Return sparse randomly generated nonsymmetric matrix, complex and real-valued."""
    dtype = request.param
    hermitian = False
    posdef = False
    N = 10

    A = random_generator.standard_normal((N, N))

    if np.issubdtype(dtype, np.complexfloating):
       A = A + 1j * random_generator.standard_normal((N, N))
    
    return scipy.sparse.csr_array(A), hermitian, posdef, N

@pytest.fixture(
    params=[
        pytest.param(
            (dtype, posdef),
            id=f"{name}-mtype={_mkl_matrix_type(dtype, True, posdef)}",
        )
        for dtype, name in [
            (np.float64, "real"),
            (np.complex128, "complex"),
        ]
        for posdef in [True, False]
    ]
)
def symmetric_matrix(request):
    """Return a sparse symmetric/Hermitian CSR matrix and its solver flags.
    Generated matrices correspond to the following PARDISO mtypes:
    1. mtype = 2 - real-valued symmetric positive-definite
    2. mtype = -2 - real-valued symmetric indefinite
    3. mtype = 4 - complex-valued Hermitian positive definite
    4. mtype = -4 - complex-valued Hermitian indefinite
    """
    dtype, posdef = request.param
    hermitian = True
    is_complex = np.issubdtype(dtype, np.complexfloating)
    diagonal = [4.0, 4.0, 3.0] if posdef else [2.0, -2.0, 3.0]
    upper = [1.0 + 1.0j, 1.0j] if is_complex else [1.0, 1.0]
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


@pytest.fixture(
    params=[
        pytest.param((True,), id="real-symmetric-posdef"),
        pytest.param((False,), id="real-symmetric-indef"),
    ]
)
def symmetric_real_matrix_random(request, random_generator):
    """Returns a sparse symmetric real-valued randomly generated matrices,
    positive definite and indefinite."""
    posdef = request.param
    N = 10
    hermitian = True

    A = random_generator.standard_normal((N, N))
    A = A @ A.T

    if posdef:
        # Eigenvalues have to be at least 1
        A += np.eye(N)
    else:
        eigenvalues = np.linalg.eigvalsh(A)
        shift = (eigenvalues[0] + eigenvalues[-1]) / 2
        A -= shift * np.eye(N)

    return scipy.sparse.csr_array(A), N, posdef, hermitian


@pytest.fixture(
    params=[
        pytest.param(
            dtype,
            id=f"{name}-mtype={_mkl_matrix_type(dtype, False, False)}",
        )
        for dtype, name in [
            (np.float64, "real"),
            (np.complex128, "complex"),
        ]
    ]
)
def nonsymmetric_matrix(request):
    """Return a nonsymmetric CSR matrix and its solver flags.
    Generated matrices correspond to the following PARDISO mtypes:
    1. mtype = 11 - real-valued nonsymmetric
    2. mtype = 13 - complex-valued nonsymmetric
    """
    dtype = request.param
    hermitian = False
    posdef = False

    A = np.array(
        [
            [1, 0, 2],
            [0, 0, 3],
            [-4, 5, 6],
        ],
        dtype=dtype,
    )
    assert not np.allclose(A, A.conj().T)
    return scipy.sparse.csr_array(A), hermitian, posdef


class Test_spsolve_nonsymmetric:

    def test_nonsymmetric(self, nonsymmetric_matrix, rhs):
        A, hermitian, posdef = nonsymmetric_matrix
        b = rhs(A.dtype)
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b
        # Solve linear system
        x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
        assert x.shape == b.shape
        expected = scipy.linalg.solve(A.toarray(), b_dense)
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_nonsymmetric_with_perm(self, nonsymmetric_matrix, rhs, random_generator):
        """Check that supplying a permutation produces correct solution."""
        A, hermitian, posdef = nonsymmetric_matrix
        b = rhs(A.dtype)
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b
        # Create a non-trivial permutation
        perm = random_generator.permutation(b_dense.shape[0])
        # Solve linear system
        x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef, perm=perm)
        assert x.shape == b.shape
        expected = scipy.linalg.solve(A.toarray(), b_dense)
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("scaling_vectors", [False, True])
    @pytest.mark.parametrize("weighted_matching", [False, True])
    def test_nonsymmetric_solver_options(self, nonsymmetric_matrix, scaling_vectors, weighted_matching):
        A, hermitian, posdef = nonsymmetric_matrix
        # Obtain RHS in a simple way; using `rhs` fixture would unflate number of tests unnecessarily
        expected = np.array([1, 2, 3], dtype=A.dtype)
        b = A @ expected

        actual, info = mkl_spsolve(
            scipy.sparse.csr_array(A),
            b,
            hermitian=hermitian,
            posdef=posdef,
            scaling_vectors=scaling_vectors,
            weighted_matching=weighted_matching,
            return_info=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

        if not hermitian:
            assert info["iparm"][10] == int(scaling_vectors)
            assert info["iparm"][12] == int(weighted_matching)

    def test_nonsymmetric_random_input(self, nonsymmetric_matrix_random, random_generator):
        A, hermitian, posdef, N = nonsymmetric_matrix_random
        x = random_generator.standard_normal(N)
        b = A @ x
        expected = scipy.linalg.solve(A.toarray(), b)
        actual = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


    def test_posdef_ignored_for_nonhermitian_input(self, nonsymmetric_matrix, rhs):
        """Test that the solution of the sparse solver is still correct despite
        posdef solver argument having wrong value. Ensure matrix type is determined
        correctly nevertheless. """
        A, hermitian, posdef = nonsymmetric_matrix

        b = rhs(A.dtype)
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b

        expected = scipy.linalg.solve(A.toarray(), b_dense)

        # Solve linear system
        x, info = mkl_spsolve(
            scipy.sparse.csr_array(A),
            b,
            hermitian=hermitian,
            posdef=True,  # Deliberately inconsistent with the input
            return_info=True,
        )
        # For comparison with SciPy output,
        # convert to dense in case the sparse solver returned sparse solution
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        assert not posdef
        assert info["MatrixType"] == (13 if A.dtype == np.complex128 else 11)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(A.toarray() @ actual, b_dense, rtol=1e-10, atol=1e-12)


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

    def test_symmetric_with_perm(self, symmetric_matrix, rhs, random_generator):
        """Check that supplying a permutation produces correct solution."""
        A, hermitian, posdef = symmetric_matrix
        b = rhs(A.dtype)
        b_dense = b.toarray() if scipy.sparse.issparse(b) else b
        # Create a non-trivial permutation
        perm = random_generator.permutation(b_dense.shape[0])
        # Solve linear system
        x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef, perm=perm)
        assert x.shape == b.shape
        expected = scipy.linalg.solve(A.toarray(), b_dense)
        actual = x.toarray() if scipy.sparse.issparse(x) else x
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_real_symmetric_random_input(
        self, symmetric_real_matrix_random, random_generator
    ):
        A, N, posdef, hermitian = symmetric_real_matrix_random
        x = random_generator.standard_normal(N)
        b = A @ x
        expected = scipy.linalg.solve(A.toarray(), b)
        actual = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_hermitian_random_input(self, random_generator):
        N = 10
        A = qutip.rand_herm(
            N,
            density=0.8,
            dtype="csr",
            distribution="pos_def",
            seed=random_generator,
        )
        x = qutip.rand_ket(N, seed=random_generator)
        b = A.full() @ x.full()
        expected = scipy.linalg.solve(A.full(), b)
        actual = mkl_spsolve(A.data.as_scipy(), b, hermitian=True, posdef=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


class Test_splu:
    # PARDISO mtype=11 (real) and mtype=13 (complex): reuse the general
    # factorization for repeated vector RHS solves
    def test_nonsymmetric_repeated_vector_rhs(self, nonsymmetric_matrix):
        A, hermitian, posdef = nonsymmetric_matrix
        N = np.array(
            [
                [3, 0, 1],
                [0, 2, 0],
                [0, 0, 0],
            ],
            dtype=A.dtype,
        )
        test_X = np.zeros((3, 3), dtype=A.dtype)
        lu = mkl_splu(A, hermitian=hermitian, posdef=posdef, verbose=True)
        for k in range(3):
            test_X[:, k] = lu.solve(N[:, k])
        lu.delete()
        expected_X = scipy.linalg.solve(A.toarray(), N)
        np.testing.assert_allclose(test_X, expected_X)
