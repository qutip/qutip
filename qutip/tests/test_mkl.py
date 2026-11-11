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
    from qutip._mkl.spsolve import mkl_splu, mkl_spsolve


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
    """Return a complex RHS in dense or sparse vector and matrix formats."""
    is_sparse, columns = request.param

    b = np.array(
        [
            [3 + 1j, 0, 1, 0],
            [0, 2, 0, 0],
            [1, 0, 2 - 2j, 0],
        ],
        dtype=np.complex128,
    )
    b = b[:, 0] if columns is None else b[:, :columns]
    return scipy.sparse.csr_array(b) if is_sparse else b


@pytest.fixture
def nonsymmetric_matrix_random(random_generator):
    """Return sparse randomly generated nonsymmetric matrix, complex-valued."""
    hermitian = False
    posdef = False
    N = 10

    A = random_generator.standard_normal((N, N))

    A = A + 1j * random_generator.standard_normal((N, N))

    return scipy.sparse.csr_array(A), hermitian, posdef, N


@pytest.fixture(
    params=[
        pytest.param(True, id="positive-definite"),
        pytest.param(False, id="indefinite"),
    ]
)
def symmetric_matrix(request):
    """Return a sparse Hermitian CSR matrix and its solver flags."""
    return _symmetric_matrix(request.param)


def _symmetric_matrix(posdef):
    hermitian = True
    diagonal = [4.0, 4.0, 3.0] if posdef else [2.0, -2.0, 3.0]
    upper = [1.0 + 1.0j, 1.0j]
    A = (
        np.diag(np.array(diagonal, dtype=np.complex128))
        + np.diag(upper, k=1)
        + np.diag(np.conjugate(upper), k=-1)
    )
    eigenvalues = np.linalg.eigvalsh(A)
    if posdef:
        assert eigenvalues.min() > 0
    else:
        assert eigenvalues.min() < 0 < eigenvalues.max()
    return scipy.sparse.csr_array(A), hermitian, posdef


@pytest.fixture
def nonsymmetric_matrix():
    """Return a nonsymmetric CSR matrix and its solver flags."""
    return _nonsymmetric_matrix()


def _nonsymmetric_matrix():
    hermitian = False
    posdef = False

    A = np.array(
        [
            [1, 0, 2],
            [0, 0, 3],
            [-4, 5, 6],
        ],
        dtype=np.complex128,
    )
    assert not np.allclose(A, A.conj().T)
    return scipy.sparse.csr_array(A), hermitian, posdef


@pytest.fixture(
    params=[
        pytest.param((hermitian, posdef), id=kind)
        for hermitian, posdef, kind in [
            (False, False, "nonsymmetric"),
            (True, True, "symmetric-posdef"),
            (True, False, "symmetric-indef"),
        ]
    ]
)
def solver_matrix(request):
    hermitian, posdef = request.param
    if hermitian:
        return _symmetric_matrix(posdef)
    return _nonsymmetric_matrix()


@pytest.mark.parametrize("use_perm", [False, True], ids=["default", "permutation"])
def test_spsolve(solver_matrix, rhs, random_generator, use_perm):
    """Solve with different matrices and RHS formats, with and without a permutation."""
    A, hermitian, posdef = solver_matrix
    b = rhs
    b_dense = b.toarray() if scipy.sparse.issparse(b) else b
    options = {}
    if use_perm:
        options["perm"] = random_generator.permutation(A.shape[0])
    x = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef, **options)
    assert x.shape == b.shape
    actual = x.toarray() if scipy.sparse.issparse(x) else x
    np.testing.assert_allclose(A @ actual, b_dense, rtol=1e-10, atol=1e-12)

@pytest.mark.parametrize(
    "dia_type",
    [scipy.sparse.dia_matrix, scipy.sparse.dia_array],
    ids=["dia-matrix", "dia-array"],
)
def test_spsolve_dia_input(solver_matrix, dia_type):
    """Check that he wrapper accepts Dia input and converts it for MKL."""
    A, hermitian, posdef = solver_matrix
    A = dia_type(A)
    expected = np.array([1, -2, 3], dtype=A.dtype)
    b = A @ expected

    actual = mkl_spsolve(
        A, b, hermitian=hermitian, posdef=posdef
    )

    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-12
    )

@pytest.mark.parametrize(
    "csc_type",
    [scipy.sparse.csc_matrix, scipy.sparse.csc_array],
    ids=["csc-matrix", "csc-array"],
)
def test_spsolve_csc_input(solver_matrix, csc_type):
    """Check that the solver wrapper accepts CSC inputs."""
    A, hermitian, posdef = solver_matrix
    A = csc_type(A)
    expected = np.array([1, -2, 3], dtype=A.dtype)
    b = A @ expected

    actual = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("scaling_vectors", [False, True])
@pytest.mark.parametrize("weighted_matching", [False, True])
def test_solver_options(solver_matrix, scaling_vectors, weighted_matching):
    """Check the solver settings for scaling and weighted matching."""
    A, hermitian, posdef = solver_matrix
    # Obtain RHS in a simple way; using `rhs` fixture would unflate number of tests unnecessarily
    expected = np.array([1, 2, 3], dtype=A.dtype)

    _, info = mkl_spsolve(
        A,
        A @ expected,
        hermitian=hermitian,
        posdef=posdef,
        scaling_vectors=scaling_vectors,
        weighted_matching=weighted_matching,
        return_info=True,
    )

    if not hermitian:
        assert info["iparm"][10] == int(scaling_vectors)
        assert info["iparm"][12] == int(weighted_matching)


@pytest.mark.parametrize("max_iter_refine", [0, 1, 10])
def test_max_iter_refine(solver_matrix, max_iter_refine):
    """Ensures maximum number of iterative refinement steps were
    set in the solver as expected."""
    A, _, _ = solver_matrix
    expected = np.array([1, 2, 3], dtype=A.dtype)
    b = A @ expected

    actual, info = mkl_spsolve(
        scipy.sparse.csr_array(A),
        b,
        max_iter_refine=max_iter_refine,
        return_info=True,
    )

    assert info["iparm"][7] == max_iter_refine
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_matrix_type_correct(solver_matrix):
    """Check the PARDISO matrix type inferred from the solver arguments."""
    A, hermitian, posdef = solver_matrix
    b = A @ np.ones(A.shape[0], dtype=A.dtype)
    _, info = mkl_spsolve(
        A, b, hermitian=hermitian, posdef=posdef, return_info=True
    )
    if hermitian:
        expected = 4
        if not posdef:
            expected = -expected
    else:
        expected = 13
    assert info["MatrixType"] == expected



def test_spsolve_nonsymmetric_random_input(
    nonsymmetric_matrix_random, random_generator
):
    """Tests the solver on non-symmetric matrices (complex-valued),
    left- and right-hand sides are randomized."""
    A, hermitian, posdef, N = nonsymmetric_matrix_random
    x = random_generator.standard_normal(N)
    b = A @ x
    actual = mkl_spsolve(A, b, hermitian=hermitian, posdef=posdef)
    np.testing.assert_allclose(A @ actual, b, rtol=1e-10, atol=1e-12)

def test_spsolve_posdef_matrix_with_default_posdef(rhs):
    """A positive-definite matrix can be solved with the default posdef flag."""
    A, hermitian, _ = _symmetric_matrix(True)
    b = rhs
    b_dense = b.toarray() if scipy.sparse.issparse(b) else b
    x, info = mkl_spsolve(A, b, hermitian=hermitian, return_info=True)
    assert x.shape == b.shape
    actual = x.toarray() if scipy.sparse.issparse(x) else x
    assert info["MatrixType"] == -4
    np.testing.assert_allclose(A @ actual, b_dense, rtol=1e-10, atol=1e-12)


def test_spsolve_posdef_ignored_for_nonhermitian_input(
    nonsymmetric_matrix, rhs
):
    """Ensures that the solution of the sparse solver is still correct despite
    posdef solver argument having a wrong value.
    Checks that the matrix type is determined
    correctly nevertheless."""
    A, hermitian, posdef = nonsymmetric_matrix
    # Choose RHS of a value type matching to A
    b = rhs
    # RHS must be dense for SciPy's solver
    b_dense = b.toarray() if scipy.sparse.issparse(b) else b

    # Solve linear system
    x, info = mkl_spsolve(
        A,
        b,
        hermitian=hermitian,
        posdef=True,  # Deliberately inconsistent with the input
        return_info=True,
    )
    # For comparison with SciPy output,
    # convert to dense in case the sparse solver returned sparse solution
    actual = x.toarray() if scipy.sparse.issparse(x) else x
    assert not posdef
    # Ensure the solver inferred the matrix type correctly based on provided solver args
    assert info["MatrixType"] == 13
    # Ensure the linear system holds
    np.testing.assert_allclose(
        A @ actual, b_dense, rtol=1e-10, atol=1e-12
    )


def test_spsolve_hermitian_random_input(random_generator):
    """Tests the solver on symmetric matrices (complex-valued),
    left- and right-hand sides are randomized."""
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
    actual = mkl_spsolve(A.data.as_scipy(), b, hermitian=True, posdef=True)
    np.testing.assert_allclose(A.data.as_scipy() @ actual, b, rtol=1e-10, atol=1e-12)


def test_splu_nonsymmetric_repeated_vector_rhs(nonsymmetric_matrix):
    """Tests reusing a factorisation of a non-symmetric matrix
    for repeated solves of vector RHS."""
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
        x = lu.solve(N[:, k])
        assert x.dtype == A.dtype
        test_X[:, k] = x
    expected_X = scipy.linalg.solve(A.toarray(), N)
    np.testing.assert_allclose(test_X, expected_X)


def test_splu_faulty_coeff_matrix_rejection():
    """Tests input validation w.r.t. coefficient matrix."""
    # Dense matrix
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
    with pytest.raises(TypeError, match="sparse"):
        mkl_splu(A)
