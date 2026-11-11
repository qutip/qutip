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
    from qutip._mkl.spsolve import (
        _mkl_matrix_type,
        _prepare_pydiso_args,
        mkl_splu,
        mkl_spsolve,
    )


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


@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("scaling_vectors", [False, True])
@pytest.mark.parametrize("weighted_matching", [False, True])
def test_solver_options(hermitian, scaling_vectors, weighted_matching):
    """Check QuTiP's mapping of scaling and matching options to pydiso."""
    options = _prepare_pydiso_args(
        max_iter_refine=10,
        hermitian=hermitian,
        scaling_vectors=scaling_vectors,
        weighted_matching=weighted_matching,
    )

    if hermitian:
        # Leave these settings to pydiso's defaults for Hermitian matrices
        assert "scaling" not in options
        assert "weighted_matching" not in options
    else:
        assert options["scaling"] == scaling_vectors
        assert options["weighted_matching"] == weighted_matching


@pytest.mark.parametrize("max_iter_refine", [0, 1, 10])
def test_max_iter_refine(max_iter_refine):
    """Check that QuTiP forwards the iterative refinement limit to pydiso."""
    options = _prepare_pydiso_args(max_iter_refine=max_iter_refine)

    assert options["max_iterative_refinement_steps"] == max_iter_refine


@pytest.mark.parametrize(
    "hermitian, posdef, expected_real, expected_complex",
    [
        (False, False, 11, 13),
        (False, True, 11, 13),
        (True, False, -2, -4),
        (True, True, 2, 4),
    ],
)
@pytest.mark.parametrize("is_complex", [False, True], ids=["real", "complex"])
def test_matrix_type_correct(
    hermitian, posdef, expected_real, expected_complex, is_complex
):
    """Check QuTiP's mapping of dtype and matrix flags to PARDISO types."""
    dtype = np.dtype(np.complex128 if is_complex else np.float64)
    expected = expected_complex if is_complex else expected_real

    assert _mkl_matrix_type(dtype, hermitian, posdef) == expected


def test_spsolve_posdef_matrix_with_default_posdef(rhs):
    """Solve a positive-definite matrix with the default posdef flag."""
    A, _, _ = _symmetric_matrix(True)
    b = rhs
    b_dense = b.toarray() if scipy.sparse.issparse(b) else b
    x = mkl_spsolve(A, b, hermitian=True)
    assert x.shape == b.shape
    actual = x.toarray() if scipy.sparse.issparse(x) else x
    np.testing.assert_allclose(A @ actual, b_dense, rtol=1e-10, atol=1e-12)


def test_spsolve_posdef_ignored_for_nonhermitian_input(
    nonsymmetric_matrix, rhs
):
    """Check that non-Hermitian solves succeed with posdef=True."""
    A, _, _ = nonsymmetric_matrix
    b_dense = rhs.toarray() if scipy.sparse.issparse(rhs) else rhs

    x = mkl_spsolve(A, rhs, hermitian=False, posdef=True)

    actual = x.toarray() if scipy.sparse.issparse(x) else x
    np.testing.assert_allclose(
        A @ actual, b_dense, rtol=1e-10, atol=1e-12
    )


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
