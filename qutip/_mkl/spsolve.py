import numpy as np
import scipy.sparse as sp
import time

from pydiso.mkl_solver import MKLPardisoSolver


# dev note: after migration to pydiso (#...), using user-supplied permutations is not possible
# TODO: add proper explanation of overrides w.r.t. Intel's pardiso iparm Parameter reference
# TODO: decide what is user facing and what is not. So far, all args here are user-facing
# TODO: maybe for power users we want to provide access to the full MKLPardisoSolver interface (i.e., more overrides)
# TODO: match descriptions with Intel's docs
def _iparm_overrides(
    hermitian: bool,
    max_iter_refine: int,
    scaling_vectors: bool,
    weighted_matching: bool,
    has_perm: bool,
):
    """
    Build PARDISO parameter overrides as expected by QuTiP.

    Parameters

    ----------
    hermitian : bool
        If passed matrix is Hermitian. Used for inferring the matrix type to be passed to
        the solver's wrapper (pydiso).
    max_iter_refine : int
        Possible values are:
            0: solver automatically performs two steps of iterative refinement;
            >0: maximum number of iterative refinement steps that the solver performs.
            <0: maximum number of iterative refinement steps with a negative sign.
                Supported only for sequential and OpenMP threading.
    scaling_vectors : bool
        If PARDISO has to use scaling vectors.
        PARDISO behaviour: By default is False for symmetric indefinite matrices and True for nonsymmetric matrices. The scaling method is applied only to nonsymmetric matrices (mtype = 11 or mtype = 13). Requires symmetric weighted matching if scaling has to be used for symmetric indefinite matrices.

    weighted_matching : bool
    has_perm : bool
        If True, iparm[4] = 1. Otherwise, iparm[4] will keep its default value of 2.

    Returns
    -------
    dict[int, int]
        Zero-based ``iparm`` indices and their override values.

        Default overrides are:
            1: 3 - use the parallel (OpenMP) version of the nested dissection algorithm.
            4: 1 if has_perm is True (i.e., if the user provided a permutation)
            7: 10 or {max_iter_refine} - iterative refinement step;
            10: 1 if matrix is not hermitian - scaling vectors;
            12: 1 if input matrix is not hermitian (metric weighted matching)
            23: 1 - use two-level factorization algorithm (improves scalability in case of parallel factorization on many OpenMP threads (>8));
            26: 0 - matrix checker. 0 means PARDISO does not check the sparse matrix representation for errors.


    Notes
    -----
    Overrides are applied before PARDISO's analysis and factorization phases. The keys
    are zero-based indices for pydiso's ``iparm`` array and match the notation used by Intel's C documentation.

    ``pydiso``'s default values:
        iparm[0] = 1  # tell pardiso to not reset these values on the first call
        iparm[1] = 2  # The nested dissection algorithm from the METIS (NOTE: overriden by us)
        iparm[3] = 0  # The factorization is always computed as required by phase.
        iparm[4] = 2  # fill perm with computed permutation vector (NOTE: we want(ed) to override it)
        iparm[5] = 0  # The array x contains the solution; right-hand side vector b is kept unchanged.
        iparm[7] = 0  # The solver automatically performs two steps of iterative refinement when perterbed pivots are obtained (NOTE: overriden by us)
        iparm[9] = 13 if matrix_type in [11, 13] else 8
        iparm[10] = 1 if matrix_type in [11, 13] else 0 (NOTE: overidden by us)
        iparm[11] = 0  # Solve a linear system AX = B (as opposed to A.T or A.H)
        iparm[12] = 1 if matrix_type in [11, 13] else 0 (NOTE: overriden by us)
        iparm[17] = -1  # Return the number of non-zeros in this value after first call (We used to set it too)
        iparm[18] = 0  # do not report flop count
        iparm[20] = 1 if matrix_type in [-2, -4, 6] else 0 (We used to set it too, but it was always zero regardless matrix type)
        iparm[23] = 0  # classic (not parallel) factorization (NOTE: overriden by us)
        iparm[24] = 0  # Parallel forward/backward solve control (default option)
        iparm[26] = 1  # Check the input matrix for errors (NOTE: overriden by us)
        iparm[27] = is_single_precision  # 1 if single, 0 if double
        iparm[30] = 0  # this would be used to enable sparse input/output for solves
        iparm[33] = 0  # optimal number of thread for CNR mode
        iparm[34] = 1  # zero based indexing (We used to set it too)
        iparm[35] = 0  # Do not compute schur complement
        iparm[36] = 0  # use CSR storage format
        iparm[38] = 0  # Do not use low rank update
        iparm[42] = 0  # Do not compute the diagonal of the inverse
        iparm[55] = 0  # Internal function used to work with pivot and calculation of diagonal arrays turned off.
        iparm[59] = 0  # operate in-core mode
    """
    overrides = {
        1: 3,
        7: max_iter_refine,
        23: 1,
        26: 0,
    }

    # Add extra arguments in case of non-Hermitian matrix type
    # TODO: does it really work correctly for non-hermitian matrices?
    if not hermitian:
        overrides |= {
            10: int(scaling_vectors),
            12: int(weighted_matching),
        }
    if has_perm:
        iparm[4] = 1  # support user-provided permutations
    return overrides


# TODO: Hermitian is set to 1 in qutip/tests/test_mkl.py, so there are no tests for nonsymmetric matrices (i.e., the path where the upper triangular matrix is taken is not tested)


class MKLFactorization:
    """
    Factorization of a sparse matrix using Intel oneMKL PARDISO.

    This class adapts :class:`pydiso.mkl_solver.MKLPardisoSolver` to QuTiP's sparse-factorization interface. The factorization can be reused to solve systems with multiple right-hand sides.

    Parameters
    ----------
    solver : pydiso.mkl_solver.MKLPardisoSolver
        Initialized and factorized pydiso solver.
    matrix_type : int
        PARDISO matrix-type code used for the factorization.
    dtype : numpy.dtype
        Data type of the factorized matrix.
    factor_time : float
        Elapsed factorization time in seconds.

    Examples
    --------
    Reuse a factorization for multiple right-hand sides:

    TODO: example

    """

    def __init__(
        self,
        solver: MKLPardisoSolver,
        matrix_type: int,
        dtype: np.dtype,
        factor_time: float,
    ):
        self._solver = solver
        self._matrix_type = matrix_type
        self._dtype = dtype
        self._is_complex = np.issubdtype(dtype, np.complexfloating)
        self._factor_time = factor_time
        self._solve_time = None
        self._info = None

    def solve(self, b, verbose=False):
        """
        Solve a sparse linear system using the stored factorization.

        Parameters
        ----------
        b : array_like
            Dense right-hand side with shape ``(n,)`` or ``(n, k)``.
        verbose : bool

        Returns
        -------
        numpy.ndarray
            Solution with the same shape as ``b`` and the data type of the
            factorized matrix.

        Raises
        ------
        RuntimeError
            If the factorization has been closed.
        TypeError
            If ``b`` is sparse, or if it is complex while the factorization is
            real-valued.
        ValueError
            If the shape of ``b`` is incompatible with the factorized matrix.
        """

        if self._solver is None:
            raise RuntimeError(
                "Solver's memory has been released. Initialise MKLFactorization again."
            )

        if sp.issparse(b):
            raise TypeError(
                "Right-hand side must be dense. Use mkl_spsolve for a sparse b"
                "instead of using MKLFactorization.solve directly"
            )
        b = np.asarray(b)

        if np.issubdtype(b.dtype, np.complexfloating) and not self._is_complex:
            raise TypeError(
                "Got a complex right-hand side: cannot solve real-valued factorization"
            )

        if b.dtype != self._dtype:
            # Pydiso wrapper would do the conversion and throw a warning;
            # hence, we do data type conversion in advance
            b = b.astype(self._dtype)

        _solve_start = time.perf_counter()
        x = self._solver.solve(b)
        self._solve_time = time.perf_counter() - _solve_start
        return x

    def info(self):
        """
        Return statistics for the most recent factorization and solve.

        Returns
        -------
        dict
            Factorization and solution statistics. ``FactorTime`` and
            ``SolveTime`` are measured in seconds; ``Factormem`` and
            ``Solvemem`` are measured in MiB; and ``IterRefine`` is the number
            of iterative refinement steps. ``SolveTime`` is ``None`` before
            the first solve.
        """

        if self._solver is None:
            return self._info
        iparm = self._solver.iparm  # TODO: is it a legal way to access iparm values?
        return {
            "FactorTime": self._factor_time,
            "SolveTime": self._solve_time,
            "Factormem": round(iparm[15] / 1024, 4),
            "Solvemem": round(iparm[16] / 1024, 4),
            "IterRefine": iparm[6],
        }

    def delete(self):
        """
        Release this object's reference
        """

        # Preserve the statistics info before memory deallocation
        self._info = self.info()
        self._solver = None


_MATRIX_TYPE_NAMES = {
    4: "Complex Hermitian positive-definite",
    -4: "Complex Hermitian indefinite",
    2: "Real symmetric positive-definite",
    -2: "Real symmetric indefinite",
    11: "Real non-symmetric",
    13: "Complex non-symmetric",
}


def _mkl_matrix_type(dtype, hermitian, posdef):
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if not hermitian:
        return 13 if is_complex else 11
    out = 4 if is_complex else 2
    return out if posdef else -out


# Returns factorisation object: important for tests
def mkl_splu(
    A,
    perm=None,
    verbose=False,
    *,
    hermitian=False,
    posdef=False,
    max_iter_refine=10,
    scaling_vectors=True,  # TODO: reflect the fact that this parameter will be used only if the matrix is non Hermitian
    weighted_matching=True,
):
    """
    Returns the LU factorization of the sparse matrix A.

    Parameters
    ----------
    A : csr_matrix
        Sparse input matrix.
    perm : ndarray (optional)
        User defined matrix factorization permutation.
    verbose : bool {False, True}
        Report factorization details.

    Returns
    -------
    lu : MKLFactorization
        Returns object containing LU factorization with a
        solve method for solving with a given RHS vector.

    """
    if not (sp.issparse(A) and A.format == "csr"):
        raise TypeError("Input matrix must be in sparse CSR format.")

    if A.shape[0] != A.shape[1]:
        raise Exception("Input matrix must be square")

    if not np.issubdtype(A.dtype, np.inexact):
        A = sp.csr_matrix(A, dtype=np.float64, copy=False)

    data_type = A.dtype

    matrix_type = _mkl_matrix_type(data_type, hermitian, posdef)

    # TODO: evaluate pydiso's logging capabilities: what is there and what we should add
    # if verbose:
    #     print('Solver Initialization')
    #     print('---------------------')
    #     print('Input matrix type: ', _MATRIX_TYPE_NAMES[mtype])
    #     print('Input matrix shape:', A.shape)
    #     print('Input matrix NNZ:  ', A.nnz)
    #     print()
    if perm is not None:
        raise NotImplementedError(
            "User-defined permutations are not supported by the pydiso backend."
        )
    # Call solver # TODO: here, we will call the solver
    _factor_start = time.perf_counter()
    iparms = _iparm_overrides(
        hermitian=hermitian,
        max_iter_refine=max_iter_refine,
        scaling_vectors=scaling_vectors,
        weighted_matching=weighted_matching,
        has_perm=False
    )
    solver = MKLPardisoSolver(
        A, matrix_type=matrix_type, verbose=verbose, iparm_overrides=iparms
    )
    _factor_time = time.perf_counter() - _factor_start
    # if verbose:
    #     print('Analysis and Factorization Stage')
    #     print('--------------------------------')
    #     print('Factorization time:       ', round(_factor_time, 4))
    #     print('Factorization memory (Mb):', round(solver.iparm[15]/1024, 4))
    #     print('NNZ in LU factors:        ', solver.iparm[17])
    #     print()
    return MKLFactorization(solver, matrix_type, data_type, _factor_time)


# TODO: issue: we cannot use perm parameter with pydiso solver
def mkl_spsolve(
    A,
    b,
    perm=None,
    verbose=False,
    *,
    return_info=False,
    hermitian=False,
    posdef=False,
    max_iter_refine=10,
    scaling_vectors=True,
    weighted_matching=True,
):
    """
    Solves a sparse linear system of equations using the
    Intel MKL Pardiso solver.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Sparse input matrix.
    b : ndarray, scipy.sparse.csr_matrix or scipy.sparse.csr_array
        The vector or matrix representing the right hand side of the equation. If a vector, b.shape must be (n,) or (n, 1).
    perm : ndarray (optional)
        User defined matrix factorization permutation.

    Returns
    -------
    x : ndarray, scipy.sparse.csr_matrix or scipy.sparse.csr_array
        The solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    """
    A = sp.csr_matrix(A)
    lu = mkl_splu(
        A,
        perm=perm,
        verbose=verbose,
        hermitian=hermitian,
        posdef=posdef,
        max_iter_refine=max_iter_refine,
        scaling_vectors=scaling_vectors,
        weighted_matching=weighted_matching,
    )
    try:
        return_sparse = sp.issparse(b) and b.ndim == 2 and b.shape[1] != 1
        if sp.issparse(b):
            # qutip's convention: a sparse RHS of shape (n, 1) produces dense solution
            b = b.toarray(order="F")
        x = lu.solve(b, verbose=verbose)
        if return_sparse:
            x = sp.csr_matrix(x)

        info = lu.info()
    finally:
        lu.delete()
    return (x, info) if return_info else x
