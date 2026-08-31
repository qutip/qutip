import numpy as np
import scipy.sparse as sp
import time

from pydiso.mkl_solver import MKLPardisoSolver

def _prepare_pydiso_args(
    max_iter_refine: int,
    perm = None
):
    """
    Maps QuTiP's PARDISO ``iparm`` overrides to keyword-named arguments for the pydiso solver.

    Parameters
    ----------
    max_iter_refine : int
        Maximum iterative-refinement steps. A value of ``0`` selects
        PARDISO's automatic refinement behavior.
    perm : array_like (optional)
        User permutation.

    Returns
    -------
    dict[string, value]
        Mapping of ``iparm``-related pydiso's arguments to values.

    Notes
    -----
    QuTiP overrides the following iparm arguments:
        - ``iparm[7]`` (maximum iterative refinement steps) is user-provided. Set via ``max_iterative_refinement_steps`` argument of MKLPardisoSolver
        - ``iparm[26]`` (parallel factorization) is set to 1 via ``parallel_factorization`` argument of MKLPardisoSolver
        - ``iparm[1]`` (fill-in reducing permutation) is set to 3 via ``fill_reducing_ordering`` argument of MKLPardisoSolver
          - Note: if a user-defined permutation is passed via ``perm``, value for ``iparm[1]`` will be ignored in pydiso.

    The rest of ``iparms`` is handled by pydiso's defaults.
    """
    overrides = {
        "fill_reducing_ordering": perm if perm else 3,
        "max_iterative_refinement_steps": max_iter_refine,
        "parallel_factorization": True,
    }
    return overrides


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
            If the factorization has been deleted (dereferenced).
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
        iparm = (
            self._solver.iparm
        )
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


def _mkl_matrix_type(dtype, hermitian, posdef):
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if not hermitian:
        return 13 if is_complex else 11
    out = 4 if is_complex else 2
    return out if posdef else -out


def mkl_splu(
    A,
    perm=None,
    verbose=False,
    *,
    hermitian=False,
    posdef=False,
    max_iter_refine=10,
):
    """
    Returns the LU factorization of the sparse matrix A.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Sparse input matrix in CSR format.
    perm : None, optional
        User-defined permutations.
    verbose : bool, default: False
        Report factorization details.
    hermitian : bool, default: False
        Treat ``A`` as Hermitian when selecting the PARDISO matrix type.
    posdef : bool, default: False
        Treat a Hermitian matrix as positive-definite.
    max_iter_refine : int, default: 10
        Maximum iterative-refinement steps. Use ``0`` for PARDISO's
        automatic behavior.

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
        A = sp.csr_array(A, dtype=np.float64, copy=None)

    data_type = A.dtype

    matrix_type = _mkl_matrix_type(data_type, hermitian, posdef)

    # Call solver
    _factor_start = time.perf_counter()
    iparms = _prepare_pydiso_args(
        max_iter_refine=max_iter_refine,
        perm=perm
    )
    solver = MKLPardisoSolver(
        A, matrix_type=matrix_type, verbose=verbose, **iparms
    )
    _factor_time = time.perf_counter() - _factor_start
    return MKLFactorization(solver, matrix_type, data_type, _factor_time)


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
):
    """
    Solves a sparse linear system of equations using the
    Intel MKL Pardiso solver.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Sparse input matrix.
    b : ndarray, scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Vector or matrix representing the right-hand side. A vector must have
        shape ``(n,)`` or ``(n, 1)``.
    perm : None, optional
        User-defined permutations are not currently supported. Passing a
        value other than ``None`` raises ``NotImplementedError``.
    verbose : bool, default: False
        Report factorization details.
    return_info : bool, default: False
        Return solver statistics together with the solution.
    hermitian : bool, default: False
        Treat ``A`` as Hermitian when selecting the PARDISO matrix type.
    posdef : bool, default: False
        Treat a Hermitian matrix as positive-definite.
    max_iter_refine : int, default: 10
        Maximum iterative-refinement steps. Use ``0`` for PARDISO's
        automatic behavior.
    Returns
    -------
    x : ndarray, scipy.sparse.csr_matrix or scipy.sparse.csr_array
        The solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    """
    A = sp.csr_array(A)
    lu = mkl_splu(
        A,
        perm=perm,
        verbose=verbose,
        hermitian=hermitian,
        posdef=posdef,
        max_iter_refine=max_iter_refine,
    )
    try:
        return_sparse = sp.issparse(b) and b.ndim == 2 and b.shape[1] != 1
        if sp.issparse(b):
            b = b.toarray(order="F")
        x = lu.solve(b, verbose=verbose)
        if return_sparse:
            x = sp.csr_matrix(x)

        info = lu.info()
    finally:
        lu.delete()
    return (x, info) if return_info else x
