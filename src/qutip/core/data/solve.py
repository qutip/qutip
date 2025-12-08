from qutip.core.data import CSR, Data, csr, Dense, Dia
import qutip.core.data as _data
import scipy.sparse.linalg as splinalg
import numpy as np
from qutip.settings import settings
import warnings
from typing import Union

if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve
else:
    mkl_spsolve = None


__all__ = ["solve_csr_dense", "solve_dia_dense", "solve_dense", "solve"]


def _splu(A, B, **kwargs):
    lu = splinalg.splu(A, **kwargs)
    return lu.solve(B)


def solve_csr_dense(matrix: Union[CSR, Dia], target: Dense, method=None,
                    options: dict={}) -> Dense:
    """
    Solve ``Ax=b`` for ``x``.

    Parameters:
    -----------

    matrix : CSR, Dia
        The matrix ``A``.

    target : Data
        The matrix or vector ``b``.

    method : str {"spsolve", "splu", "mkl_spsolve", etc.}, default="spsolve"
        The function to use to solve the system. Any function from
        scipy.sparse.linalg which solve the equation Ax=b can be used.
        `splu` from the same and `mkl_spsolve` are also valid choice.

    options : dict
        Keywork options to pass to the solver. Refer to the documenentation in
        scipy.sparse.linalg of the used method for a list of supported keyword.
        The keyword "csc" can be set to ``True`` to convert the sparse matrix
        before passing it to the solver.

    .. note::
        Options for ``mkl_spsolve`` are presently only found in the source
        code.

    Returns:
    --------
    x : Dense
        Solution to the system Ax = b.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only solve using square matrix")
    if matrix.shape[1] != target.shape[0]:
        raise ValueError("target does not match the system")

    b = target.as_ndarray()

    method = method or "spsolve"

    if method == "splu":
        solver = _splu
    elif method == "lstsq":
        solver = splinalg.lsqr
    elif method == "solve":
        solver = splinalg.spsolve
    elif hasattr(splinalg, method):
        solver = getattr(splinalg, method)
    elif method == "mkl_spsolve" and mkl_spsolve is None:
        raise ValueError("mkl is not available")
    elif method == "mkl_spsolve":
        solver = mkl_spsolve
        # mkl does not support dia.
        if isinstance(matrix, Dia):
            matrix = _data.to("CSR", matrix)
    else:
        raise ValueError(f"Unknown sparse solver {method}.")

    options = options.copy()
    M = matrix.as_scipy()
    if options.pop("csc", False) or isinstance(matrix, Dia):
        M = M.tocsc()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            out = solver(M, b, **options)
        except splinalg.MatrixRankWarning:
            raise ValueError("Matrix is singular")

    if isinstance(out, tuple) and len(out) == 2:
        # iterative method return a success flag
        out, check = out
        if check == 0:
            # Successful
            pass
        elif check > 0:
            raise RuntimeError(
                f"scipy.sparse.linalg.{method} error: Tolerance was not"
                f" reached. Error code: {check}"
            )

        elif check < 0:
            raise RuntimeError(
                f"scipy.sparse.linalg.{method} error: Bad input. "
                f"Error code: {check}"
            )
    elif isinstance(out, tuple) and len(out) > 2:
        # least sqare method return residual, flag, etc.
        out, *_ = out
    return Dense(out, copy=False)


solve_dia_dense = solve_csr_dense


def solve_dense(matrix: Dense, target: Data, method=None,
                options: dict={}) -> Dense:
    """
    Solve ``Ax=b`` for ``x``.

    Parameters:
    -----------

    matrix : Dense
        The matrix ``A``.

    target : Data
        The matrix or vector ``b``.

    method : str {"solve", "lstsq"}, default="solve"
        The function from numpy.linalg to use to solve the system.

    options : dict
        Options to pass to the solver. "lstsq" use "rcond" while, "solve" do
        not use any.

    Returns:
    --------
    x : Dense
        Solution to the system Ax = b.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only solve using square matrix")
    if matrix.shape[1] != target.shape[0]:
        raise ValueError("target does not match the system")

    if isinstance(target, Dense):
        b = target.as_ndarray()
    else:
        b = target.to_array()

    if method in ["solve", None]:
        try:
            out = np.linalg.solve(matrix.as_ndarray(), b)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular")
    elif method == "lstsq":
        out, *_ = np.linalg.lstsq(
            matrix.as_ndarray(),
            b,
            rcond=options.get("rcond", None)
        )
    else:
        raise ValueError(f"Unknown solver {method},"
                         " 'solve' and 'lstsq' are supported.")
    return Dense(out, copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


solve = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('target', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('method', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=None),
        _inspect.Parameter('options', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default={}),
    ]),
    name='solve',
    module=__name__,
    inputs=('matrix', 'target'),
    out=True,
)
solve.__doc__ = """
    Solve ``Ax=b`` for ``x``.

    Parameters:
    -----------
    matrix : Data
    The matrix ``A``.

    target : Data
    The matrix or vector ``b``.

    method : str
        The function to use to solve the system. Function which solve the
        equation Ax=b from scipy.sparse.linalg (CSR ``matrix``) or
        numpy.linalg (Dense ``matrix``) can be used.
        Sparse cases also accept `splu` and `mkl_spsolve`.
        `solve` and `lstsq` will work for any data-type.

    options : dict
        Keywork options to pass to the solver. Refer to the documenentation
        for the chosen method in scipy.sparse.linalg or numpy.linalg.
        The keyword "csc" can be set to ``True`` to convert the sparse matrix
        in sparse cases.

    .. note::
        Options for ``mkl_spsolve`` are presently only found in the source
        code.

    Returns:
    --------
    x : Data
        Solution to the system Ax = b.
"""
solve.add_specialisations([
    (CSR, Dense, Dense, solve_csr_dense),
    (Dia, Dense, Dense, solve_dia_dense),
    (Dense, Dense, Dense, solve_dense),
], _defer=True)


del _Dispatcher
del _inspect
