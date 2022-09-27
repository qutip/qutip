from qutip.core.data import CSR, Data, csr, Dense
import qutip.core.data as _data
import scipy.sparse.linalg as splinalg
import numpy as np
from qutip.settings import settings
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve
else:
    mkl = None


def _splu(A, B, **kwargs):
    lu = splinalg.splu(A, **kwargs)
    return lu.solve(B)


def solve_csr(matrix: CSR, target: Data, method: str ="spsolve",
              options: dict={}) -> Data:
    """
    Solve ``Ax=b`` for ``x``.

    Parameters:
    -----------

    matrix : CSR
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

    if isinstance(target, CSR) and csr.nnz(target) < np.prod(target.shape)*0.1:
        b = target.as_scipy()
    elif isinstance(target, Dense):
        b = target.as_ndarray()
    else:
        b = target.to_array()


    if method == "splu":
        solver = _splu
    elif hasattr(splinalg, method):
        solver = getattr(splinalg, method)
    elif method in ["mkl_spsolve", "mkl"]:
        solver = getattr(mkl, method)
    elif method.startswith("mkl") and mkl is None:
        raise ValueError("mkl is not available")
    else:
        raise ValueError(f"Unknown sparse solver {method}.")

    M = matrix.as_scipy()
    if options.pop("csc", False):
        M = M.tocsc()

    out = solver(matrix.as_scipy(), b, **options)

    if isinstance(out, tuple):
        out, check = out
        if check == 0:
            # Successful
            pass
        elif check > 0:
            raise RunTimeError(
                f"scipy.sparse.linalg.{method} error: Tolerance was not"
                " reached. Error code: {check}"
            )

        elif check < 0:
            raise RunTimeError(
                f"scipy.sparse.linalg.{method} error: Bad input. "
                "Error code: {check}"
            )
    # spsolve can return both scipy sparse matrix or ndarray
    return _data.create(out, copy=False)


def solve_dense(matrix: Dense, target: Data, method: str="solve",
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


    if method == "solve":
        out = np.linalg.solve(matrix.as_ndarray(), b)
    elif method == "lstsq":
        out, *_ = np.linalg.lstsq(
            matrix.as_ndarray(),
            b,
            rcond=options.get("rcond", None)
        )
    else:
        raise ValueError(f"Unknown solver {method},"
                         " 'solve' and 'lstsq' are supported.")
    print(out)
    return _data.create(out, copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


solve = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('target', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('method', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('options', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
    (CSR, CSR, Dense, solve_csr),
    (CSR, Dense, Dense, solve_csr),
    (Dense, CSR, Dense, solve_dense),
    (Dense, Dense, Dense, solve_dense),
], _defer=True)
