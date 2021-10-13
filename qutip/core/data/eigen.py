import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .dense import Dense, from_csr
from .csr import CSR
from .properties import isherm as _isherm
from qutip.settings import settings

__all__ = [
    'eigs', 'eigs_csr', 'eigs_dense',
]


if settings.install["eigh_unsafe"]:
    def _orthogonalize(vec, other):
        cross = np.sum(np.conj(other) * vec)
        vec -= cross * other
        norm = np.sum(np.conj(vec) * vec)**0.5
        vec /= norm

    def eigh(mat, eigvals=None):
        val, vec = scipy.linalg.eig(mat)
        val = np.real(val)
        idx = np.argsort(val)
        val = val[idx]
        vec = vec[:, idx]
        if eigvals:
            val = val[eigvals[0]:eigvals[1]+1]
            vec = vec[:, eigvals[0]:eigvals[1]+1]
        same_eigv = 0
        for i in range(1, len(val)):
            if abs(val[i] - val[i-1]) < 1e-12:
                same_eigv += 1
                for j in range(same_eigv):
                    _orthogonalize(vec[:, i], vec[:, i-j-1])
            else:
                same_eigv = 0
        return val, vec

    def eigvalsh(a, eigvals=None):
        val = scipy.linalg.eigvals(a)
        val = np.sort(np.real(val))
        if eigvals:
            return val[eigvals[0]:eigvals[1]+1]
        return val
else:
    eigh = scipy.linalg.eigh
    eigvalsh = scipy.linalg.eigvalsh


def _eigs_dense(data, isherm, vecs, eigvals, num_large, num_small):
    """
    Internal functions for computing eigenvalues and eigenstates for a dense
    matrix.
    """
    N = data.shape[0]
    kwargs = {}
    if eigvals != 0 and isherm:
        kwargs['eigvals'] = ([0, num_small-1] if num_small
                             else [N-num_large, N-1])
    if vecs:
        driver = eigh if isherm else scipy.linalg.eig
        evals, evecs = driver(data, **kwargs)
    else:
        driver = eigvalsh if isherm else scipy.linalg.eigvals
        evals = driver(data, **kwargs)
        evecs = None

    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))

    if vecs:
        evecs = np.array([evecs[:, k] for k in perm]).T

    if not isherm and eigvals > 0:
        if vecs:
            if num_small > 0:
                evals, evecs = evals[:num_small], evecs[:num_small]
            elif num_large > 0:
                evals, evecs = evals[(N - num_large):], evecs[(N - num_large):]
        else:
            if num_small > 0:
                evals = evals[:num_small]
            elif num_large > 0:
                evals = evals[(N - num_large):]
    return np.array(evals), evecs


def _eigs_csr(data, isherm, vecs, eigvals, num_large, num_small, tol, maxiter):
    """
    Internal functions for computing eigenvalues and eigenstates for a sparse
    matrix.
    """
    N = data.shape[0]
    big_vals = np.array([])
    small_vals = np.array([])
    evecs = None

    remove_one = False
    if eigvals == (N - 1):
        # calculate all eigenvalues and remove one at output if using sparse
        eigvals = 0
        num_small = num_large = N // 2
        num_small += N % 2
        remove_one = True

    if vecs:
        if isherm:
            if num_large > 0:
                big_vals, big_vecs = sp.linalg.eigsh(data, k=num_large,
                                                     which='LA', tol=tol,
                                                     maxiter=maxiter)
            if num_small > 0:
                small_vals, small_vecs = sp.linalg.eigsh(
                    data, k=num_small, which='SA',
                    tol=tol, maxiter=maxiter)

        else:
            if num_large > 0:
                big_vals, big_vecs = sp.linalg.eigs(data, k=num_large,
                                                    which='LR', tol=tol,
                                                    maxiter=maxiter)
            if num_small > 0:
                small_vals, small_vecs = sp.linalg.eigs(
                    data, k=num_small, which='SR',
                    tol=tol, maxiter=maxiter)

        if num_large != 0 and num_small != 0:
            evecs = np.hstack([small_vecs, big_vecs])
        elif num_large != 0 and num_small == 0:
            evecs = big_vecs
        elif num_large == 0 and num_small != 0:
            evecs = small_vecs
    else:
        if isherm:
            if num_large > 0:
                big_vals = sp.linalg.eigsh(
                    data, k=num_large, which='LA',
                    return_eigenvectors=False, tol=tol, maxiter=maxiter)
            if num_small > 0:
                small_vals = sp.linalg.eigsh(
                    data, k=num_small, which='SA',
                    return_eigenvectors=False, tol=tol, maxiter=maxiter)
        else:
            if num_large > 0:
                big_vals = sp.linalg.eigs(
                    data, k=num_large, which='LR',
                    return_eigenvectors=False, tol=tol, maxiter=maxiter)
            if num_small > 0:
                small_vals = sp.linalg.eigs(
                    data, k=num_small, which='SR',
                    return_eigenvectors=False, tol=tol, maxiter=maxiter)

    evals = np.hstack((small_vals, big_vals))
    if isherm:
        evals = np.real(evals)

    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))

    if vecs:
        evecs = np.array([evecs[:, k] for k in perm]).T

    # remove last element if requesting N-1 eigs and using sparse
    if remove_one:
        evals = np.delete(evals, -1)
        if vecs:
            evecs = np.delete(evecs, -1)

    return np.array(evals), evecs


def _eigs_check_shape(data):
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")


def _eigs_fix_eigvals(data, eigvals, sort):
    N = data.shape[0]
    if eigvals == N:
        eigvals = 0
    if eigvals > N:
        raise ValueError("Number of requested eigen vals/vecs must be <= N.")
    if sort not in ('low', 'high'):
        raise ValueError("'sort' must be 'low' or 'high'")

    # set number of large and small eigenvals/vecs
    if eigvals == 0:  # user wants all eigs (default)
        num_small = num_large = N // 2
        num_small += N % 2
    else:  # if user wants only a few eigen vals/vecs
        num_small, num_large = (eigvals, 0) if sort == 'low' else (0, eigvals)
    return eigvals, num_large, num_small


def eigs_csr(data, isherm=None, vecs=True, sort='low', eigvals=0,
             tol=0, maxiter=100000):
    """
    Return eigenvalues and eigenvectors for a CSR matrix.  This specialisation
    may take some extra keyword arguments in addition to the full documentation
    specified in :func:`.eigs`.

    This method is typically slower and less accurate than the dense eigenvalue
    solver; you probably want that, unless memory concerns deem it impossible.

    Extra keyword arguments
    -----------------------
    tol : float (0)
        Tolerance for sparse eigensolver.  Sufficiently small tolerances (such
        as 0) cause the solver to use machine precision.
    maxiter : int (100_000)
        Max number of iterations used by sparse eigensolver.
    """
    if not isinstance(data, CSR):
        raise TypeError("expected data in CSR format but got "
                        + str(type(data)))
    if data.shape[0] < 4:
        # For small matrix, the sparse solver can't compute all eigenvalues.
        return eigs_dense(from_csr(data), isherm, vecs, sort, eigvals)
    _eigs_check_shape(data)
    eigvals, num_large, num_small = _eigs_fix_eigvals(data, eigvals, sort)
    isherm = isherm if isherm is not None else _isherm(data)
    evals, evecs = _eigs_csr(data.as_scipy(), isherm, vecs, eigvals,
                             num_large, num_small, tol, maxiter)
    if sort == 'high':
        # Flip arrays around.
        if vecs:
            evecs = np.fliplr(evecs)
        evals = evals[::-1]
    return (evals, Dense(evecs, copy=False)) if vecs else evals


def eigs_dense(data, isherm=None, vecs=True, sort='low', eigvals=0):
    """
    Return eigenvalues and eigenvectors for a Dense matrix.  Takes no special
    keyword arguments; see the primary documentation in :func:`.eigs`.
    """
    if not isinstance(data, Dense):
        raise TypeError("expected data in Dense format but got "
                        + str(type(data)))
    _eigs_check_shape(data)
    eigvals, num_large, num_small = _eigs_fix_eigvals(data, eigvals, sort)
    isherm = isherm if isherm is not None else _isherm(data)
    evals, evecs = _eigs_dense(data.as_ndarray(), isherm, vecs, eigvals,
                               num_large, num_small)
    if sort == 'high':
        # Flip arrays around.
        if vecs:
            evecs = np.fliplr(evecs)
        evals = evals[::-1]
    return (evals, Dense(evecs, copy=False)) if vecs else evals


from .dispatch import Dispatcher as _Dispatcher

# We use eigs_dense as the signature source, since in this case it has the
# complete signature that we allow, so we don't need to manually set it.
eigs = _Dispatcher(eigs_dense, name='eigs', inputs=('data',), out=False)
eigs.__doc__ =\
    """
    Return eigenvalues and (optionally) eigenvectors for a data-layer object.

    Some particular specialisations of this function may take additional
    keyword arguments (such as the CSR solver).  See their particular
    docstrings for details on those.

    Parameters
    ----------
    data : Data
        Input matrix
    isherm : bool, optional
        Indicate whether the matrix is Hermitian or not.  There are special
        Hermitian eigenvalue and -vector solvers for this case, which will take
        care of orthonormalisation and ensuring better accuracy.  If this is
        not specified either way, it will be calculated from the data.
    vecs : bool, optional (True)
        Whether the eigenvectors should be returned as well.
    sort : {'low', 'high'}, optional
        Sort the output of the eigenvalues and -vectors ordered by the relevant
        size of the real part of the eigenvalue from 'low' to high or from
        'high' to low.  If not all of the eigenvalues are requested, this
        influences which eigenvalues will be found.
    eigvals : int, optional
        Number of eigenvalues and -vectors to return.  If `0`, then returns
        all.

    Returns
    -------
    eigenvalues : np.ndarray
        The requested eigenvalues, sorted in the expected order.  The dtype is
        `np.complex128`, unless `isherm=True`, in which case it will be
        `np.float64`.
    eigenvectors : Data
        Only if `vecs=True`.  An array of the eigenvectors corresponding to the
        order of the eigenvalues.
    """
eigs.add_specialisations([
    (CSR, eigs_csr),
    (Dense, eigs_dense),
], _defer=True)

del _Dispatcher
