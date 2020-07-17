import numpy as np
import scipy.linalg
import scipy.sparse as sp

__all__ = ['eigs_csr']


def _dense_eigs(data, isherm, vecs, N, eigvals, num_large, num_small):
    """
    Internal functions for computing eigenvalues and eigenstates for a dense
    matrix.
    """
    kwargs = {}
    if eigvals != 0 and isherm:
        kwargs['eigvals'] = ([0, num_small-1] if num_small
                             else [N-num_large, N-1])
    if vecs:
        driver = scipy.linalg.eigh if isherm else scipy.linalg.eig
        evals, evecs = driver(data, **kwargs)
    else:
        driver = scipy.linalg.eigvalsh if isherm else scipy.linalg.eigvals
        evals = driver(data, **kwargs)
        evecs = None

    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))

    if vecs:
        evecs = np.array([evecs[:, k] for k in perm])

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

    return np.array(evals), np.array(evecs)


def _sp_eigs(data, isherm, vecs, N, eigvals, num_large, num_small, tol,
             maxiter):
    """
    Internal functions for computing eigenvalues and eigenstates for a sparse
    matrix.
    """
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
                big_vecs = sp.csr_matrix(big_vecs, dtype=complex)
            if num_small > 0:
                small_vals, small_vecs = sp.linalg.eigsh(
                    data, k=num_small, which='SA',
                    tol=tol, maxiter=maxiter)

        else:
            if num_large > 0:
                big_vals, big_vecs = sp.linalg.eigs(data, k=num_large,
                                                    which='LR', tol=tol,
                                                    maxiter=maxiter)
                big_vecs = sp.csr_matrix(big_vecs, dtype=complex)
            if num_small > 0:
                small_vals, small_vecs = sp.linalg.eigs(
                    data, k=num_small, which='SR',
                    tol=tol, maxiter=maxiter)

        if num_large != 0 and num_small != 0:
            evecs = sp.hstack([small_vecs, big_vecs], format='csr')
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
        evecs = np.array([evecs[:, k] for k in perm])

    # remove last element if requesting N-1 eigs and using sparse
    if remove_one:
        evals = np.delete(evals, -1)
        if vecs:
            evecs = np.delete(evecs, -1)

    return np.array(evals), np.array(evecs)


def eigs_csr(data, isherm, vecs=True, sparse=False, sort='low',
             eigvals=0, tol=0, maxiter=100000):
    """Returns Eigenvalues and Eigenvectors for a sparse matrix.
    Uses dense eigen-solver unless user sets sparse=True.

    Parameters
    ----------
    data : csr_matrix
        Input matrix
    isherm : bool
        Indicate whether the matrix is hermitian or not
    vecs : bool {True , False}
        Flag for requesting eigenvectors
    sparse : bool {False , True}
        Flag to use sparse solver
    sort : str {'low' , 'high}
        Return lowest or highest eigenvals/vecs
    eigvals : int
        Number of eigenvals/vecs to return.  Default = 0 (return all)
    tol : float
        Tolerance for sparse eigensolver.  Default = 0 (Machine precision)
    maxiter : int
        Max. number of iterations used by sparse sigensolver.

    Returns
    -------
    Array of eigenvalues and (by default) array of corresponding Eigenvectors.

    """
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")
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

    # Dispatch to sparse/dense solvers
    if sparse:
        evals, evecs = _sp_eigs(data.as_scipy(), isherm, vecs, N, eigvals,
                                num_large, num_small, tol, maxiter)
    else:
        evals, evecs = _dense_eigs(data.to_array(), isherm, vecs, N, eigvals,
                                   num_large, num_small)

    if sort == 'high':  # flip arrays to largest values first
        if vecs:
            evecs = np.flipud(evecs)
        evals = np.flipud(evals)

    return (evals, evecs) if vecs else evals
