__all__ = ['simdiag']

import numpy as np
import scipy.linalg as la
from qutip.qobj import Qobj


def simdiag(ops, evals=True):
    """Simultaneous diagonalization of commuting Hermitian matrices.

    Parameters
    ----------
    ops : list/array
        ``list`` or ``array`` of qobjs representing commuting Hermitian
        operators.

    Returns
    --------
    eigs : tuple
        Tuple of arrays representing eigvecs and eigvals of quantum objects
        corresponding to simultaneous eigenvectors and eigenvalues for each
        operator.

    """
    tol = 1e-14
    start_flag = 0
    if not any(ops):
        raise ValueError('Need at least one input operator.')
    if not isinstance(ops, (list, np.ndarray)):
        ops = np.array([ops])
    num_ops = len(ops)
    for jj in range(num_ops):
        A = ops[jj]
        shape = A.shape
        if shape[0] != shape[1]:
            raise TypeError('Matricies must be square.')
        if start_flag == 0:
            s = shape[0]
        if s != shape[0]:
            raise TypeError('All matrices. must be the same shape')
        if not A.isherm:
            raise TypeError('Matricies must be Hermitian')
        for kk in range(jj):
            B = ops[kk]
            if (A * B - B * A).norm() / (A * B).norm() > tol:
                raise TypeError('Matricies must commute.')

    A = ops[0]
    eigvals, eigvecs = la.eig(A.full())
    zipped = list(zip(-eigvals, range(len(eigvals))))
    zipped.sort()
    ds, perm = zip(*zipped)
    ds = -np.real(np.array(ds))
    perm = np.array(perm)
    eigvecs_array = np.array(
        [np.zeros((A.shape[0], 1), dtype=complex) for k in range(A.shape[0])])

    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        eigvecs_array[kk][:, 0] = eigvecs[:, perm[kk]]
    k = 0
    rng = np.arange(len(eigvals))
    while k < len(ds):
        # find degenerate eigenvalues, get indicies of degenerate eigvals
        inds = np.array(abs(ds - ds[k]) < max(tol, tol * abs(ds[k])))
        inds = rng[inds]
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            eigvecs_array[inds] = degen(tol, eigvecs_array[inds], ops[1:])
        k = max(inds) + 1
    eigvals_out = np.zeros((num_ops, len(ds)), dtype=np.float64)
    kets_out = np.empty((len(ds),), dtype=object)
    kets_out[:] = [
        Qobj(eigvecs_array[j] / la.norm(eigvecs_array[j]),
             dims=[ops[0].dims[0], [1]], shape=[ops[0].shape[0], 1])
        for j in range(len(ds))
    ]
    if not evals:
        return kets_out
    else:
        for kk in range(num_ops):
            for j in range(len(ds)):
                eigvals_out[kk, j] = np.real(np.dot(
                    eigvecs_array[j].conj().T,
                    ops[kk].data * eigvecs_array[j]))
        return eigvals_out, kets_out


def degen(tol, in_vecs, ops):
    """
    Private function that finds eigen vals and vecs for degenerate matrices..
    """
    n = len(ops)
    if n == 0:
        return in_vecs
    A = ops[0]
    vecs = np.column_stack(in_vecs)
    eigvals, eigvecs = la.eig(np.dot(vecs.conj().T, A.data.dot(vecs)))
    zipped = list(zip(-eigvals, range(len(eigvals))))
    zipped.sort()
    ds, perm = zip(*zipped)
    ds = -np.real(np.array(ds))
    perm = np.array(perm)
    vecsperm = np.zeros(eigvecs.shape, dtype=complex)
    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        vecsperm[:, kk] = eigvecs[:, perm[kk]]
    vecs_new = np.dot(vecs, vecsperm)
    vecs_out = np.array(
        [np.zeros((A.shape[0], 1), dtype=complex) for k in range(len(ds))])
    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        vecs_out[kk][:, 0] = vecs_new[:, kk]
    k = 0
    rng = np.arange(len(ds))
    while k < len(ds):
        inds = np.array(abs(ds - ds[k]) < max(
            tol, tol * abs(ds[k])))  # get indicies of degenerate eigvals
        inds = rng[inds]
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            vecs_out[inds] = degen(tol, vecs_out[inds], ops[1:n])
        k = max(inds) + 1
    return vecs_out
