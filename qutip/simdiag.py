__all__ = ['simdiag']

import numpy as np
import scipy.linalg as la
from qutip.qobj import Qobj


def _degen(tol, vecs, ops, i=0):
    """
    Private function that finds eigen vals and vecs for degenerate matrices..
    """
    if len(ops) == i:
        return vecs

    # New eigenvectors are sometime not orthogonal.
    for j in range(1, vecs.shape[1]):
        for k in range(j):
            dot = vecs[:, j].dot(vecs[:, k].conj())
            if np.abs(dot) > tol:
                vecs[:, j] = ((vecs[:, j] - dot * vecs[:, k])
                              / (1 - np.abs(dot)**2)**0.5)

    subspace = vecs.conj().T @ ops[i].data @ vecs
    eigvals, eigvecs = la.eig(subspace)

    perm = np.argsort(eigvals)
    eigvals = eigvals[perm]

    vecs_new = vecs @ eigvecs[:, perm]
    for k in range(len(eigvals)):
        vecs_new[:, k] = vecs_new[:, k] / la.norm(vecs_new[:, k])

    k = 0
    while k < len(eigvals):
        ttol = max(tol, tol * abs(eigvals[k]))
        inds, = np.where(abs(eigvals - eigvals[k]) < ttol)
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            vecs_new[:, inds] = _degen(tol, vecs_new[:, inds], ops, i+1)
        k = inds[-1] + 1
    return vecs_new


def simdiag(ops, evals: bool = True, *,
            tol: float = 1e-14, safe_mode: bool = True):
    """Simultaneous diagonalization of commuting Hermitian matrices.

    Parameters
    ----------
    ops : list/array
        ``list`` or ``array`` of qobjs representing commuting Hermitian
        operators.

    evals : bool [True]
        Whether to return the eigenvalues for each ops and eigenvectors or just
        the eigenvectors.

    tol : float [1e-14]
        Tolerance for detecting degenerate eigenstates.

    safe_mode : bool [True]
        Whether to check that all ops are Hermitian and commuting. If set to
        ``False`` and operators are not commuting, the eigenvectors returned
        will often be eigenvectors of only the first operator.

    Returns
    --------
    eigs : tuple
        Tuple of arrays representing eigvecs and eigvals of quantum objects
        corresponding to simultaneous eigenvectors and eigenvalues for each
        operator.

    """
    if not ops:
        raise ValueError("No input matrices.")
    N = ops[0].shape[0]
    num_ops = len(ops) if safe_mode else 0
    for jj in range(num_ops):
        A = ops[jj]
        shape = A.shape
        if shape[0] != shape[1]:
            raise TypeError('Matricies must be square.')
        if shape[0] != N:
            raise TypeError('All matrices. must be the same shape')
        if not A.isherm:
            raise TypeError('Matricies must be Hermitian')
        for kk in range(jj):
            B = ops[kk]
            if (A * B - B * A).norm() / (A * B).norm() > tol:
                raise TypeError('Matricies must commute.')

    eigvals, eigvecs = la.eigh(ops[0].full())
    perm = np.argsort(eigvals)
    eigvecs = eigvecs[:, perm]
    eigvals = eigvals[perm]

    k = 0
    while k < N:
        # find degenerate eigenvalues, get indicies of degenerate eigvals
        ttol = max(tol, tol * abs(eigvals[k]))
        inds, = np.where(abs(eigvals - eigvals[k]) < ttol)
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            eigvecs[:, inds] = _degen(tol, eigvecs[:, inds], ops, 1)
        k = inds[-1] + 1

    for k in range(N):
        eigvecs[:, k] = eigvecs[:, k] / la.norm(eigvecs[:, k])

    kets_out = [
        Qobj(eigvecs[:, j],
             dims=[ops[0].dims[0], [1]], shape=[ops[0].shape[0], 1])
        for j in range(N)
    ]
    eigvals_out = np.zeros((len(ops), N), dtype=np.float64)
    if not evals:
        return kets_out
    else:
        for kk in range(len(ops)):
            for j in range(N):
                eigvals_out[kk, j] = ops[kk].matrix_element(kets_out[j],
                                                            kets_out[j]).real
        return eigvals_out, kets_out
