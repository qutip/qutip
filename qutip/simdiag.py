__all__ = ['simdiag']

import numpy as np
import scipy.linalg as la
from . import Qobj
from .core import data as _data


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

    subspace = vecs.conj().T @ ops[i].full() @ vecs
    eigvals, eigvecs = la.eigh(subspace)

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


def simdiag(
    ops,
    evals: bool = True, *,
    tol: float = 1e-14,
    safe_mode: bool = True,
    use_dense_solver: bool = True,
):
    """Simultaneous diagonalization of commuting Hermitian matrices.

    Parameters
    ----------
    ops : list, array
        ``list`` or ``array`` of qobjs representing commuting Hermitian
        operators.

    evals : bool, default: True
        Whether to return the eigenvalues for each ops and eigenvectors or just
        the eigenvectors.

    tol : float, default: 1e-14
        Tolerance for detecting degenerate eigenstates.

    safe_mode : bool, default: True
        Whether to check that all ops are Hermitian and commuting. If set to
        ``False`` and operators are not commuting, the eigenvectors returned
        will often be eigenvectors of only the first operator.

    use_dense_solver: bool, default: True
        Whether to force use of numpy dense eigen solver. When ``False``
        sparse operators will use scipy sparse eigen solver which is not
        appropriate for this use.

    Returns
    -------
    eigs : tuple
        Tuple of arrays representing eigvals and eigvecs of quantum objects
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
            raise TypeError('Matrices must be square.')
        if shape[0] != N:
            raise TypeError('All matrices. must be the same shape')
        if not A.isherm:
            raise TypeError('Matrices must be Hermitian')
        for kk in range(jj):
            B = ops[kk]
            if (A * B - B * A).norm() / (A * B).norm() > tol:
                raise TypeError('Matrices must commute.')

    if use_dense_solver:
        # Do not use sparse eigen solver.
        ops = [op.to("Dense") for op in ops]

    eigvals, eigvecs = _data.eigs(ops[0].data, True, True)
    eigvecs = eigvecs.to_array()

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
        Qobj(eigvecs[:, j], dims=[ops[0].dims[0], [1]])
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
