# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['simdiag']

import numpy as np
import scipy.linalg as la
from qutip.qobj import Qobj


def simdiag(ops, evals=True):
    """Simulateous diagonalization of communting Hermitian matrices..

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
    zipped = zip(-eigvals, range(len(eigvals)))
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
            eigvecs_array[inds] = degen(
                tol, eigvecs_array[inds],
                np.array([ops[kk] for kk in range(1, num_ops)]))
        k = max(inds) + 1
    eigvals_out = np.zeros((num_ops, len(ds)), dtype=float)
    kets_out = np.array([Qobj(eigvecs_array[j] / la.norm(eigvecs_array[j]),
                              dims=[ops[0].dims[0], [1]],
                              shape=[ops[0].shape[0], 1])
                         for j in range(len(ds))])
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
    zipped = zip(-eigvals, range(len(eigvals)))
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
            vecs_out[inds] = degen(tol, vecs_out[inds],
                                   np.array([ops[jj] for jj in range(1, n)]))
        k = max(inds) + 1
    return vecs_out
