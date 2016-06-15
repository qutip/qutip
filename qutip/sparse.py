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
"""
This module contains a collection of routines for operating on sparse
matrices on the scipy.sparse formats, for use internally by other modules
throughout QuTiP.
"""

__all__ = ['sp_fro_norm', 'sp_inf_norm', 'sp_L2_norm', 'sp_max_norm',
           'sp_one_norm', 'sp_reshape', 'sp_eigs', 'sp_expm', 'sp_permute',
           'sp_reverse_permute', 'sp_bandwidth', 'sp_profile']

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
from scipy.linalg.blas import get_blas_funcs
_dznrm2 = get_blas_funcs("znrm2")
from qutip.cy.sparse_utils import (_sparse_profile, _sparse_permute,
                                   _sparse_reverse_permute, _sparse_bandwidth,
                                   _isdiag)
from qutip.settings import debug

import qutip.logging_utils
logger = qutip.logging_utils.get_logger()

if debug:
    import inspect


def sp_fro_norm(data):
    """
    Frobius norm for sparse matrix
    """
    out = np.sum(np.abs(data.data)**2)
    return np.sqrt(out)


def sp_inf_norm(data):
    """
    Infinity norm for sparse matrix
    """
    return np.max([np.sum(np.abs(data.getrow(k).data))
                   for k in range(data.shape[0])])


def sp_L2_norm(data):
    """
    L2 norm sparse vector
    """
    if 1 not in data.shape:
        raise TypeError("Use L2-norm only for vectors.")

    if len(data.data):
        return _dznrm2(data.data)
    else:
        return 0


def sp_max_norm(data):
    """
    Max norm for sparse matrix
    """
    return np.max(np.abs(data.data)) if any(data.data) else 0


def sp_one_norm(data):
    """
    One norm for sparse matrix
    """
    return np.max(np.array([np.sum(np.abs((data.getcol(k).data)))
                            for k in range(data.shape[1])]))


def sp_reshape(A, shape, format='csr'):
    """
    Reshapes a sparse matrix.

    Parameters
    ----------
    A : sparse_matrix
        Input matrix in any format
    shape : list/tuple
        Desired shape of new matrix
    format : string {'csr','coo','csc','lil'}
        Optional string indicating desired output format

    Returns
    -------
    B : csr_matrix
        Reshaped sparse matrix

    References
    ----------

        http://stackoverflow.com/questions/16511879/reshape-sparse-matrix-efficiently-python-scipy-0-12

    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('Shape must be a list of two integers')

    C = A.tocoo()
    nrows, ncols = C.shape
    size = nrows * ncols
    new_size = shape[0] * shape[1]

    if new_size != size:
        raise ValueError('Total size of new array must be unchanged.')

    flat_indices = ncols * C.row + C.col
    new_row, new_col = divmod(flat_indices, shape[1])
    B = sp.coo_matrix((C.data, (new_row, new_col)), shape=shape)

    if format == 'csr':
        return B.tocsr()
    elif format == 'coo':
        return B
    elif format == 'csc':
        return B.tocsc()
    elif format == 'lil':
        return B.tolil()
    else:
        raise ValueError('Return format not valid.')


def _dense_eigs(data, isherm, vecs, N, eigvals, num_large, num_small):
    """
    Internal functions for computing eigenvalues and eigenstates for a dense
    matrix.
    """
    if debug:
        logger.debug(inspect.stack()[0][3] + ": vectors = " + str(vecs))

    evecs = None

    if vecs:
        if isherm:
            if eigvals == 0:
                evals, evecs = la.eigh(data)
            else:
                if num_small > 0:
                    evals, evecs = la.eigh(
                        data, eigvals=[0, num_small - 1])
                if num_large > 0:
                    evals, evecs = la.eigh(
                        data, eigvals=[N - num_large, N - 1])
        else:
            evals, evecs = la.eig(data)
    else:
        if isherm:
            if eigvals == 0:
                evals = la.eigvalsh(data)
            else:
                if num_small > 0:
                    evals = la.eigvalsh(data, eigvals=[0, num_small - 1])
                if num_large > 0:
                    evals = la.eigvalsh(data, eigvals=[N - num_large, N - 1])
        else:
            evals = la.eigvals(data)

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
    if debug:
        print(inspect.stack()[0][3] + ": vectors = " + str(vecs))

    big_vals = np.array([])
    small_vals = np.array([])
    evecs = None

    remove_one = False
    if eigvals == (N - 1):
        # calculate all eigenvalues and remove one at output if using sparse
        eigvals = 0
        num_small = int(np.ceil(N / 2.0))
        num_large = N - num_small
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


def sp_eigs(data, isherm, vecs=True, sparse=False, sort='low',
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

    if debug:
        print(inspect.stack()[0][3])

    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")

    N = data.shape[0]
    if eigvals == N:
        eigvals = 0

    if eigvals > N:
        raise ValueError("Number of requested eigen vals/vecs must be <= N.")

    # set number of large and small eigenvals/vecs
    if eigvals == 0:  # user wants all eigs (default)
        D = int(np.ceil(N / 2.0))
        num_large = N - D
        if not np.mod(N, 2):
            M = D
        else:
            M = D - 1
        num_small = N - M
    else:  # if user wants only a few eigen vals/vecs
        if sort == 'low':
            num_small = eigvals
            num_large = 0
        elif sort == 'high':
            num_large = eigvals
            num_small = 0
        else:
            raise ValueError("Invalid option for 'sort'.")

    # Dispatch to sparse/dense solvers
    if sparse:
        evals, evecs = _sp_eigs(data, isherm, vecs, N, eigvals, num_large,
                                num_small, tol, maxiter)
    else:
        evals, evecs = _dense_eigs(data.todense(), isherm, vecs, N, eigvals,
                                   num_large, num_small)

    if sort == 'high':  # flip arrays to largest values first
        if vecs:
            evecs = np.flipud(evecs)
        evals = np.flipud(evals)

    return (evals, evecs) if vecs else evals


def sp_expm(A, sparse=False):
    """
    Sparse matrix exponential.    
    """
    if _isdiag(A.indices, A.indptr, A.shape[0]):
        A = sp.diags(np.exp(A.diagonal()), shape=A.shape, 
                    format='csr', dtype=complex)
        return A
    if sparse:
        E = spla.expm(A.tocsc())
    else:
        E = spla.expm(A.toarray())
    return sp.csr_matrix(E)
    


def sp_permute(A, rperm=(), cperm=(), safe=True):
    """
    Permutes the rows and columns of a sparse CSR/CSC matrix
    according to the permutation arrays rperm and cperm, respectively.
    Here, the permutation arrays specify the new order of the rows and
    columns. i.e. [0,1,2,3,4] -> [3,0,4,1,2].

    Parameters
    ----------
    A : csr_matrix, csc_matrix
        Input matrix.
    rperm : array_like of integers
        Array of row permutations.
    cperm : array_like of integers
        Array of column permutations.
    safe : bool
        Check structure of permutation arrays.

    Returns
    -------
    perm_csr : csr_matrix, csc_matrix
        CSR or CSC matrix with permuted rows/columns.

    """
    rperm = np.asarray(rperm, dtype=np.int32)
    cperm = np.asarray(cperm, dtype=np.int32)
    nrows = A.shape[0]
    ncols = A.shape[1]
    if len(rperm) == 0:
        rperm = np.arange(nrows, dtype=np.int32)
    if len(cperm) == 0:
        cperm = np.arange(ncols, dtype=np.int32)
    if safe:
        if len(np.setdiff1d(rperm, np.arange(nrows))) != 0:
            raise Exception('Invalid row permutation array.')
        if len(np.setdiff1d(cperm, np.arange(ncols))) != 0:
            raise Exception('Invalid column permutation array.')

    shp = A.shape
    kind = A.getformat()
    if kind == 'csr':
        flag = 0
    elif kind == 'csc':
        flag = 1
    else:
        raise Exception('Input must be Qobj, CSR, or CSC matrix.')

    data, ind, ptr = _sparse_permute(A.data, A.indices, A.indptr,
                                     nrows, ncols, rperm, cperm, flag)
    if kind == 'csr':
        return sp.csr_matrix((data, ind, ptr), shape=shp, dtype=data.dtype)
    elif kind == 'csc':
        return sp.csc_matrix((data, ind, ptr), shape=shp, dtype=data.dtype)


def sp_reverse_permute(A, rperm=(), cperm=(), safe=True):
    """
    Performs a reverse permutations of the rows and columns of a sparse CSR/CSC
    matrix according to the permutation arrays rperm and cperm, respectively.
    Here, the permutation arrays specify the order of the rows and columns used
    to permute the original array.

    Parameters
    ----------
    A : csr_matrix, csc_matrix
        Input matrix.
    rperm : array_like of integers
        Array of row permutations.
    cperm : array_like of integers
        Array of column permutations.
    safe : bool
        Check structure of permutation arrays.

    Returns
    -------
    perm_csr : csr_matrix, csc_matrix
        CSR or CSC matrix with permuted rows/columns.

    """
    rperm = np.asarray(rperm, dtype=np.int32)
    cperm = np.asarray(cperm, dtype=np.int32)
    nrows = A.shape[0]
    ncols = A.shape[1]
    if len(rperm) == 0:
        rperm = np.arange(nrows, dtype=np.int32)
    if len(cperm) == 0:
        cperm = np.arange(ncols, dtype=np.int32)
    if safe:
        if len(np.setdiff1d(rperm, np.arange(nrows))) != 0:
            raise Exception('Invalid row permutation array.')
        if len(np.setdiff1d(cperm, np.arange(ncols))) != 0:
            raise Exception('Invalid column permutation array.')

    shp = A.shape
    kind = A.getformat()
    if kind == 'csr':
        flag = 0
    elif kind == 'csc':
        flag = 1
    else:
        raise Exception('Input must be Qobj, CSR, or CSC matrix.')

    data, ind, ptr = _sparse_reverse_permute(A.data, A.indices, A.indptr,
                                             nrows, ncols, rperm, cperm, flag)

    if kind == 'csr':
        return sp.csr_matrix((data, ind, ptr), shape=shp, dtype=data.dtype)
    elif kind == 'csc':
        return sp.csc_matrix((data, ind, ptr), shape=shp, dtype=data.dtype)


def sp_bandwidth(A):
    """
    Returns the max(mb), lower(lb), and upper(ub) bandwidths of a
    sparse CSR/CSC matrix.

    If the matrix is symmetric then the upper and lower bandwidths are
    identical. Diagonal matrices have a bandwidth equal to one.

    Parameters
    ----------
    A : csr_matrix, csc_matrix
        Input matrix

    Returns
    -------
    mb : int
        Maximum bandwidth of matrix.
    lb : int
        Lower bandwidth of matrix.
    ub : int
        Upper bandwidth of matrix.

    """
    nrows = A.shape[0]
    ncols = A.shape[1]

    if A.getformat() == 'csr':
        return _sparse_bandwidth(A.indices, A.indptr, nrows)
    elif A.getformat() == 'csc':
        # Normal output is mb,lb,ub but since CSC
        # is transpose of CSR switch lb and ub
        mb, ub, lb = _sparse_bandwidth(A.indices, A.indptr, ncols)
        return mb, lb, ub
    else:
        raise Exception('Invalid sparse input format.')


def sp_profile(A):
    """Returns the total, lower, and upper profiles of a sparse matrix.

    If the matrix is symmetric then the upper and lower profiles are
    identical. Diagonal matrices have zero profile.

    Parameters
    ----------
    A : csr_matrix, csc_matrix
        Input matrix
    """
    if sp.isspmatrix_csr(A):
        up = _sparse_profile(A.indices, A.indptr, A.shape[0])
        A = A.tocsc()
        lp = _sparse_profile(A.indices, A.indptr, A.shape[0])

    elif sp.isspmatrix_csc(A):
        lp = _sparse_profile(A.indices, A.indptr, A.shape[0])
        A = A.tocsr()
        up = _sparse_profile(A.indices, A.indptr, A.shape[0])

    else:
        raise TypeError('Input sparse matrix must be in CSR or CSC format.')

    return up+lp, lp, up


def sp_isdiag(A):
    """Determine if sparse CSR matrix is diagonal.
    
    Parameters
    ----------
    A : csr_matrix, csc_matrix
        Input matrix
        
    Returns
    -------
    isdiag : int
        True if matix is diagonal, False otherwise.
    
    """
    if not sp.isspmatrix_csr(A):
        raise TypeError('Input sparse matrix must be in CSR format.')
    return _isdiag(A.indices, A.indptr, A.shape[0])
