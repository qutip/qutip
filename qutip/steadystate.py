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
Module contains functions for solving for the steady state density matrix of
open quantum systems defined by a Liouvillian or Hamiltonian and a list of
collapse operators.
"""
import warnings

import numpy as np
from numpy.linalg import svd
from scipy import prod, randn
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import *

from qutip.qobj import Qobj, issuper, isoper
from qutip.superoperator import *
from qutip.operators import qeye
from qutip.random_objects import rand_dm
from qutip.sparse import *
from qutip.graph import symrcm
from qutip.states import ket2dm
import qutip.settings as settings

if settings.debug:
    import inspect


def steadystate(A, c_op_list=[], method='direct', sparse=True, use_rcm=True,
                sym=False, use_precond=True, M=None, drop_tol=1e-3,
                fill_factor=12, diag_pivot_thresh=None, maxiter=1000, tol=1e-5,
                use_umfpack=False):

    """Calculates the steady state for quantum evolution subject to the
    supplied Hamiltonian or Liouvillian operator and (if given a Hamiltonian) a
    list of collapse operators.

    If the user passes a Hamiltonian then it, along with the list of collapse
    operators, will be converted into a Liouvillian operator in Lindblad form.


    Parameters
    ----------
    A : qobj
        A Hamiltonian or Liouvillian operator.

    c_op_list : list
        A list of collapse operators.

    method : str {'direct', 'iterative', 'iterative-bicg', 'lu', 'svd', 'power'}
        Method for solving the underlying linear equation. Direct solver
        'direct' (default), iterative GMRES method 'iterative',
        iterative method BICGSTAB 'iterative-bicg', LU decomposition 'lu',
        SVD 'svd' (dense), or inverse-power method 'power'.

    sparse : bool, default = True
        Solve for the steady state using sparse algorithms. If set to False,
        the underlying Liouvillian operator will be converted into a dense
        matrix. Use only for 'smaller' systems.

    use_rcm : bool, default = True
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the
        LU factorization of the Liouvillian.

    maxiter : int, optional
        Maximum number of iterations to perform if using an iterative method
        such as 'iterative' (default=1000), or 'power' (default=10).

    tol : float, optional, default=1e-5
        Tolerance used for terminating solver solution when using iterative
        solvers.

    use_precond : bool optional, default = True
        ITERATIVE ONLY. Use an incomplete sparse LU decomposition as a
        preconditioner for the 'iterative' LGMRES and BICG solvers.
        Speeds up convergence time by orders of magnitude in many cases.

    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A. The preconditioner should approximate the inverse
        of A. Effective preconditioning dramatically improves the rate of
        convergence, for iterative methods.  Does not affect other solvers.

    fill_factor : float, default=12
        ITERATIVE ONLY. Specifies the fill ratio upper bound (>=1) of the iLU
        preconditioner.  Lower values save memory at the cost of longer
        execution times and a possible singular factorization.

    drop_tol : float, default=1e-3
        ITERATIVE ONLY. Sets the threshold for the magnitude of preconditioner
        elements that should be dropped.  Can be reduced for a courser
        factorization at the cost of an increased number of iterations, and a
        possible singular factorization.

    diag_pivot_thresh : float, default=None
        ITERATIVE ONLY. Sets the threshold between [0,1] for which diagonal
        elements are considered acceptable pivot points when using a
        preconditioner.  A value of zero forces the pivot to be the diagonal
        element.

    Returns
    -------
    dm : qobj
        Steady state density matrix.


    Notes
    -----
    The SVD method works only for dense operators (i.e. small systems).

    """
    n_op = len(c_op_list)

    if isoper(A):
        if n_op == 0:
            raise TypeError('Cannot calculate the steady state for a ' +
                            'non-dissipative system ' +
                            '(no collapse operators given)')
        else:
            A = liouvillian_fast(A, c_op_list)
    if not issuper(A):
        raise TypeError('Solving for steady states requires ' +
                        'Liouvillian (super) operators')

    if use_umfpack:
        warnings.warn("The use of use_umfpack is deprecated.")

    if method == 'direct':
        if sparse:
            return _steadystate_direct_sparse(A, use_rcm=use_rcm,
                                              use_umfpack=use_umfpack)
        else:
            return _steadystate_direct_dense(A)

    elif method == 'iterative':
        return _steadystate_iterative(A, tol=tol, use_precond=use_precond, M=M,
                                      use_rcm=use_rcm, sym=sym,
                                      maxiter=maxiter, fill_factor=fill_factor,
                                      drop_tol=drop_tol,
                                      diag_pivot_thresh=diag_pivot_thresh)

    elif method == 'iterative-bicg':
        return _steadystate_iterative_bicg(A, tol=tol, use_precond=use_precond,
                                           M=M, use_rcm=use_rcm,
                                           maxiter=maxiter,
                                           fill_factor=fill_factor,
                                           drop_tol=drop_tol,
                                           diag_pivot_thresh=diag_pivot_thresh)

    elif method == 'lu':
        return _steadystate_lu(A, use_umfpack=use_umfpack)

    elif method == 'svd':
        return _steadystate_svd_dense(A, atol=1e-12, rtol=0,
                                      all_steadystates=False)

    elif method == 'power':
        return _steadystate_power(A, maxiter=10, tol=tol, itertol=tol)

    else:
        raise ValueError('Invalid method argument for steadystate.')


def steady(L, maxiter=10, tol=1e-6, itertol=1e-5, method='solve',
           use_umfpack=True, use_precond=False):
    """
    Deprecated. See steadystate instead.
    """
    message = "steady has been deprecated, use steadystate instead"
    warnings.warn(message, DeprecationWarning)
    return steadystate(L, [], maxiter=maxiter, tol=tol,
                       use_umfpack=use_umfpack, use_precond=use_precond)


def _steadystate_direct_sparse(L, use_rcm=True, use_umfpack=False):
    """
    Direct solver that uses scipy sparse matrices
    """
    if settings.debug:
        print('Starting direct solver...')

    dims = L.dims[0]
    weight = np.abs(L.data.data.max())
    n = prod(L.dims[0][0])
    b = np.zeros((n ** 2, 1), dtype=complex)
    b[0, 0] = weight
    L = L.data + sp.csr_matrix(
        (weight*np.ones(n), (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
        shape=(n ** 2, n ** 2))
    L.sort_indices()
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)

    if use_rcm:
        perm = symrcm(L)
        L = sparse_permute(L, perm, perm)
        b = b[np.ix_(perm,)]

    v = spsolve(L, b)
    if use_rcm:
        rev_perm = np.argsort(perm)
        v = v[np.ix_(rev_perm,)]

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)
    return Qobj(data, dims=dims, isherm=True)


def _steadystate_direct_dense(L):
    """
    Direct solver that use numpy dense matrices. Suitable for
    small system, with a few states.
    """
    if settings.debug:
        print('Starting direct dense solver...')

    dims = L.dims[0]
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0

    L = L.data.todense()
    L[0, :] = np.diag(np.ones(n)).reshape((1, n ** 2))
    v = np.linalg.solve(L, b)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=dims, isherm=True)


def _iterative_precondition(A, n, drop_tol, diag_pivot_thresh, fill_factor):
    """
    Internal function for preconditioning the steadystate problem for use
    with iterative solvers.
    """
    try:
        P = spilu(A, drop_tol=drop_tol, diag_pivot_thresh=diag_pivot_thresh,
                  fill_factor=fill_factor, options=dict(ILU_MILU='SMILU_3'))

        P_x = lambda x: P.solve(x)
        M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
    except:
        warnings.warn("Preconditioning failed. Continuing without.",
                      UserWarning)
        M = None

    return M


def _steadystate_iterative(L, tol=1e-5, use_precond=True, M=None,
                           use_rcm=True, sym=False, fill_factor=12,
                           maxiter=1000, drop_tol=1e-3, diag_pivot_thresh=None,
                           use_umfpack=False):
    """
    Iterative steady state solver using the LGMRES algorithm
    and a sparse incomplete LU preconditioner.
    """

    if settings.debug:
        print('Starting GMRES solver...')

    dims = L.dims[0]
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    L = L.data.tocsc() + sp.csc_matrix(
        (1e-1 * np.ones(n), (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
        shape=(n ** 2, n ** 2))

    if use_rcm:
        if settings.debug:
            print('Original bandwidth ', sparse_bandwidth(L))
        perm = symrcm(L)
        rev_perm = np.argsort(perm)
        L = sparse_permute(L, perm, perm, 'csc')
        b = b[np.ix_(perm,)]
        if settings.debug:
            print('RCM bandwidth ', sparse_bandwidth(L))

    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    L.sort_indices()

    if M is None and use_precond:
        M = _iterative_precondition(L, n, drop_tol, diag_pivot_thresh,
                                    fill_factor)

    v, check = gmres(L, b, tol=tol, M=M, maxiter=maxiter)
    if check > 0:
        raise Exception("Steadystate solver did not reach tolerance after " +
                        str(check) + " steps.")
    elif check < 0:
        raise Exception(
            "Steadystate solver failed with fatal error: " + str(check) + ".")

    if use_rcm:
        v = v[np.ix_(rev_perm,)]

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=dims, isherm=True)


def _steadystate_iterative_bicg(L, tol=1e-5, use_precond=True, use_rcm=True,
                                M=None, maxiter=1000, drop_tol=1e-3,
                                diag_pivot_thresh=None, fill_factor=12,
                                use_umfpack=False):
    """
    Iterative steady state solver using the BICG algorithm
    and a sparse incomplete LU preconditioner.
    """

    if settings.debug:
        print('Starting BICG solver...')

    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    dims = L.dims[0]
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    L = L.data.tocsc() + sp.csc_matrix(
        (np.ones(n), (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
        shape=(n ** 2, n ** 2))
    L.sort_indices()

    if use_rcm:
        if settings.debug:
            print('Original bandwidth ', sparse_bandwidth(L))
        perm = symrcm(L)
        rev_perm = np.argsort(perm)
        L = sparse_permute(L, perm, perm, 'csc')
        b = b[np.ix_(perm,)]
        if settings.debug:
            print('RCM bandwidth ', sparse_bandwidth(L))

    if M is None and use_precond:
        M = _iterative_precondition(L, n, drop_tol,
                                    diag_pivot_thresh, fill_factor)

    v, check = bicgstab(L, b, tol=tol, M=M)

    if use_rcm:
        v = v[np.ix_(rev_perm,)]

    if check > 0:
        raise Exception("Steadystate solver did not reach tolerance after " +
                        str(check) + " steps.")
    elif check < 0:
        raise Exception(
            "Steadystate solver failed with fatal error: " + str(check) + ".")

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)
    return Qobj(data, dims=dims, isherm=True)


def _steadystate_lu(L, use_rcm=True, use_umfpack=False):
    """
    Find the steady state(s) of an open quantum system by computing the
    LU decomposition of the underlying matrix.
    """
    if settings.debug:
        print('Starting LU solver...')

    dims = L.dims[0]
    weight = np.abs(L.data.data.max())
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2, dtype=complex)
    b[0] = weight
    L = L.data.tocsc() + sp.csc_matrix(
        (weight * np.ones(n),
         (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
        shape=(n ** 2, n ** 2))

    L.sort_indices()
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    if use_rcm:
        perm = symrcm(L)
        L = sparse_permute(L, perm, perm)
        b = b[np.ix_(perm,)]

    solve = factorized(L)
    v = solve(b)
    if use_rcm:
        rev_perm = np.argsort(perm)
        v = v[np.ix_(rev_perm,)]
    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=dims, isherm=True)


def _steadystate_svd_dense(L, atol=1e-12, rtol=0, all_steadystates=False):
    """
    Find the steady state(s) of an open quantum system by solving for the
    nullspace of the Liouvillian.
    """
    if settings.debug:
        print('Starting SVD solver...')

    u, s, vh = svd(L.full(), full_matrices=False)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    if all_steadystates:
        rhoss_list = []
        for n in range(ns.shape[1]):
            rhoss = Qobj(vec2mat(ns[:, n]), dims=L.dims[0])
            rhoss_list.append(rhoss / rhoss.tr())
        return rhoss_list

    else:
        rhoss = Qobj(vec2mat(ns[:, 0]), dims=L.dims[0])
        return rhoss / rhoss.tr()


def _steadystate_power(L, maxiter=10, tol=1e-6, itertol=1e-5):
    """
    Inverse power method for steady state solving.
    """
    if settings.debug:
        print('Starting iterative power method Solver...')

    use_solver(assumeSortedIndices=True)
    rhoss = Qobj()
    sflag = issuper(L)
    if sflag:
        rhoss.dims = L.dims[0]
    else:
        rhoss.dims = [L.dims[0], 1]
    n = prod(rhoss.shape)
    L = L.data.tocsc() - (tol ** 2) * sp.eye(n, n, format='csc')
    L.sort_indices()
    v = mat2vec(rand_dm(rhoss.shape[0], 0.5 / rhoss.shape[0] + 0.5).full())

    it = 0
    while (la.norm(L * v, np.inf) > tol) and (it < maxiter):
        v = spsolve(L, v)
        v = v / la.norm(v, np.inf)
        it += 1
    if it >= maxiter:
        raise Exception('Failed to find steady state after ' +
                        str(maxiter) + ' iterations')
    # normalise according to type of problem
    if sflag:
        trow = sp.eye(rhoss.shape[0], rhoss.shape[0], format='coo')
        trow = sp_reshape(trow, (1, n))
        data = v / sum(trow.dot(v))
    else:
        data = data / la.norm(v)

    data = sp.csr_matrix(vec2mat(data))
    rhoss.data = 0.5 * (data + data.conj().T)
    rhoss.isherm = True

    return rhoss.tidyup() if settings.auto_tidyup else rhoss


def steadystate_nonlinear(L_func, rho0, args={}, maxiter=10,
                          random_initial_state=False, tol=1e-6, itertol=1e-5,
                          use_umfpack=True):
    """
    Steady state for the evolution subject to the nonlinear Liouvillian
    (which depends on the density matrix).

    .. note:: Experimental. Not at all certain that the inverse power method
              works for state-dependent Liouvillian operators.
    """
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    if random_initial_state:
        rhoss = rand_dm(rho0.shape[0], 1.0, dims=rho0.dims)
    elif isket(rho0):
        rhoss = ket2dm(rho0)
    else:
        rhoss = Qobj(rho0)

    v = mat2vec(rhoss.full())

    n = prod(rhoss.shape)
    tr_vec = sp.eye(rhoss.shape[0], rhoss.shape[0], format='coo')
    tr_vec = tr_vec.reshape((1, n))

    it = 0
    while it < maxiter:

        L = L_func(rhoss, args)
        L = L.data.tocsc() - (tol ** 2) * sp.eye(n, n, format='csc')
        L.sort_indices()

        v = spsolve(L, v, use_umfpack=use_umfpack)
        v = v / la.norm(v, np.inf)

        data = v / sum(tr_vec.dot(v))
        data = reshape(data, (rhoss.shape[0], rhoss.shape[1])).T
        rhoss.data = sp.csr_matrix(data)

        it += 1

        if la.norm(L * v, np.inf) <= tol:
            break

    if it >= maxiter:
        raise ValueError('Failed to find steady state after ' +
                         str(maxiter) + ' iterations')

    rhoss = 0.5 * (rhoss + rhoss.dag())
    return rhoss.tidyup() if settings.auto_tidyup else rhoss
