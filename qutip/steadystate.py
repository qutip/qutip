# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""
Module contains functions for solving for the steady state density matrix of
open qunatum systems defined by a Louvillian or Hamiltonian and a list of
collapse operators.
"""

import time
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
from qutip.sparse import _sp_inf_norm, sp_reshape
from qutip.states import ket2dm
import qutip.settings as qset


def steadystate(
    A, c_op_list=[], method='direct', sparse=True, maxiter=5000, tol=1e-5, 
    use_precond=True, M=None, perm_method='AUTO', drop_tol=1e-1, 
    diag_pivot_thresh=0.33, verbose=False):

    """Calculates the steady state for the evolution subject to the
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
        'direct' (default), iterative LGMRES method 'iterative',
        iterative method BICG 'iterative-bicg', LU decomposition 'lu',
        SVD 'svd' (dense), or inverse-power method 'power'.

    sparse : bool default=True
        Solve for the steady state using sparse algorithms. If set to False,
        the underlying Liouvillian operator will be converted into a dense
        matrix. Use only for 'smaller' systems.

    maxiter : int optional
        Maximum number of iterations to perform if using an iterative method
        such as 'iterative' (default=5000), or 'power' (default=10).

    tol : float optional, default=1e-5
        Tolerance used for terminating solver solution when using iterative
        solvers.

    use_precond : bool optional, default = True
        ITERATIVE ONLY. Use an incomplete sparse LU decomposition as a
        preconditioner for the 'iterative' LGMRES and BICG solvers.
        Speeds up convergence time by orders of magnitude in many cases.

    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A. The preconditioner should approximate the inverse of A. 
        Effective preconditioning dramatically improves the rate of convergence, 
        for iterative methods.  Does not affect other solvers.
    
    perm_method : str {'AUTO', 'AUTO-BREAK', 'COLAMD', 'MMD_ATA', 'NATURAL'}
        ITERATIVE ONLY. Sets the method for column ordering the incomplete
        LU preconditioner used by the 'iterative' method.  When set to 'AUTO'
        (default), the solver will attempt to precondition the system using
        'COLAMD'. If this fails, the solver will use no preconditioner.  Using
        'AUTO-BREAK' will cause the solver to issue an exception and stop if
        the 'COLAMD' method fails.

    drop_tol : float default=1e-1
        ITERATIVE ONLY. Sets the threshold for the magnitude of preconditioner
        elements that should be dropped.

    diag_pivot_thresh : float default=0.33
        ITERATIVE ONLY. Sets the threshold for which diagonal elements are
        considered acceptable pivot points when using a preconditioner.

    verbose : bool default=False
        Flag for printing out detailed information on the steady state solver.


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

    if method == 'direct':
        if sparse:
            return _steadystate_direct_sparse(A, verbose=verbose)
        else:
            return _steadystate_direct_dense(A, verbose=verbose)

    elif method == 'iterative':
        return _steadystate_iterative(A, tol=tol, use_precond=use_precond, M=M,
                                      maxiter=maxiter, perm_method=perm_method,
                                      drop_tol=drop_tol, verbose=verbose,
                                      diag_pivot_thresh=diag_pivot_thresh)

    elif method == 'iterative-bicg':
        return _steadystate_iterative_bicg(A, tol=tol, use_precond=use_precond,
                                           M=M, maxiter=maxiter,
                                           perm_method=perm_method,
                                           drop_tol=drop_tol, verbose=verbose,
                                           diag_pivot_thresh=diag_pivot_thresh)

    elif method == 'lu':
        return _steadystate_lu(A, verbose=verbose)

    elif method == 'svd':
        return _steadystate_svd_dense(A, atol=1e-12, rtol=0,
                                      all_steadystates=False, verbose=verbose)

    elif method == 'power':
        return _steadystate_power(A, maxiter=10, tol=tol, itertol=tol,
                                verbose=verbose)

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


def steadystate_nonlinear(L_func, rho0, args={}, maxiter=10,
                          random_initial_state=False, tol=1e-6, itertol=1e-5,
                          use_umfpack=True, verbose=False):
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
    return rhoss.tidyup() if qset.auto_tidyup else rhoss


def _steadystate_direct_sparse(L, verbose=False):
    """
    Direct solver that use scipy sparse matrices
    """
    if verbose:
        print('Starting direct solver...')

    n = prod(L.dims[0][0])
    b = sp.csr_matrix(([1.0], ([0], [0])), shape=(n ** 2, 1), dtype=complex)
    M = L.data + sp.csr_matrix((np.ones(n),
            (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
            shape=(n ** 2, n ** 2))

    use_solver(assumeSortedIndices=True)
    M.sort_indices()

    if verbose:
        start_time = time.time()

    v = spsolve(M, b)

    if verbose:
        print('Direct solver time: ', time.time() - start_time)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=L.dims[0], isherm=True)


def _steadystate_direct_dense(L, verbose=False):
    """
    Direct solver that use numpy dense matrices. Suitable for
    small system, with a few states.
    """
    if verbose:
        print('Starting direct dense solver...')

    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0

    M = L.data.todense()
    M[0, :] = np.diag(np.ones(n)).reshape((1, n ** 2))
    if verbose:
        start_time = time.time()
    v = np.linalg.solve(M, b)

    if verbose:
        print('Direct dense solver time: ', time.time() - start_time)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=L.dims[0], isherm=True)


def _iterative_precondition(A, n, perm_method, drop_tol, diag_pivot_thresh,
                            verbose=False):
    """
    Internal function for preconditioning the steadystate problem for use
    with iterative solvers.
    """

    if verbose:
        start_time = time.time()

    if perm_method in ['AUTO', 'AUTO-BREAK']:
        try:
            P = spilu(
                A, drop_tol=1e-1, permc_spec="COLAMD", diag_pivot_thresh=0.33)
            P_x = lambda x: P.solve(x)
            M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
            if verbose:
                print('Preconditioned with COLAMD ordering.')
                print('Preconditioning time: ', time.time() - start_time)
        except:
            if perm_method == 'AUTO':
                warnings.warn("Preconditioning failed. Continuing without.",
                              UserWarning)
                M = None
            else:
                raise Exception('Preconditioning failed. Halting solver.')

    else:

        if verbose:
            start_time = time.time()

        try:
            P = spilu(A, drop_tol=drop_tol, permc_spec=perm_method,
                      diag_pivot_thresh=diag_pivot_thresh)
            P_x = lambda x: P.solve(x)
            M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
        except:
            warnings.warn("Preconditioning failed. Continuing without.",
                          UserWarning)
            M = None

    if verbose:
        print('Preconditioning time: ', time.time() - start_time)

    return M


def _steadystate_iterative(L, tol=1e-5, use_precond=True, M=None,
                           maxiter=5000, perm_method='AUTO',
                           drop_tol=1e-1, diag_pivot_thresh=0.33,
                           verbose=False):
    """
    Iterative steady state solver using the LGMRES algorithm
    and a sparse incomplete LU preconditioner.
    """

    if verbose:
        print('Starting LGMRES solver...')

    use_solver(assumeSortedIndices=True)
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    A = L.data.tocsc() + sp.csc_matrix((np.ones(n),
            (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
            shape=(n ** 2, n ** 2))
    A.sort_indices()
    
    if use_precond and M is None:
        M = _iterative_precondition(A, n, perm_method, drop_tol,
                                    diag_pivot_thresh, verbose)
    elif use_precond==False and M is None:
        M = None

    if verbose:
        start_time = time.time()

    v, check = lgmres(A, b, tol=tol, M=M, maxiter=maxiter)
    if check > 0:
        raise Exception("Steadystate solver did not reach tolerance after " +
                        str(check) + " steps.")
    elif check < 0:
        raise Exception(
            "Steadystate solver failed with fatal error: " + str(check) + ".")

    if verbose:
        print('LGMRES solver time: ', time.time() - start_time)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=L.dims[0], isherm=True)


def _steadystate_iterative_bicg(L, tol=1e-5, use_precond=True, M=None,
                                maxiter=5000, perm_method='AUTO',
                                drop_tol=1e-1, diag_pivot_thresh=0.33,
                                verbose=False):
    """
    Iterative steady state solver using the BICG algorithm
    and a sparse incomplete LU preconditioner.
    """

    if verbose:
        print('Starting BICG solver...')

    use_solver(assumeSortedIndices=True)
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    A = L.data.tocsc() + sp.csc_matrix((np.ones(n),
            (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
            shape=(n ** 2, n ** 2))
    A.sort_indices()

    if use_precond and M is None:
        M = _iterative_precondition(A, n, perm_method, drop_tol,
                                    diag_pivot_thresh, verbose)
    elif use_precond==False and M is None:
        M = None

    if verbose:
        start_time = time.time()

    v, check = bicgstab(A, b, tol=tol, M=M)

    if check > 0:
        raise Exception("Steadystate solver did not reach tolerance after " +
                        str(check) + " steps.")
    elif check < 0:
        raise Exception(
            "Steadystate solver failed with fatal error: " + str(check) + ".")

    if verbose:
        print('BICG solver time: ', time.time() - start_time)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=L.dims[0], isherm=True)


def _steadystate_lu(L, verbose=False):
    """
    Find the steady state(s) of an open quantum system by computing the
    LU decomposition of the underlying matrix.
    """
    use_solver(assumeSortedIndices=True)
    if verbose:
        print('Starting LU solver...')
        start_time = time.time()
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    A = L.data.tocsc() + sp.csc_matrix((np.ones(n),
            (np.zeros(n), [nn * (n + 1) for nn in range(n)])),
            shape=(n ** 2, n ** 2))

    A.sort_indices()
    solve = factorized(A)
    v = solve(b)
    if verbose:
        print('LU solver time: ', time.time() - start_time)

    data = vec2mat(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=L.dims[0], isherm=True)


def _steadystate_svd_dense(L, atol=1e-12, rtol=0, all_steadystates=False,
                           verbose=False):
    """
    Find the steady state(s) of an open quantum system by solving for the
    nullspace of the Liouvillian.
    """
    if verbose:
        print('Starting SVD solver...')
        start_time = time.time()

    u, s, vh = svd(L.full(), full_matrices=False)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    if verbose:
        print('SVD solver time: ', time.time() - start_time)

    if all_steadystates:
        rhoss_list = []
        for n in range(ns.shape[1]):
            rhoss = Qobj(vec2mat(ns[:, n]), dims=L.dims[0])
            rhoss_list.append(rhoss / rhoss.tr())
        return rhoss_list

    else:
        rhoss = Qobj(vec2mat(ns[:, 0]), dims=L.dims[0])
        return rhoss / rhoss.tr()


def _steadystate_power(L, maxiter=10, tol=1e-6, itertol=1e-5,
                       verbose=False):
    """
    Inverse power method for steady state solving.
    """
    if verbose:
        print('Starting iterative power method Solver...')
    use_solver(assumeSortedIndices=True)
    rhoss = Qobj()
    sflag = issuper(L)
    if sflag:
        rhoss.dims = L.dims[0]
        rhoss.shape = [prod(rhoss.dims[0]), prod(rhoss.dims[1])]
    else:
        rhoss.dims = [L.dims[0], 1]
        rhoss.shape = [prod(rhoss.dims[0]), 1]
    n = prod(rhoss.shape)
    L = L.data.tocsc() - (tol ** 2) * sp.eye(n, n, format='csc')
    L.sort_indices()
    v = mat2vec(rand_dm(rhoss.shape[0], 0.5 / rhoss.shape[0] + 0.5).full())
    if verbose:
        start_time = time.time()
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
    if verbose:
        print('Power solver time: ', time.time() - start_time)
    if qset.auto_tidyup:
        return rhoss.tidyup()
    else:
        return rhoss
