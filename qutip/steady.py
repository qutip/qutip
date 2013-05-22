# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
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
open qunatum systems defined by a Louvillian or Hamiltonian and list of 
collapse operators.
"""

import warnings
import numpy as np
from numpy.linalg import svd
from scipy import prod, randn
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import *

from qutip.qobj import *
from qutip.superoperator import *
from qutip.operators import qeye
from qutip.random_objects import rand_dm
from qutip.sparse import _sp_inf_norm
from qutip.states import ket2dm
import qutip.settings as qset


def steadystate(H, c_op_list, maxiter=10, tol=1e-6, itertol=1e-5,
                method='solve', use_umfpack=True, use_precond=False):
    """Calculates the steady state for the evolution subject to the
    supplied Hamiltonian and list of collapse operators.

    This function builds the Louvillian from the Hamiltonaian and
    calls the :func:`qutip.steady.steady` function.

    Parameters
    ----------
    H : qobj
        Hamiltonian operator.

    c_op_list : list
        A ``list`` of collapse operators.

    maxiter : int
        Maximum number of iterations to perform, default = 100.

    tol : float
        Tolerance used for terminating solver solution, default = 1e-6.

    itertol : float
        Tolerance used for iterative Ax=b solver, default = 1e-5.

    method : str
        Method for solving linear equations. Direct solver 'solve' (default) or
        iterative biconjugate gradient method 'bicg'.

    use_umfpack: bool, default = True
        Use the UMFpack backend for the direct solver.  If 'False', the solver
        uses the SuperLU backend.  This option does not affect the 'bicg'
        method.

    use_precond: bool, default = False
        Use an incomplete sparse LU decomposition as a preconditioner for the
        stabilized bi-conjugate gradient 'bicg' method.

    Returns
    -------
    ket : qobj
        Ket vector for steady state.

    Notes
    -----
    Uses the inverse power method.
    See any Linear Algebra book with an iterative methods section.
    Using UMFpack may result in 'out of memory' errors for some
    Liouvillians.

    """
    n_op = len(c_op_list)

    if n_op == 0:
        raise ValueError('Cannot calculate the steady state for a ' +
                         'nondissipative system (no collapse operators given)')

    L = liouvillian_fast(H, c_op_list)
    return steady(L, maxiter=maxiter, tol=tol, itertol=itertol,
                  method=method, use_umfpack=use_umfpack,
                  use_precond=use_precond)


def steady(L, maxiter=10, tol=1e-6, itertol=1e-5, method='solve',
           use_umfpack=True, use_precond=False):
    """Steady state for the evolution subject to the
    supplied Louvillian.

    Parameters
    ----------
    L : qobj
        Liouvillian superoperator.

    maxiter : int
        Maximum number of iterations to perform, default = 100.

    tol : float
        Tolerance used for terminating solver solution, default = 1e-6.

    itertol : float
        Tolerance used for iterative Ax=b solver, default = 1e-5.

    method : str
        Method for solving linear equations. Direct solver 'solve' (default) or
        iterative biconjugate gradient method 'bicg'.

    use_umfpack: bool {True, False}
        Use the UMFpack backend for the direct solver.  If 'False', the solver
        uses the SuperLU backend.  This option does not affect the 'bicg'
        method.

    use_precond: bool {False, True}
        Use an incomplete sparse LU decomposition as a preconditioner for the
        stabilized bi-conjugate gradient 'bicg' method.

    Returns
    --------
    ket : qobj
        Ket vector for steady state.

    Notes
    -----
    Uses the inverse power method.
    See any Linear Algebra book with an iterative methods section.
    Using UMFpack may result in 'out of memory' errors for some
    Liouvillians.

    """
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    if (not isoper(L)) and (not issuper(L)):
        raise TypeError('Steady states can only be found for operators ' +
                        'or superoperators.')
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
    # generate sparse iLU preconditioner if requested
    if method == 'bicg' and use_precond:
        try:
            P = spilu(L, permc_spec='MMD_AT_PLUS_A')
            P_x = lambda x: P.solve(x)
        except:
            warnings.warn("Preconditioning failed. Continuing without.",
                          UserWarning)
            M = None
        else:
            M = LinearOperator((n, n), matvec=P_x)
    else:
        M = None
    it = 0
    while (la.norm(L * v, np.inf) > tol) and (it < maxiter):
        if method == 'bicg':
            v, check = bicgstab(L, v, tol=itertol, M=M)
        else:
            v = spsolve(L, v, permc_spec="MMD_AT_PLUS_A",
                        use_umfpack=use_umfpack)
        v = v / la.norm(v, np.inf)
        it += 1
    if it >= maxiter:
        raise ValueError('Failed to find steady state after ' +
                         str(maxiter) + ' iterations')
    # normalise according to type of problem
    if sflag:
        trow = sp.eye(rhoss.shape[0], rhoss.shape[0], format='lil')
        trow = trow.reshape((1, n)).tocsr()
        data = v / sum(trow.dot(v))
    else:
        data = data / la.norm(v)
    data = reshape(data, (rhoss.shape[0], rhoss.shape[1])).T
    data = sp.csr_matrix(data)
    rhoss.data = 0.5 * (data + data.conj().T)
    rhoss.isherm = True
    if qset.auto_tidyup:
        return rhoss.tidyup()
    else:
        return rhoss


def steady_nonlinear(L_func, rho0, args={}, maxiter=10,
                     random_initial_state=False,
                     tol=1e-6, itertol=1e-5, use_umfpack=True):
    """
    Steady state for the evolution subject to the nonlinear Liouvillian
    (which depends on the density matrix).

    .. note:: Experimental. Not at all certain that the inverse power method
              works for state-dependent liouvillian operators.
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
    tr_vec = sp.eye(rhoss.shape[0], rhoss.shape[0], format='lil')
    tr_vec = tr_vec.reshape((1, n)).tocsr()

    it = 0
    while it < maxiter:

        L = L_func(rhoss, args)
        L = L.data.tocsc() - (tol ** 2) * sp.eye(n, n, format='csc')
        L.sort_indices()

        v = spsolve(L, v, permc_spec="MMD_AT_PLUS_A", use_umfpack=use_umfpack)
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

    return rhoss.tidyup() if qset.auto_tidyup else rhoss


def steadystate_direct(H, c_ops, sparse=True, use_umfpack=False):
    """
    Simple steady state solver that use a direct solve method.

    .. note:: Experimental.
    """
    L = liouvillian_fast(H, c_ops)
    if sparse:
        return steady_direct_sparse(L, use_umfpack=use_umfpack)
    else:
        return steady_direct_dense(L)


def steady_direct(L, sparse=True, use_umfpack=False):
    """
    Simple steady state solver that use a direct solve method.

    .. note:: Experimental.
    """
    if sparse:
        return steady_direct_sparse(L, use_umfpack=use_umfpack)
    else:
        return steady_direct_dense(L)


def steady_direct_sparse(L, use_umfpack=False):
    """
    Direct solver that use scipy sparse matrices

    .. note:: Experimental.
    """

    n = prod(L.dims[0][0])
    b = sp.csc_matrix(([1.0], ([0], [0])), shape=(n ** 2, 1))
    #M = L.data.tocsc() + sp.eye(n, n, format='lil').reshape((n ** 2, n ** 2)).tocsc()
    M = L.data.tocsc() + sp.csc_matrix((np.ones(n), (np.zeros(n), [nn * (n + 1) for nn in range(n)])), shape=(n ** 2, n ** 2))    
    v = spsolve(M, b, permc_spec="MMD_AT_PLUS_A", use_umfpack=use_umfpack)
    
    return Qobj(vec2mat(v), dims=L.dims[0], isherm=True)


def steady_direct_dense(L):
    """
    Direct solver that use numpy dense matrices. Suitable for 
    small system, with a few states.

    .. note:: Experimental.
    """
    
    n = prod(L.dims[0][0])
    
    b = np.zeros(n ** 2)
    b[0] = 1.0

    M = L.data.todense()
    M[0,:] = np.diag(np.ones(n)).reshape((1, n ** 2))
    
    v = np.linalg.solve(M, b)
    
    return Qobj(v.reshape(n, n), dims=L.dims[0], isherm=True)


def steadystate_svd_dense(H, c_ops, atol=1e-12, rtol=0,
                          all_steadystates=False):
    """
    Find the steadystate(s) of an open quantum system by solving for the
    nullspace of the Liouvillian.

    .. note:: Experimental.
    """
    
    L = liouvillian_fast(H, c_ops)
    
    u, s, vh = svd(L.full(), full_matrices=False)

    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    
    if all_steadystates:
        rhoss_list = [] 
        for n in range(ns.shape[1]):
            rhoss = Qobj(vec2mat(ns[:,n]), dims=H.dims)
            rhoss_list.append(rhoss / rhoss.tr())
        return rhoss_list
    else:
        rhoss = Qobj(vec2mat(ns[:,0]), dims=H.dims)
        return rhoss / rhoss.tr()


