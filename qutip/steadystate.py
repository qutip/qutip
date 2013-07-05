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
import time
from qutip.qobj import *
from qutip.superoperator import *
from qutip.operators import qeye
from qutip.random_objects import rand_dm
from qutip.sparse import _sp_inf_norm, sp_reshape
from qutip.states import ket2dm
import qutip.settings as qset


def steadystate(A, c_op_list=[], method='direct', sparse=True, use_umfpack=True, 
                maxiter=5000, tol=1e-5, use_precond=True, perm_method='AUTO', verbose=False):
    
    """Calculates the steady state for the evolution subject to the
    supplied Hamiltonian or Lindblad operator and (if given a Hamiltonian) a list of 
    collapse operators.

    If the user passes a Hamiltonian then it, along with the list of collapse operators, 
    will be converted into a Liouvillian operator in Lindblad form.

    
    Parameters
    ----------
    A : qobj
        A Hamiltonian or Lindblad operator.

    c_op_list : list
        A list of collapse operators.
    
    method : str {'direct','iterative','svd','power'}
        Method for solving the underlying linear equation. Direct solver 'direct' (default),
        iterative LGMRES method 'iterative', SVD 'svd' (dense), or inverse-power method 'power'.
    
    sparse : bool default=True
        Solve for the steady state using sparse algorithms.  If set to False, the underlying
        Liouvillian operator will be converted into a dense matrix.  Use only for 'smaller'
        systems.
    
    use_umfpack : bool optional, default = True
        Use the UMFpack backend for the direct solver 'direct'.  If 'False', the solver
        uses the SuperLU backend.  This option does not affect the iterative methods,
        'iterative' and 'power'.  Used only when sparse=True.
    
    maxiter : int optional
        Maximum number of iterations to perform if using an iterative method such
        as 'iterative' (default=5000), or 'power' (default=10).

    tol : float optional, default=1e-5
        Tolerance used for terminating solver solution when using iterative solvers.

    use_precond : bool optional, default = True
        Use an incomplete sparse LU decomposition as a preconditioner for the 'iterative'
        LGMRES solver.  Speeds up convergence time by orders of magnitude.

    perm_method : str {'AUTO', 'COLAMD', 'MMD_ATA', 'NATURAL'}
        Sets the method for column ordering the incomplete LU preconditioner
        used by the 'iterative' method.  When set to 'AUTO' (default), the 
        solver will attempt to precondition the system first using 'COLAMD'.
        If this fails, the other two methods will be tried in the order given
        above.
    
    verbose : bool default=False
        Flag for printing out detailed information on the steady state solver. 
    
    
    Returns
    -------
    dm : qobj
        Steady state density matrix.


    Notes
    -----
    The SVD method works only for dense operators (i.e. small systems).
    
    Setting use_umfpack=True (default) may result in 'out of memory' errors 
    if your system size becomes to large.

    """
    n_op = len(c_op_list)

    if A.type=='oper':
        if n_op == 0:
            raise UserWarning('Cannot calculate the steady state for a ' +
                         'non-dissipative system (no collapse operators given)')
        else:
            A = liouvillian_fast(A, c_op_list)
    if not issuper(A):
        raise UserWarning('Solving for steady states requires Liouvillian (super) operators')
    
    if method=='direct':
        if sparse:
            return _steady_direct_sparse(A, use_umfpack=use_umfpack, verbose=verbose)
        else:
            return _steady_direct_dense(A, verbose=verbose)
    
    elif method=='iterative':
        return _steadystate_iterative(A, tol=tol, use_precond=use_precond, 
                maxiter=maxiter, perm_method=perm_method, verbose=verbose)
    
    elif method=='svd':
        return _steadystate_svd_dense(A, atol=1e-12, rtol=0, 
                all_steadystates=False, verbose=verbose)
    
    elif method=='power':
        return _steady_power(A, maxiter=10, tol=tol, itertol=tol,
                use_umfpack=use_umfpack, verbose=verbose)
    
    else:
        raise ValueError('Invalid method argument for steadystate.')


def steady_nonlinear(L_func, rho0, args={}, maxiter=10,
                     random_initial_state=False,
                     tol=1e-6, itertol=1e-5, use_umfpack=True, verbose=False):
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
                         
    rhoss=0.5*(rhoss+rhoss.dag())
    return rhoss.tidyup() if qset.auto_tidyup else rhoss





def _steady_power(L, maxiter=10, tol=1e-6, itertol=1e-5, use_umfpack=True, verbose=False):
    """
    Inverse power method for steady state solving.
    """
    if verbose:
        print('Starting iterative power method Solver...')
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
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
        start_time=time.time()
    it = 0
    while (la.norm(L * v, np.inf) > tol) and (it < maxiter):
        v = spsolve(L, v, use_umfpack=use_umfpack)
        v = v / la.norm(v, np.inf)
        it += 1
    if it >= maxiter:
        raise UserWarning('Failed to find steady state after ' +
                         str(maxiter) + ' iterations')
    # normalise according to type of problem
    if sflag:
        trow = sp.eye(rhoss.shape[0], rhoss.shape[0], format='coo')
        trow = sp_reshape(trow,(1, n))
        data = v / sum(trow.dot(v))
    else:
        data = data / la.norm(v)
    data = sp_reshape(data, (rhoss.shape[0], rhoss.shape[1])).T
    data = sp.csr_matrix(data)
    rhoss.data = 0.5 * (data + data.conj().T)
    rhoss.isherm = True
    if verbose:
        print('Power solver time: ',time.time()-start_time)
    if qset.auto_tidyup:
        return rhoss.tidyup()
    else:
        return rhoss


def _steady_direct_sparse(L, use_umfpack=True, verbose=False):
    """
    Direct solver that use scipy sparse matrices
    """
    if verbose:
        print('Starting direct solver...')
    n = prod(L.dims[0][0])

    b = sp.csr_matrix(([1.0], ([0], [0])), shape=(n ** 2, 1))
    M = L.data + sp.csr_matrix((np.ones(n), (np.zeros(n), \
            [nn * (n + 1) for nn in range(n)])), shape=(n ** 2, n ** 2))
   
    use_solver(assumeSortedIndices=True, useUmfpack=use_umfpack)
    M.sort_indices()  
    if verbose:
        start_time=time.time()
    v = spsolve(M, b, use_umfpack=use_umfpack)
    if verbose:
        print('Direct solver time: ',time.time()-start_time)
    out=Qobj(vec2mat(v), dims=L.dims[0], isherm=True)
    return 0.5*(out+out.dag())


def _steadystate_iterative(L, tol=1e-5, use_precond=True, maxiter=5000, 
                            perm_method='AUTO', verbose=False):
    """
    Iterative steady state solver using the LGMRES algorithm
    and a sparse incomplete LU preconditioner.
    """
    if verbose:
        print('Starting LGMRES solver...')
    n = prod(L.dims[0][0])
    b = np.zeros(n ** 2)
    b[0] = 1.0
    A = L.data.tocsc() + sp.csc_matrix((np.ones(n), (np.zeros(n), \
            [nn * (n + 1) for nn in range(n)])), shape=(n ** 2, n ** 2))

    if use_precond and perm_method=='AUTO':
        if verbose:
            start_time=time.time()
        try:
            P = spilu(A,drop_tol=1e-1, permc_spec="COLAMD")
            P_x = lambda x: P.solve(x)
            M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
            if verbose:
                print('Preconditioned with COLAMD ordering.')
        except:
            try:
                P = spilu(A,drop_tol=1e-1, permc_spec="MMD_ATA")
                P_x = lambda x: P.solve(x)
                M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
                if verbose:
                    print('Preconditioned with MMD_ATA ordering.')
            except:
                try:
                    P = spilu(A,drop_tol=1e-1, permc_spec="NATURAL")
                    P_x = lambda x: P.solve(x)
                    M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
                    if verbose:
                        print('Preconditioned with NATURAL ordering.')
                except:
                    warnings.warn("Preconditioning failed. Continuing without.",
                          UserWarning)
                    M = None
        if verbose:   
            print('Preconditioning time: ',time.time()-start_time)
    elif use_precond:
        if verbose:
            start_time=time.time()
        try:
            P = spilu(A,drop_tol=1e-1, permc_spec=perm_method)
            P_x = lambda x: P.solve(x)
            M = LinearOperator((n ** 2, n ** 2), matvec=P_x)
        except:
            warnings.warn("Preconditioning failed. Continuing without.",
                  UserWarning)
            M = None
        if verbose:   
            print('Preconditioning time: ',time.time()-start_time)
    else:
        M = None
    if verbose:
        start_time=time.time()
    v, check = lgmres(A, b, tol=tol, M=M, maxiter=maxiter)
    if check>0:
        raise UserWarning("Steadystate solver did not reach tolerance after "+str(check)+" steps.")
    elif check<0:
        raise UserWarning("Steadystate solver failed with fatal error: "+str(check)+".")
    if verbose:   
        print('LGMRES solver time: ',time.time()-start_time)
    out=Qobj(vec2mat(v), dims=L.dims[0],isherm=True)
    return Qobj(0.5*(out+out.dag()),dims=out.dims,shape=out.shape,isherm=True)


def _steady_direct_dense(L, verbose=False):
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
    M[0,:] = np.diag(np.ones(n)).reshape((1, n ** 2))
    if verbose:
        start_time=time.time()
    v = np.linalg.solve(M, b)
    if verbose:   
        print('Direct dense solver time: ',time.time()-start_time)
    out = Qobj(v.reshape(n, n), dims=L.dims[0], isherm=True)
    
    return Qobj(0.5*(out+out.dag()),dims=out.dims,shape=out.shape,isherm=True)


def _steadystate_svd_dense(L, atol=1e-12, rtol=0, all_steadystates=False, verbose=False):
    """
    Find the steady state(s) of an open quantum system by solving for the
    nullspace of the Liouvillian.

    """
    if verbose:
        print('Starting SVD solver...')
        start_time=time.time()
    
    u, s, vh = svd(L.full(), full_matrices=False)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    if verbose:   
        print('SVD solver time: ',time.time()-start_time)
    if all_steadystates:
        rhoss_list = [] 
        for n in range(ns.shape[1]):
            rhoss = Qobj(vec2mat(ns[:,n]), dims=H.dims)
            rhoss_list.append(rhoss / rhoss.tr())
        return rhoss_list
    else:
        rhoss = Qobj(vec2mat(ns[:,0]), dims=H.dims)
        return rhoss / rhoss.tr()


