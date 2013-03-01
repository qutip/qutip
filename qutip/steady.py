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
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""
Module contains functions for iteratively solving for the steady state
density matrix of an open qunatum system defind by a Louvillian.
"""

import numpy as np
from scipy import prod, finfo, randn
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import *
from qutip.qobj import *
from qutip.superoperator import *
from qutip.operators import qeye
from qutip.random_objects import rand_dm
from qutip.sparse import _sp_inf_norm
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

    L = liouvillian(H, c_op_list)
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
    L = L.data.tocsc() - finfo(float).eps * sp.eye(n, n, format='csc')
    L.sort_indices()
    v = mat2vec(rand_dm(rhoss.shape[0], 0.5 / rhoss.shape[0] + 0.5).full())
    # generate sparse iLU preconditioner if requested
    if method == 'bicg' and use_precond:
        try:
            P = spilu(L, permc_spec='MMD_AT_PLUS_A')
            P_x = lambda x: P.solve(x)
        except:
            print("Preconditioning failed.")
            print("Continuing without.")
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

    if qset.auto_tidyup:
        return Qobj(rhoss).tidyup()
    else:
        return Qobj(rhoss)
