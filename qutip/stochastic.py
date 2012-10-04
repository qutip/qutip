# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
This module contains experimental functions for solving stochastic schrodinger
and master equations.

Release target: 2.2.0

Todo: 

1) test and debug

2) store measurement records

3) add more sme solvers

4) cythonize some rhs or d1,d2 functions

5) parallelize

"""

import inspect

import numpy as np
import scipy
from scipy.linalg import norm

from qutip.Odedata import Odedata
from qutip.expect import expect
from qutip.Qobj import Qobj
from qutip.superoperator import spre, spost, mat2vec, vec2mat, liouvillian
from qutip.cyQ.cy_mc_funcs import mc_expect, spmv

debug = True

def ssesolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=1, 
             solver='euler-maruyama', method='homodyne', 
             nsubsteps=10, d1=None, d2=None, rhs=None):
    """
    Solve stochastic Schrodinger equation. Dispatch to specific solvers 
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug: print inspect.stack()[0][3]

    if (d1 is None) or (d2 is None):
            
        if method == 'homodyne':
            d1 = d1_psi_homodyne
            d2 = d2_psi_homodyne

        elif method == 'heterodyne':
            d1 = d1_psi_heterodyne
            d2 = d2_psi_heterodyne

        else:
            raise Exception("Unregognized method '%s'." % method)

    if solver == 'euler-maruyama':
        return ssesolve_generic(H, psi0, tlist, c_ops, e_ops, 
                                _rhs_psi_euler_maruyama, d1, d2, ntraj, nsubsteps)

    elif solver == 'platen':
        return ssesolve_generic(H, psi0, tlist, c_ops, e_ops, 
                                _rhs_psi_platen, d1, d2, ntraj, nsubsteps)

    elif solver == 'milstein':
        raise NotImplementedError("Solver '%s' not yet implemented." % solver)

    else:
        raise Exception("Unrecongized solver '%s'." % solver)


def smesolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=1, 
             solver='euler-maruyama', nsubsteps=10):
    """
    Solve stochastic master equation. Dispatch to specific solvers 
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug: print inspect.stack()[0][3]

    if solver == 'euler-maruyama':
        return smesolve_generic(H, psi0, tlist, c_ops, e_ops, 
                                _rhs_rho_euler_maruyama, 
                               d1_rho_homodyne, d2_rho_homodyne, ntraj, nsubsteps)
    else:
        raise Exception("Unrecongized solver '%s'." % solver)



#-------------------------------------------------------------------------------
# Generic parameterized stochastic Schrodinger equation solver
#
def ssesolve_generic(H, psi0, tlist, c_ops, e_ops, rhs, d1, d2, ntraj, nsubsteps):
    """
    internal

    .. note::

        Experimental.

    """
    if debug: print inspect.stack()[0][3]

    N_store = len(tlist)
    N_substeps = nsubsteps
    N = N_store * N_substeps
    dt = (tlist[1]-tlist[0]) / N_substeps
 
    print("N = %d. dt=%.2e" % (N, dt))

    data = Odedata()

    data.expect = np.zeros((len(e_ops), N_store), dtype=complex)
     
    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic Schrodinger equations
    A_ops = []
    for c_idx, c in enumerate(c_ops):
        A_ops.append([c.data, (c+c.dag()).data, (c.dag()*c).data])

    progress_acc = 0.0
    for n in range(ntraj):

        if debug and (100 * float(n)/ntraj) >= progress_acc:
            print("Progress: %.2f" % (100 * float(n)/ntraj))
            progress_acc += 10.0

        psi_t = psi0.full() 

        states_list = _ssesolve_single_trajectory(H, dt, tlist, N_store, N_substeps, psi_t, A_ops, e_ops, data, rhs, d1, d2)

        # if average -> average...
        data.states.append(states_list)

    # average
    data.expect = data.expect/ntraj

    return data


def _ssesolve_single_trajectory(H, dt, tlist, N_store, N_substeps, psi_t, 
                                A_ops, e_ops, data, rhs, d1, d2):
    """
    Internal function. See ssesolve.
    """
    #if debug: print inspect.stack()[0][3]
   
    dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps)

    states_list = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                data.expect[e_idx, t_idx] += expect(e, Qobj(psi_t))
        else:
            states_list.append(Qobj(psi_t))

        for j in range(N_substeps):

            dpsi_t = (-1.0j * dt) * (H.data * psi_t)

            for a_idx, A in enumerate(A_ops):

                dpsi_t += rhs(H.data, psi_t, A, dt, dW[a_idx, t_idx, j], d1, d2)

            # increment and renormalize the wave function
            psi_t += dpsi_t
            psi_t /= norm(psi_t, 2)

    return states_list


#-------------------------------------------------------------------------------
# Generic parameterized stochastic master equation solver
#
def smesolve_generic(H, rho0, tlist, c_ops, e_ops, rhs, d1, d2, ntraj, nsubsteps):
    """
    internal

    .. note::

        Experimental.

    """
    if debug: print inspect.stack()[0][3]

    N_store = len(tlist)
    N_substeps = nsubsteps
    N = N_store * N_substeps
    dt = (tlist[1]-tlist[0]) / N_substeps
 
    print("N = %d. dt=%.2e" % (N, dt))

    data = Odedata()

    data.expect = np.zeros((len(e_ops), N_store), dtype=complex)
     
    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic master equations
    A_ops = []
    for c_idx, c in enumerate(c_ops):

        # xxx: precompute useful operator expressions...
        cdc = c.dag() * c
        Ldt = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)
        LdW = spre(c) + spost(c.dag())
        Lm  = spre(c) + spost(c.dag()) # currently same as LdW

        A_ops.append([Ldt.data, LdW.data, Lm.data])


    # Liouvillian for the unitary part
    L = -1.0j*(spre(H) - spost(H)) # XXX: should we split the ME in stochastic 
                                   # and deterministic collapse operators here?

    progress_acc = 0.0
    for n in range(ntraj):

        if debug and (100 * float(n)/ntraj) >= progress_acc:
            print("Progress: %.2f" % (100 * float(n)/ntraj))
            progress_acc += 10.0

        rho_t = mat2vec(rho0.full())

        states_list = _smesolve_single_trajectory(L, dt, tlist, N_store, N_substeps, 
                                                  rho_t, A_ops, e_ops, data, rhs, d1, d2)

        # if average -> average...
        data.states.append(states_list)

    # average
    data.expect = data.expect/ntraj

    return data


def _smesolve_single_trajectory(L, dt, tlist, N_store, N_substeps, rho_t, 
                                A_ops, e_ops, data, rhs, d1, d2):
    """
    Internal function. See smesolve.
    """
   
    dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps)

    states_list = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                # XXX: need to keep hilbert space structure
                data.expect[e_idx, t_idx] += expect(e, Qobj(vec2mat(rho_t))) 
        else:
            states_list.append(Qobj(rho_t)) # dito

        for j in range(N_substeps):

            drho_t = spmv(L.data.data, L.data.indices, L.data.indptr, rho_t) * dt

            for a_idx, A in enumerate(A_ops):

                drho_t += rhs(L.data, rho_t, A, dt, dW[a_idx, t_idx, j], d1, d2)

            rho_t += drho_t

    return states_list


#-------------------------------------------------------------------------------
# Helper-functions for stochastic DE
#
# d1 = deterministic part of the contribution to the DE RHS function, to be
#      multiplied by the increament dt
#
# d1 = stochastic part of the contribution to the DE RHS function, to be
#      multiplied by the increament dW
#


#
# For SSE
#

# Function sigurature:
#
# def d(A, psi):
#
#     psi = wave function at the current time stemp
#
#     A[0] = c
#     A[1] = c + c.dag()
#     A[2] = c.dag() * c
#
#     where c is a collapse operator. The combinations of c's stored in A are
#     precomputed before the time-evolution is started to avoid repeated 
#     computations.


def d1_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize
    """

    e1 = mc_expect(A[1].data, A[1].indices, A[1].indptr, 0, psi)
    return  0.5 * (e1 * spmv(A[0].data, A[0].indices, A[0].indptr, psi) - 
                        spmv(A[2].data, A[2].indices, A[2].indptr, psi) - 
                   0.25 * e1**2 * psi) 

def d2_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize
    """

    e1 = mc_expect(A[1].data, A[1].indices, A[1].indptr, 0, psi)
    return (spmv(A[0].data, A[0].indices, A[0].indptr, psi) - 0.5 * e1 * psi) 


def d1_psi_heterodyne(A, psi):
    """
    not working/tested
    Todo: cythonize
    """
    e1 = mc_expect(A[0].data, A[0].indices, A[0].indptr, 0, psi)

    B = A[0].T.conj()
    e2 = mc_expect(B.data,    B.indices,    B.indptr,    0, psi)

    return (   e2 * spmv(A[0].data, A[0].indices, A[0].indptr, psi) 
            - 0.5 * spmv(A[2].data, A[2].indices, A[2].indptr, psi) 
            - 0.5 * e1 * e2 * psi) 

def d2_psi_heterodyne(A, psi):
    """
    not working/tested
    Todo: cythonize
    """

    e1 = mc_expect(A[0].data, A[0].indices, A[0].indptr, 0, psi)
    return spmv(A[0].data, A[0].indices, A[0].indptr, psi) - e1 * psi 


def d1_current(A, psi):
    """
    Todo: cythonize, requires poisson increments
    """

    n1 = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi), 2)
    return -0.5 * (spmv(A[2].data, A[2].indices, A[2].indptr, psi) - n1**2 * psi)

def d2_current(A, psi):
    """
    Todo: cythonize, requires poisson increments
    """
    psi_1 = spmv(A[0].data, A[0].indices, A[0].indptr, psi)
    n1 = norm(psi_1, 2)
    return psi_1 / n1 - psi


#
# For SME
#

# def d(A, rho):
#
#     rho = wave function at the current time stemp
#
#     A[0] = Ldt (liouvillian contribution for a collapse operator)
#     A[1] = LdW (stochastic contribution)
#     A[3] = Lm   
#


def _rho_expect(oper, state):
    prod = spmv(oper.data, oper.indices, oper.indptr, state)
    return sum(vec2mat(prod).diagonal())


def d1_rho_homodyne(A, rho):
    """
    not tested
    Todo: cythonize
    """

    return spmv(A[0].data, A[0].indices, A[0].indptr, rho) 

def d2_rho_homodyne(A, rho):
    """
    not tested
    Todo: cythonize
    """

    e1 = _rho_expect(A[2], rho)
    return spmv(A[1].data, A[1].indices, A[1].indptr, rho) - e1 * rho


#-------------------------------------------------------------------------------
# Euler-Maruyama rhs functions for the stochastic Schrodinger and master 
# equations
#

def _rhs_psi_euler_maruyama(H, psi_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """
    
    return d1(A, psi_t) * dt + d2(A, psi_t) * dW


def _rhs_rho_euler_maruyama(L, rho_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """
    
    return d1(A, rho_t) * dt + d2(A, rho_t) * dW

#-------------------------------------------------------------------------------
# Platen method
#

def _rhs_psi_platen(H, psi_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """

    sqrt_dt = np.sqrt(dt)

    dpsi_t_H = (-1.0j * dt) * spmv(H.data, H.indices, H.indptr, psi_t)

    psi_t_1 = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t) * dW
    psi_t_p = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t) * sqrt_dt
    psi_t_m = psi_t + dpsi_t_H + d1(A, psi_t) * dt - d2(A, psi_t) * sqrt_dt

    dpsi_t =  0.50 * (d1(A, psi_t_1) + d1(A, psi_t)) * dt + \
              0.25 * (d2(A, psi_t_p) + d2(A, psi_t_m) + 2 * d2(A, psi_t)) * dW + \
              0.25 * (d2(A, psi_t_p) - d2(A, psi_t_m)) * (dW**2 - dt) * sqrt_dt
    
    return dpsi_t


