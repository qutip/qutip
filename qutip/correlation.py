#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from scipy import *
from Qobj import *
from superoperator import *
from odesolve import *
from essolve import *
from mcsolve import *
from steady import *

def correlation_ss_es(H, tlist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(\\tau)B(0)\\right>`
    using the quantum regression theorem, and exponential series.
 
    Args:
    
        H (Qobj): system Hamiltonian.
        
        rho0 (Qobj): initial density matrix.
        
        tlist (*list/array*): list of times for :math:`t`.
        
        c_op_list (list of Qobj's): list of collapse operators.
        
        a_op (Qobj): for A operator.
        
        b_op (Qobj): for B operator.
    
    Returns: 
        
        array of expectation values.
    """

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    # find the steady state
    rho0 = steady(L)

    # evaluate the correlation function
    solC_tau = ode2es(L, b_op * rho0)

    return esval(expect(a_op, solC_tau), tlist)

def correlation_es(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(t+\\tau)B(t)\\right>`
    using exponential series and the quantum regression theorem.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `rho0` (:class:`qutip.Qobj`) initial density matrix.
        
        `tlist` (*list/array*) list of times for :math:`t`.

        `taulist` (*list/array*) list of times for :math:`\\tau`.
        
        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
                
    Returns *array* of expectation values.
    """

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    if rho0 == None:
        rho0 = steady(L)

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    solES_t = ode2es(L, rho0)

    for t_idx in range(len(tlist)):

        rho_t = esval_op(solES_t, [tlist[t_idx]])

        solES_tau = ode2es(L, b_op * rho_t)

        C_mat[t_idx, :] = esval(expect(a_op, solES_tau), taulist)
   
    return C_mat


def correlation_ss_ode(H, tlist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(\\tau)B(0)\\right>`
    using the quantum regression theorem, and and the ode solver.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `rho0` (:class:`qutip.Qobj`) initial density matrix.
        
        `tlist` (*list/array*) list of times for :math:`t`.
        
        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
    
    Returns *array* of expectation values.
    
    Returns *array* of expectation values
    """

    L = liouvillian(H, c_op_list)
    rho0 = steady(L)

    return odesolve(H, b_op * rho0, tlist, c_op_list, [a_op])[0]

def correlation_ode(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(t+\\tau)B(t)\\right>`
    using the ode solver, and the quantum regression theorem.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `rho0` (:class:`qutip.Qobj`) initial density matrix.
        
        `tlist` (*list/array*) list of times for :math:`t`.

        `taulist` (*list/array*) list of times for :math:`\\tau`.
        
        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
                
    Returns *array* of expectation values.
    """

    if rho0 == None:
        rho0 = steadystate(H, co_op_list)

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    rho_t = odesolve(H, rho0, tlist, c_op_list, [])

    for t_idx in range(len(tlist)):
        
        C_mat[t_idx,:] = odesolve(H, b_op * rho_t[t_idx], taulist, c_op_list, [a_op])[0]

    return C_mat

def correlation_ss_mc(H, tlist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(t+\\tau)B(t)\\right>`
    using the quantum regression theorem, and the monte-carlo solver
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `rho0` (:class:`qutip.Qobj`) initial density matrix.
        
        `tlist` (*list/array*) list of times for :math:`t`.
        
        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
    
    Returns *array* of expectation values.
    """

    rho0 = steadystate(L, co_op_list)

    ntraj = 100
    return mcsolve(H, b_op * rho0, tlist, ntraj, c_op_list, [a_op])[0]

def correlation_mc(H, psi0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function :math:`\left<A(t+\\tau)B(t)\\right>`
    using the Monte-Carle solver, and the quantum regression theorem.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `rho0` (:class:`qutip.Qobj`) initial density matrix.
        
        `tlist` (*list/array*) list of times for :math:`t`.

        `taulist` (*list/array*) list of times for :math:`\\tau`.
        
        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
                
    Returns *array* of expectation values.
    """

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    ntraj = 100

    mc_opt = Mcoptions()
    mc_opt.progressbar = False

    psi_t = mcsolve(H, psi0, tlist, ntraj, c_op_list, [], mc_opt)

    for t_idx in range(len(tlist)):

        psi0_t = psi_t[0][t_idx]

        C_mat[t_idx, :] = mcsolve(H, b_op * psi0_t, tlist, ntraj, c_op_list, [a_op], mc_opt)

    return C_mat


# ------------------------------------------------------------------------------
# SPECTRUM
# ------------------------------------------------------------------------------

def spectrum_ss(H, wlist, c_op_list, a_op, b_op):
    """
    Calculate the spectrum corresponding to a correlation function
    :math:`\left<A(\\tau)B(t)\\right>`, i.e., the Fourier transform of the
    correlation funciton:
    
    .. math::
    
        S(\omega) = \int_{-\infty}^{\infty} \left<A(t)B(0)\\right> e^{-i\omega t} dt.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) system Hamiltonian.
        
        `wlist` (*list/array*) list of times for :math:`\omega`.

        `c_op_list` (list of :class:`qutip.Qobj`) list of collapse operators.
        
        `a_op` (:class:`qutip.Qobj`) for A operator.
        
        `b_op` (:class:`qutip.Qobj`) for B operator.
                
    Returns *array* with spectrum :math:`S(\omega)` for the frequencies specified in `wlist`.
    """

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steady(L)

    a_op_ss = expect(a_op, rho0)
    b_op_ss = expect(b_op, rho0)

    # eseries solution for (b * rho0)(t)
    es = ode2es(L, b_op * rho0)
   
    # correlation
    corr_es = expect(a_op, es)

    # covarience
    cov_es = corr_es - real(conjugate(a_op_ss) * b_op_ss)

    # spectrum
    spectrum = esspec(cov_es, wlist)

    return spectrum

