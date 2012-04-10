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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from scipy import *
from qutip.Qobj import *

from qutip.superoperator import *
from qutip.odesolve import *
from qutip.essolve import *
from qutip.steady import steadystate
from qutip.states import basis
from qutip.states import projection
from qutip.Odeoptions import Odeoptions

def propagator(H, t, c_op_list, H_args=None):
    """
    Calculate the propagator U(t) for the density matrix or wave function such that
    :math:`\psi(t) = U(t)\psi(0)` or :math:`\\rho_{\mathrm vec}(t) = U(t) \\rho_{\mathrm vec}(0)`
    where :math:`\\rho_{\mathrm vec}` is the vector representation of the density matrix.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) Hamiltonian.
        
        `t` (*float*) time.
        
        `c_op_list` (list of :class:`qutip.Qobj`) collapse operators.
        
        `H_args` (*list/array*) [optional] parameters to callback functions for time-dependent Hamiltonians.
    
    Returns a :class:`qutip.Qobj` instance representing the propagator :math:`U(t)`.
    """

    if len(c_op_list) == 0:
        # calculate propagator for the wave function

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        elif isinstance(H, list):
            H0 = H[0]
            N = H0.shape[0]
        else:
            N = H.shape[0]

        u = zeros([N, N], dtype=complex)
        
        opt = Odeoptions()
        opt.rhs_reuse = True
                
        for n in range(0, N):
            psi0 = basis(N, n)
            psi_t = mesolve(H, psi0, [0, t], [], [], H_args, opt)
            u[:,n] = psi_t[1].full().T

        # todo: evolving a batch of wave functions:
        #psi_0_list = [basis(N, n) for n in range(N)]
        #psi_t_list = mesolve(H, psi_0_list, [0, t], [], [], H_args, opt)
        #for n in range(0, N):
        #    u[:,n] = psi_t_list[n][1].full().T

    else:
        # calculate the propagator for the vector representation of the 
        # density matrix

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        elif isinstance(H, list):
            H0 = H[0]
            N = H0.shape[0]            
        else:
            N = H.shape[0]

        u = zeros([N*N, N*N], dtype=complex)
        
        for n in xrange(0, N*N):
            psi0  = basis(N*N, n)
            rho0  = Qobj(vec2mat(psi0.full()))
            rho_t = mesolve(H, rho0, [0, t], c_op_list, [], H_args)
            u[:,n] = mat2vec(rho_t[1].full()).T

    return Qobj(u)


def get_min_and_index(lst): 
    """
    Private function for obtaining min and max indicies.
    """
    minval,minidx = lst[0],0 
    for i,v in enumerate(lst[1:]): 
        if v < minval: 
            minval,minidx = v,i+1 
    return minval,minidx 


def propagator_steadystate(U):
    """
    Find the steady state for successive applications of the propagator :math:`U`.
    
    Arguments:
    
        `U` (:class:`qutip.Qobj`) Operator representing the propagator.
    
    Returns a :class:`qutip.Qobj` instance representing the steady-state vector.
    """

    evals,evecs = la.eig(U.full())

    ev_min, ev_idx = get_min_and_index(abs(evals-1.0))

    N = int(sqrt(len(evals)))

    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]))

    return rho * (1 / real(rho.tr()))
