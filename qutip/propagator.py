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

import types
import numpy as np
import scipy.linalg as la

from qutip.Qobj import Qobj
from qutip.superoperator import vec2mat, mat2vec
from qutip.mesolve import mesolve
from qutip.essolve import essolve
from qutip.steady import steadystate
from qutip.states import basis
from qutip.states import projection
from qutip.Odeoptions import Odeoptions

def propagator(H, t, c_op_list, H_args=None, opt=None):
    """
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\\rho_{\mathrm vec}(t) = U(t) \\rho_{\mathrm vec}(0)`
    where :math:`\\rho_{\mathrm vec}` is the vector representation of the
    density matrix.
    
    Parameters
    ----------
    H : qobj or list
        Hamiltonian as a Qobj instance of a nested list of Qobjs and 
        coefficients in the list-string or list-function format for
        time-dependent Hamiltonians (see description in :func:`qutip.mesolve`).  
    t : float or array-like 
        Time or list of times for which to evaluate the propagator.   
    c_op_list : list 
        List of qobj collapse operators.
    H_args : list/array/dictionary 
        Parameters to callback functions for time-dependent Hamiltonians.
    
    Returns
    -------
     a : qobj 
        Instance representing the propagator :math:`U(t)`.
    
    """

    if opt == None:
        opt = Odeoptions()
        opt.rhs_reuse = True

    tlist = [0, t] if isinstance(t,(int,float,np.int64,np.float64)) else t
    
    if len(c_op_list) == 0:
        # calculate propagator for the wave function

        if isinstance(H, types.FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        elif isinstance(H, list):
            if isinstance(H[0], list):
                H0 = H[0][0]
                N = H0.shape[0]            
            else:
                H0 = H[0]
                N = H0.shape[0] 
        else:
            N = H.shape[0]

        u = np.zeros([N, N, len(tlist)], dtype=complex)

        for n in range(0, N):
            psi0 = basis(N, n)
            output = mesolve(H, psi0, tlist, [], [], H_args, opt)
            for k, t in enumerate(tlist):
                u[:,n,k] = output.states[k].full().T

        # todo: evolving a batch of wave functions:
        #psi_0_list = [basis(N, n) for n in range(N)]
        #psi_t_list = mesolve(H, psi_0_list, [0, t], [], [], H_args, opt)
        #for n in range(0, N):
        #    u[:,n] = psi_t_list[n][1].full().T

    else:
        # calculate the propagator for the vector representation of the 
        # density matrix

        if isinstance(H, types.FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        elif isinstance(H, list):
            if isinstance(H[0], list):
                H0 = H[0][0]
                N = H0.shape[0]            
            else:
                H0 = H[0]
                N = H0.shape[0]            
        else:
            N = H.shape[0]

        u = np.zeros([N*N, N*N, len(tlist)], dtype=complex)
        
        for n in range(0, N*N):
            psi0  = basis(N*N, n)
            rho0  = Qobj(vec2mat(psi0.full()))
            output = mesolve(H, rho0, tlist, c_op_list, [], H_args, opt)
            for k, t in enumerate(tlist):
                u[:,n,k] = mat2vec(output.states[k].full()).T

    if len(tlist) == 2:
        return Qobj(u[:,:,1])
    else:
        return [Qobj(u[:,:,k]) for k in range(len(tlist))]


def _get_min_and_index(lst): 
    """
    Private function for obtaining min and max indicies.
    """
    minval,minidx = lst[0],0 
    for i,v in enumerate(lst[1:]): 
        if v < minval: 
            minval,minidx = v,i+1 
    return minval,minidx 


def propagator_steadystate(U):
    """Find the steady state for successive applications of the propagator
    :math:`U`.
    
    Parameters
    ----------
    U : qobj 
        Operator representing the propagator.
    
    Returns
    ------- 
    a : qobj
        Instance representing the steady-state density matrix.
    
    """

    evals,evecs = la.eig(U.full())

    ev_min, ev_idx = _get_min_and_index(abs(evals-1.0))

    N = int(np.sqrt(len(evals)))
    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]))
    rho = rho * (1.0/rho.tr())
    rho = 0.5*(rho+rho.dag()) #make sure rho is herm
    return rho
