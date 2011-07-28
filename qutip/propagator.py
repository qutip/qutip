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
#from Counter import *
from odesolve import *
from essolve import *
from basis import *

def propagator(H, t, c_op_list, H_args=None):
    """
    Calculate the propagator U(t) for the density matrix or wave function
    such that 

        psi(t) = U(t) psi(0)

    or 

        rho_vec(t) = U(t) rho_vec(0)

    where rho_vec is the vector representation of the density matrix.
    """

    if len(c_op_list) == 0:
        # calculate propagator for the wave function

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        else:
            N = H.shape[0]

        u = zeros([N, N], dtype=complex)
        
        for n in range(0, N):

            psi0 = basis(N, n)
            psi_t = odesolve(H, psi0, [0, t], c_op_list, [], H_args)

            u[:,n] = psi_t[1].full().T

    else:
        # calculate the propagator for the vector representation of the 
        # density matrix

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        else:
            N = H.shape[0]

        u = zeros([N*N, N*N], dtype=complex)
        
        for n in range(0, N*N):

            psi0  = basis(N*N, n)
            rho0  = Qobj(vec2mat(psi0.full()))
            rho_t = odesolve(H, rho0, [0, t], c_op_list, [], H_args)

            u[:,n] = mat2vec(rho_t[1].full()).T


    return Qobj(u)

def get_min_and_index(lst): 
    minval,minidx = lst[0],0 
    for i,v in enumerate(lst[1:]): 
        if v < minval: 
            minval,minidx = v,i+1 
    return minval,minidx 

def propagator_steadystate(U):
    """
    Find the steady state for successive applications of the propagator U.
    """

    evals,evecs = la.eig(U.full())

    ev_min, ev_idx = get_min_and_index(abs(evals-1.0))

    N = int(sqrt(len(evals)))

    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]))

    return rho * (1 / real(rho.tr()))


def floquet_states(H, t, c_op_list, H_args=None):
    """
    Calculate the floquet states for a driven system with period t.
    """

    # get the unitary propagator
    U = propagator(H, t, [], H_args)

    # find the eigenstates for the propagator
    evals,evecs = la.eig(U.full())

    eargs = angle(evals, False)

#    print "eargs =", eargs
    #if (p2 > M_PI/2) p2 -= 2*M_PI;        
    #if (p2 < M_PI/2) p2 += 2*M_PI;        
#    for i in [list(array(eargs) < pi/2).index(True)]:
#        print "shift up:   ", i
#        eargs[i] += 2 * pi
#    print "eargs =", eargs
#    for i in [list(array(eargs) > pi/2).index(True)]:
#        print "shift down: ", i
#        eargs[i] -= 2 * pi    
#    print "eargs =", eargs

    # sort by the angle (might need shifts)
    order = argsort(eargs)

    evals_order = evals[order]
    evecs_order = evecs[order]

    e_quasi = - angle(evals_order, False) / t

    return evecs_order, e_quasi


def floquet_states_t(fval, fvec, H, t, c_op_list, H_args=None):
    """
    Calculate the floquet states for a driven system with period t.
    """

    # get the unitary propagator
    U = propagator(H, t, [], H_args)

    fvec_t = (U.data * fvec) * diag(exp( - fval * t))

    return fvec_t

