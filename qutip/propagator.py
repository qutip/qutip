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
from spre import *
from spost import *
from Counter import *
from ode_solve import *
from ode2es import *
from mcsolve import *

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
    
        print "N =", N

        u = zeros([N, N], dtype=complex)
        
        for n in range(0, N):

            psi0 = basis(N, n)
            psi_t = me_ode_solve(H, psi0, [0, t], c_op_list, [], H_args)

            #print "psi(t) =", psi_t[1].full().T

            u[:,n] = psi_t[1].full().T


    else:
        # calculate the propagator for the vector representation of the 
        # density matrix

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        else:
            N = H.shape[0]
    
        print "N =", N

        u = zeros([N*N, N*N], dtype=complex)
        
        for n in range(0, N*N):

            psi0  = basis(N*N, n)
            rho0  = Qobj(psi0.full().reshape([N,N]))
            rho_t = me_ode_solve(H, rho0, [0, t], c_op_list, [], H_args)

            #print "rho(t) =", rho_t[1].full().reshape([1, N*N])

            u[:,n] = rho_t[1].full().reshape([1, N*N])


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

    rho = Qobj((evecs.T)[ev_idx].reshape(N, N))

    rho = rho * (1 / real(rho.tr()))

    return rho









