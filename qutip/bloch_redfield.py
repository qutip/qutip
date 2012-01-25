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

from types import *
from scipy.integrate import *
from qutip.tidyup import tidyup
from qutip.Qobj import *
from qutip.superoperator import *
from qutip.expect import *
from qutip.states import *

# ------------------------------------------------------------------------------
# 
# 
# 
def brmesolve(R, rho0, tlist):
    """
    Solve the dynamics for the system using the Bloch-Redfeild master equation.
    
    ..note:: Experimental
    
    """
    
    # not yet implemented
    
    return rho0
    

# ------------------------------------------------------------------------------
# 
# 
# 
#def helper_matrix_element(bra, op, ket):
#    return (kets[0].data.T * H.data * kets[0].data)[0,0]

def bloch_redfield_tensor(H, c_ops, spectra_cb):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.
    
    ..note:: Experimental
    
    """
    
    # TODO: optimize by using data directly instead of Qobjs
    
    # TODO: check that H is a Qobj

    # use the eigenbasis         
    # kets, evals = H.eigenstates()
    #W    = zeros((N,N))
    #for n in range(N):
    #    for m in range(N):
    #        W[n,m] = evals[n] - evals[m]
    #N = len(evals)  
    #R = zeros((N*N,N*N), dtype=complex)
    
    # use computational basis
    N = H.shape[0]
    kets = [basis(N,i) for i in range(N)]
    W = H.data
    
    
    # unitary part
    R = -1.0j*(spre(H) - spost(H))
    
    for I in range(N*N):
        a,b = vec2mat_index(N, I)
        for J in range(N*N):
            c,d = vec2mat_index(N, J)
   
            # unitary part: use spre and spost above, same as this:
            # R[I,J] = -1j * (H[a,c] * (b == d) - H[d,b] * (a == c))   
   
            # dissipative part
            
            # for each operator coupling the system to the environment
            for k in range(len(c_ops)):

                R.data[I,J] += c_ops[k][a,c] * c_ops[k][d,b] * 2 * pi * (spectra_cb[k](W[c,a]) + spectra_cb[k](W[d,b]))
                       
                s1 = s2 = 0
                for n in range(N):                         
                    s1 += c_ops[k][a,n] * c_ops[k][n,c] * 2 * pi * spectra_cb[k](W[n,c])
                    s2 += c_ops[k][d,n] * c_ops[k][n,b] * 2 * pi * spectra_cb[k](W[n,d])
       
                R.data[I,J] += - (b == d) * s1 - (a == c) * s2
           
    return R
    
    
    
    
