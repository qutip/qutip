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
from eseries import *
from expect import *
from superoperator import *


# ------------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
# 
def essolve(H, rho0, tlist, c_op_list, expt_op_list):
    """
    Evolution of a state vector or density matrix (`rho0`) for a given
    Hamiltonian (`H`) and set of collapse operators (`c_op_list`), by expressing
    the ODE as an exponential series. The output is either the state vector at
    arbitrary points in time (`tlist`), or the expectation values of the supplied
    operators (`expt_op_list`). 
   
    Arguments:
    
        H (Qobj): system Hamiltonian, or a callback function for time-dependent Hamiltonians.
        
        rho0 (Qobj): initial density matrix.
        
        tlist (list/array): list of times for :math:`t`.
        
        c_op_list (list of Qobj's): list of collapse operators.
        
        expt_op_list (list of Qobj's): list of operators for which to evaluate expectation values.


    Returns:
     
     An (array) of expectation values of wavefunctions/density matrices
     for the times specified by 'tlist'.        

    .. note::
    
        This solver does not support time-dependent Hamiltonians.

    """
    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)

    # Calculate the Liouvillian
    if c_op_list == None or len(c_op_list) == 0:
        L = H
    else:
        L = liouvillian(H, c_op_list)

    es = ode2es(L, rho0)

    # evaluate the expectation values      
    if n_expt_op == 0:
        result_list = [Qobj() for k in xrange(n_tsteps)]
    else:
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    for n in range(0, n_expt_op):
        result_list[n,:] = esval(expect(expt_op_list[n],es),tlist)

    return result_list

# ------------------------------------------------------------------------------
#
#
def ode2es(L, rho0):
    """
    Create an exponential series that describes the time evolution for the
    initial density matrix (or state vector) `rho0`, given the Liouvillian 
    (or Hamiltonian) `L`.
    
    Arguments:
        
        L (Qobj): Liouvillian of the system.
        
        rho0 (Qobj): Initial state vector or density matrix.
    
    Returns 
    
        eseries representing the system dynamics.
    """

    if issuper(L):
 
        # check initial state
        if isket(rho0):
            # Got a wave function as initial state: convert to density matrix.
            rho0 = rho0 * rho0.dag()
    
        w, v = la.eig(L.full())
        # w[i]   = eigenvalue i
        # v[:,i] = eigenvector i
    
        rlen = prod(rho0.shape)
        r0 = mat2vec(rho0.full())
        v0 = la.solve(v,r0)
        vv = v * sp.spdiags(v0.T, 0, rlen, rlen)
    
        out = None
        for i in range(rlen):
            qo = Qobj(vec2mat(vv[:,i]), dims=rho0.dims, shape=rho0.shape)
            if out:
                out += eseries(qo, w[i])
            else:
                out  = eseries(qo, w[i])

    elif isoper(L):

        if not isket(rho0):
            raise TypeError('Second argument must be a ket if first is a Hamiltonian.')

        w, v = la.eig(L.full())
        # w[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        rlen = prod(rho0.shape)
        r0 = rho0.full()
        v0 = la.solve(v,r0)
        vv = v * sp.spdiags(v0.T, 0, rlen, rlen)

        out = None
        for i in range(rlen):
            qo = Qobj(matrix(vv[:,i]).T, dims=rho0.dims, shape=rho0.shape)
            if out:
                out += eseries(qo, -1.0j * w[i])
            else:
                out  = eseries(qo, -1.0j * w[i])

    else:
        raise TypeError('First argument must be a Hamiltonian or Liouvillian.')
        
    return estidy(out)


