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
from scipy import prod, finfo
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import spsolve

from qutip.Qobj import *
from qutip.istests import *
from qutip.superoperator import *


# ------------------------------------------------------------------------------
# 
def steadystate(H, c_op_list,maxiter=100,tol=1e-6):
    """
    Calculate the steady state for the evolution subject to the 
    supplied Hamiltonian and lsit of collapse operators. 
    Does nothing more than form the Louvillian for you and call steady.
    
    Args:
    
        `H` (:class:`qutip.Qobj`) Hamiltonian operator.
        
        `c_op_list` (list of :class:`qutip.Qobj`) List of collapse operators.
        
        `maxiter` (*int*) Max number of iterations to perform, default = 100.
        
        `tol` (*float*) Tolerance, default = 1e-6
    
    Returns:
    
        a :class:`qutip.Qobj` instance representing the steady state of system.
    """
    n_op = len(c_op_list)

    if n_op == 0:
        raise ValueError('Cannot calculate the steady state for a nondissipative system (no collapse operators given)')

    L = liouvillian(H, c_op_list)
    return steady(L,maxiter,tol)

def steady(L,maxiter=100,tol=1e-6):
	"""
	Calculate the steady state for the evolution subject to the 
	supplied Louvillian using the inverse power method. 
	See any Linear Algebra book with a iterative methods.
    
    Arguments:
    
        `L` (:class:`qutip.Qobj`) Liouvillian superoperator.
              
        `maxiter` (*int*) Max number of iterations to perform, default = 100.
        
        `tol` (*float*) Tolerance, default = 1e-6
    
    Returns a :class:`qutip.Qobj` instance representing the steady state of system.
    """
	eps=finfo(float).eps
	if (not isoper(L)) & (not issuper(L)):
		raise TypeError('Steady states can only be found for operators or superoperators.')
	rhoss=Qobj()
	sflag=issuper(L)
	if sflag:
		rhoss.dims=L.dims[0]
		rhoss.shape=[prod(rhoss.dims[0]),prod(rhoss.dims[1])]
	else:
		rhoss.dims=[l.dims[0],1]
		rhoss.shape=[prod(rhoss.dims[0]),1]
	n=len(L.data.todense())
	L1=L.data+eps*la.norm(L.data.todense(),inf)*sp.eye(n,n)
	v=randn(n,1)
	it=0
	while (la.norm(L.data*v,inf)>tol) and (it<maxiter):
		v=spsolve(L1,v)
		v=v/la.norm(v,inf)
		it+=1
	if it>=maxiter:
		raise ValueError('Failed to find steady state after ' + str(maxiter) +' iterations')
	rhoss.data=v
	#normalise according to type of problem
	if sflag:
		trow=eye(rhoss.shape[0],rhoss.shape[0])
		trow=reshape(trow,(1,n))
		rhoss.data=rhoss.data/sum(dot(trow,rhoss.data))
	else:
		rhoss.data=rhoss.data/la.norm(rhoss.data)
	rhoss.data=reshape(rhoss.data,(rhoss.shape[0],rhoss.shape[1])).T
	return Qobj(rhoss.data,dims=rhoss.dims,shape=rhoss.shape)
	
	
		
