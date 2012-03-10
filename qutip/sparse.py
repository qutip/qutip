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

"""
This module contains a collection of sparse routines to get around 
having to use dense matrices.
"""
import scipy.sparse as sp
from scipy import floor,mod,union1d,real,array
import scipy.linalg as la

def sp_L2_norm(op):
    """
    L2 norm for ket or bra Qobjs
    """
    if op.type=='super' or op.type=='oper':
        raise TypeError("Use L2-norm for ket or bra states only.")
    return la.norm(op.data.data,2)


def sp_inf_norm(op):
    """
    Infinity norm for Qobj
    """
    return max([sum(abs((op.data[k,:]).data)) for k in xrange(op.shape[0])])



def sp_one_norm(op):
    """
    One norm for Qobj
    """
    return max(array([sum(abs((op.data[:,k]).data)) for k in xrange(op.shape[1])]))


def sp_eigs(op,vecs=True):
    """
    Returns Eigenvalues and Eigenvectors for Qobj.  Uses sparse eigen-solver.
    
    Args:
    
        op (Qobj): Input Qobj
        
        vecs (bool): Return eigenvectors
    
    Returns:
    
        Eigenvalues and (by default) Eigenvectors
    """
    if op.type=='ket' or op.type=='bra':
        raise TypeError("Can only diagonalize operators and superoperators")
    N=op.shape[0]
    if N>=10:
        D=int(floor(N/2))
        if mod(N,2):D+=1 #if odd dimensions
        if vecs:
            big_vals,big_vecs=sp.linalg.eigs(op.data,k=N-D,which='LM')
            big_vecs=sp.csr_matrix(big_vecs,dtype=complex)
            small_vals,small_vecs=sp.linalg.eigs(op.data,k=N-(N-D),which='SM')
            small_vecs=sp.csr_matrix(small_vecs,dtype=complex)
            evecs=sp.hstack([big_vecs,small_vecs],format='csr') #combine eigenvector sets
        else:
            big_vals=sp.linalg.eigs(op.data,k=N-D,which='LM',return_eigenvectors=False)
            small_vals=sp.linalg.eigs(op.data,k=N-(N-D),which='SM',return_eigenvectors=False)
        evals=union1d(big_vals,small_vals)#combine eigenvalue sets
    else:#for dims <10 use faster dense routine
        if vecs:
            evals,evecs=la.eig(op.full())
            evecs=sp.csr_matrix(evecs,dtype=complex)
        else:
            evals=la.eigvals(op.full())
    if op.isherm:#hermitian eigvals are always real
        evals=real(evals)
    if vecs:
        return evals,evecs
    else:
        return evals