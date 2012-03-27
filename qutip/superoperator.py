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
import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy import prod, transpose, reshape
from qutip.Qobj import *
from qutip.istests import *
from qutip.operators import destroy


def liouvillian(H, c_op_list):
    """Assembles the Liouvillian superoperator from a Hamiltonian 
    and a ``list`` of collapse operators.
    
    Parameters
    H : qobj
        System Hamiltonian.
        
    c_op_list : array_like 
        A ``list`` or ``array`` of collpase operators.
    
    Returns
    -------
    L : qobj
        Louvillian superoperator.
    
    """
    L = -1.0j*(spre(H) - spost(H))
    n_op = len(c_op_list)
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)
    return L


def mat2vec(mat):
    """
    Private function reshaping matrix to vector.
    """
    return mat.T.reshape(prod(shape(mat)),1)

def vec2mat(vec):
    """
    Private function reshaping vector to matrix.
    """
    n = int(sqrt(len(vec)))
    return vec.reshape((n,n)).T

def vec2mat_index(N, I):
    """
    Convert a vector index to a matrix index pair that is compatible with the
    vector to matrix rearrangement done by the vec2mat function.
    """
    j = int(I/N) 
    i = I - N * j
    return i,j

def mat2vec_index(N, i, j):
    """
    Convert a matrix index pair to a vector index that is compatible with the
    matrix to vector rearrangement done by the mat2vec function.
    """
    return i + N * j

def spost(A):
	"""Superoperator formed from post-multiplication by operator A

    Parameters
    ----------
    A : qobj
        Quantum operator for post multiplication.
    
    Returns
    -------
    super : qobj
       Superoperator formed from input qauntum object.
	"""
	if not isoper(A):
		raise TypeError('Input is not a quantum object')

	d=A.dims[0]
	S=Qobj()
	S.dims=[[d[:],A.dims[1][:]],[d[:],A.dims[0][:]]]
	S.shape=[prod(S.dims[0][0])*prod(S.dims[0][1]),prod(S.dims[1][0])*prod(S.dims[1][1])]
	S.data=sp.kron(A.data.T,sp.identity(prod(d)))
	return Qobj(S)
	

def spre(A):
	"""Superoperator formed from pre-multiplication by operator A.
    
    Parameters
    ----------
    A : qobj
        Quantum operator for pre-multiplication.
    
    Returns
    --------
    super :qobj
        Superoperator formed from input quantum object.
    
    """
	if not isoper(A):
		raise TypeError('Input is not a quantum object')
	d=A.dims[1]
	S=Qobj()
	S.dims=[[A.dims[0][:],d[:]],[A.dims[1][:],d[:]]]
	S.shape=[prod(S.dims[0][0])*prod(S.dims[0][1]),prod(S.dims[1][0])*prod(S.dims[1][1])]
	S.data=sp.kron(sp.identity(prod(d)),A.data)
	return Qobj(S)
	

