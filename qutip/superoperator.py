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
import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy import prod, transpose, reshape
from Qobj import *
from istests import *
from operators import destroy


def liouvillian(H, c_op_list):
    """
    Assemble the Liouvillian superoperator from a Hamiltonian and a list
    of collapse operators
    """
    L = -1j*(spre(H) - spost(H))
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

    return L

def spost(A,*args):
	if not isoper(A):
		raise TypeError('Input is not a quantum object')

	d=A.dims[0]
	S=Qobj()
	S.dims=[[d[:],A.dims[1][:]],[d[:],A.dims[0][:]]]
	S.shape=[prod(S.dims[0][0])*prod(S.dims[0][1]),prod(S.dims[1][0])*prod(S.dims[1][1])]
	S.data=sp.kron(A.data.T,sp.identity(prod(d)))
	return S
	

def spre(A):
	if not isoper(A):
		raise TypeError('Input is not a quantum object')
	d=A.dims[1]
	S=Qobj()
	S.dims=[[A.dims[0][:],d[:]],[A.dims[1][:],d[:]]]
	S.shape=[prod(S.dims[0][0])*prod(S.dims[0][1]),prod(S.dims[1][0])*prod(S.dims[1][1])]
	S.data=sp.kron(sp.identity(prod(d)),A.data)
	return S
	

