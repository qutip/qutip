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
import os
import multiprocessing
from scipy import *
from scipy.linalg import *
import scipy.sparse as sp

from jmat import jmat
from Qobj import Qobj

#
# operators:
#
#

def sigmap():
	return jmat(1/2.,'+')

def sigmax():
	return 2.0*jmat(1.0/2,'x')

def sigmay():
	return 2.0*jmat(1.0/2,'y')

def sigmaz():
	return 2.0*jmat(1.0/2,'z')

def sigmam():
    return jmat(1/2.,'-')



#
#DESTROY returns annihilation operator for N dimensional Hilbert space
# out = destroy(N), N is integer value &  N>0
#
def destroy(N):
	'''
	Destruction (lowering) operator for Hilbert space of dimension N
	input: N = size of hilbert space
	output: Qobj
	'''
	if not isinstance(N,int):#raise error if N not integer
		raise ValueError("Hilbert space dimension must be integer value")
	return Qobj(sp.spdiags(sqrt(range(0,N)),1,N,N,format='csr'))

#
#CREATE returns creation operator for N dimensional Hilbert space
# out = create(N), N is integer value &  N>0
#
def create(N):
	'''
	Creation (raising) operator for Hilbert space of dimension N
	input: N = size of hilbert space
	output: Qobj
	'''
	if not isinstance(N,int):#raise error if N not integer
		raise ValueError("Hilbert space dimension must be integer value")
	qo=destroy(N) #create operator using destroy function
	qo.data=qo.data.T #transpsoe data in Qobj
	return Qobj(qo)


#
#QEYE returns identity operator for an N dimensional space
# a = qeye(N), N is integer & N>0
#
def qeye(N):
	if logical_or(not isinstance(N,int),N<0):#check if N is int and N>0
		raise ValueError("N must be integer N>=0")
	return Qobj(eye(N))
	
