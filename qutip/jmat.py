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
from scipy import any,fix,arange,sqrt
import scipy.sparse as sp
from Qobj import *

def jmat(j,*args):
	if (fix(2*j)!=2*j) or (j<0):
		raise TypeError('j must be a non-negative integer or half-integer')
	if not args:
		a1=Qobj(0.5*(jplus(j)+jplus(j).conj().T))
		a2=Qobj(0.5*1j*(jplus(j)-jplus(j).conj().T))
		a3=Qobj(jz(j))
		return [a1,a2,a3]
	if args[0]=='+':
		A=jplus(j)
	elif args[0]=='-':
		A=jplus(j).conj().T
	elif args[0]=='x':
		A=0.5*(jplus(j)+jplus(j).conj().T)
	elif args[0]=='y':
		A=-0.5*1j*(jplus(j)-jplus(j).conj().T)
	elif args[0]=='z':
		A=jz(j)
	else:
		raise TypeError('Invlaid type')
	return Qobj(A.tocsr())


def jplus(j):
	m=arange(j,-j-1,-1)
	N=len(m)
	return sp.spdiags(sqrt(j*(j+1.0)-(m+1.0)*m),1,N,N)


def jz(j):
	m=arange(j,-j-1,-1)
	N=len(m)
	return sp.spdiags(m,0,N,N)
