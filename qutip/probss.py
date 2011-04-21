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
from qobj import *
from tensor import *
from spre import *
from spost import *
from steady import *
from operators import *
from expect import *

def probss(E,kappa,gamma,g,wc,w0,wl,N):
	ida=qeye(N)
	idatom=qeye(2)
	a=tensor(destroy(N),idatom)
	sm=tensor(ida,sigmam())
	#Hamiltonian
	H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
	#Collapse operators
	C1=sqrt(2*kappa)*a
	C2=sqrt(gamma)*sm
	C1dC1=C1.dag()*C1
	C2dC2=C2.dag()*C2
	#Liouvillian
	LH=-1j*(spre(H)-spost(H))
	L1=spre(C1)*spost(C1.dag())-0.5*spre(C1dC1)-0.5*spost(C1dC1)
	L2=spre(C2)*spost(C2.dag())-0.5*spre(C2dC2)-0.5*spost(C2dC2)
	L=LH+L1+L2
	#find steady state
	rhoss=steady(L)
	#calculate expectation values
	count1=expect(C1dC1,rhoss)
	count2=expect(C2dC2,rhoss)
	infield=expect(a,rhoss)
	return count1,count2,infield



