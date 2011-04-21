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
from qobj import qobj
import scipy.sparse as sp
from scipy.special import lpmv
from istests import isket


def orbital(theta,phi,*args):
	psi=0.0

	if isinstance(args[0],list):
		# use the list in args[0] 
		args = args[0]

	for k in range(0,len(args)):
		ket=args[k]
		if not isket(ket):
			raise TypeError('Invalid input ket in orbital')
		sk=ket.shape
		l=(sk[0]-1)/2
		if l!=floor(l):
			raise ValueError('Kets must have odd number of components in orbital')
		if l==0:
			SPlm=sqrt(2)*ones([1,size(theta)])
		else:
			SPlm=sch_lpmv(l,cos(theta))

		fac = sqrt((2.0*l+1)/(8*pi))

		psi = psi + (sqrt(2) * fac * ket.full()[l]).T * SPlm[0,:]

		for m in range(1,l+1):
			psi=psi+((-1.0)**m*fac*ket.full()[l-m]*exp(1j*m*phi)).T*SPlm[m,:]
		for m in range(-l,0):
			psi=psi+(fac*ket.full()[l-m]*exp(1j*m*phi)).T*SPlm[abs(m),:]
	return psi
		



#Schmidt Semi-normalized Associated Legendre Functions
def sch_lpmv(n,x):
	'''
	Outputs array of Schmidt Seminormalized Associated Legendre Functions S_{n}^{m}
	for m<=n.
	'''
	sch=array([1.0])
	sch2=array([(-1.0)**m*sqrt((2.0*factorial(n-m))/factorial(n+m)) for m in range(1,n+1)])
	sch=append(sch,sch2)
	if isinstance(x,float) or len(x)==1:
		leg=lpmv(range(0,n+1),n,x)
		return array([sch*leg]).T
	else:
		for j in range(0,len(x)):
			leg=lpmv(range(0,n+1),n,x[j])
			if j==0:
				out=array([sch*leg]).T
			else:
				out=append(out,array([sch*leg]).T,axis=1)
	return out


#print sch_lpmv(5,array([.1,.2,.3]))

