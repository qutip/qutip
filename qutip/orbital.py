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
from scipy import *
from qutip.Qobj import Qobj
import scipy.sparse as sp
from qutip.istests import isket
try:#for scipy v <= 0.90
    from scipy import factorial
except:#for scipy v >= 0.10
    from scipy.misc import factorial

def orbital(theta,phi,*args):
	"""Calculates an angular wave function on a sphere.
    ``psi = orbital(theta,phi,ket1,ket2,...)`` calculates 
    the angular wave function on a sphere at the mesh of points 
    defined by theta and phi which is :math:`\sum_{lm} c_{lm} Y_{lm}(theta,phi)`
    where :math:`C_{lm}` are the coefficients specified by the list of kets. 
    Each ket has 2l+1 components for some integer l.
    
    Parameters
    ----------
    theta : list/array
        Polar angles

    phi : list/array 
        Azimuthal angles

    args : list/array 
        ``list`` of ket vectors.
    
    Returns
    ------- 
    ``array`` for angular wave function    
             
	"""
	psi=0.0
	if isinstance(args[0],list):
		# use the list in args[0] 
		args = args[0]

	for k in xrange(len(args)):
		ket=args[k]
		if not ket.type=='ket':
			raise TypeError('Invalid input ket in orbital')
		sk=ket.shape
		nchk=(sk[0]-1)/2.0
		if nchk != floor(nchk):
			raise ValueError('Kets must have odd number of components in orbital')
		l=(sk[0]-1)/2
		if l==0:
			SPlm=sqrt(2)*ones((size(theta),1),dtype=complex)
		else:
			SPlm=_sch_lpmv(l,cos(theta))
		fac = sqrt((2.0*l+1)/(8*pi))
		kf=ket.full()
		psi += sqrt(2)*fac*kf[l,0]*ones((size(phi),size(theta)),dtype=complex)*SPlm[0]
		for m in xrange(1,l+1):
			psi+= ((-1.0)**m*fac*kf[l-m,0])*array([exp(1.0j*1*phi)]).T*ones((size(phi),size(theta)),dtype=complex)*SPlm[1]
		for m in xrange(-l,0):
			psi=psi+(fac*kf[l-m,0])*array([exp(1.0j*1*phi)]).T*ones((size(phi),size(theta)),dtype=complex)*SPlm[abs(m)]
	return psi
		



#Schmidt Semi-normalized Associated Legendre Functions
def _sch_lpmv(n,x):
	'''
	Outputs array of Schmidt Seminormalized Associated Legendre Functions S_{n}^{m} for m<=n.
	
	Parameters
	----------
	n : int 
	    Degree of polynomial.
	
	x : float 
	    Point at which to evaluate
	
	Returns
	-------
	array of values for Legendre functions. 
	
	'''
	from scipy.special import lpmv
	sch=array([1.0])
	sch2=array([(-1.0)**m*sqrt((2.0*factorial(n-m))/factorial(n+m)) for m in xrange(1,n+1)])
	sch=append(sch,sch2)
	if isinstance(x,float) or len(x)==1:
		leg=lpmv(arange(0,n+1),n,x)
		return array([sch*leg]).T
	else:
		for j in xrange(0,len(x)):
			leg=lpmv(range(0,n+1),n,x[j])
			if j==0:
				out=array([sch*leg]).T
			else:
				out=append(out,array([sch*leg]).T,axis=1)
	return out
