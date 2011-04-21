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
import scipy.linalg as la

def rotation(params,type):
	'''
	% ROTATION translates a 3-d rotation into various descriptions
	%  [naxis,U,euler,R] = rotation(params,type) takes one of the following 
	% inputs:
	%
	% params = [nx,ny,nz], type = 'axis':             
	%   is a rotation about (nx,ny,nz) by angle |(nx,ny,nz)| radians
	% params = 2x2 unitary matrix, type = 'SU(2)':    
	%   is a rotation specified in SU(2)
	% params = [alpha,beta,gamma], type = 'Euler':    
	%   is a rotation through Euler angles (alpha,beta,gamma)
	% params = 3x3 orthogonal matrix, type = 'SO(3)': 
	%   is a rotation specified in SO(3)
	%
	% Output is the rotation converted to all of the above descriptions.
	'''
	type=type.lower()
	if type=='euler':
		R=euler2so3axis(params)
		naxis=so32axis(R)
		U=axis2su2(naxis)
		euler=su22euler(U)
	elif type=='so(3)':
		naxis=so32axis(params)
		U=axis2su2(naxis)
		euler=su22euler(U)
		R=euler2so3(euler)
	elif type=='axis':
		U=axis2su2(params)
		euler=su22euler(U)
		R=euler2so3(euler)
		naxis=so32axis(R)
	elif type=='su(2)':
		euler=su22euler(params)
		R=euler2so3(euler)
		naxis=so32axis(R)
		U=axis2su2(naxis)
	else:
		raise TypeError('Type should be string of Euler, SO(3), SU(2), or Axis')
	return naxis,U,euler,R


def euler2so3(euler):
	alpha=euler[0]
	beta=euler[1]
	gamma=euler[2]
	ca=cos(alpha)
	cb=cos(beta)
	cg=cos(gamma)
	sa=sin(alpha)
	sb=sin(beta)
	sg=sin(gamma)
	m1=array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
	m2=array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
	m3=array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
	return dot(dot(m1,m2),m3)

#not finished#################
def su32axis(R):
	if size(R)!=(3,3) or la.norm(dot(R.T,R)-eye(3),inf)>1.0e-6 or la.det(R)<0:
		raise TypeError('Invalid rotation matrix in SO(3)')
	[V,D]=la.eig(R)
	D=la.diag(D)
	ind=nonzero(abs(D-1)<1.0e-6)
	ind=ind[0]
	n=V[:,ind]
	n=n/la.norm(n)
	nn=la.dot(n,n.T)
	rn=array([[0, -n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
	enn=eye(3)-nn
	

def axis2su2(naxis):
	if la.norm(naxis)==0:
		return eye(2)
	else:
		n=naxis/la.norm(naxis)
		s=sin(la.norm(naxis)/2.0)
		a=cos(la.norm(naxis)/2.0)-1j*n[2]*s
		b=-n[1]*s-1j*n[0]*s
		return array([[a,b],[-b.T,a.T]])


def su22euler(U):
	if size(U)!=(2,2) or la.norm(la.dot(U.T,U)-eye(2),inf)>1.0e-6 or la.norm(la.dot(U,U.T)-eye(2),inf)>1.0e-6 or la.det(U)<0:
		raise TypeError('Invalid rotation matrix in SU(2)')
	a=U[0,0]
	b=U[0,1]
	am=abs(a)
	bm=abs(b)
	if am!=0 and bm!=0:
		beta=2*arctan
















		