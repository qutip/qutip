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
import scipy.sparse as sp
from scipy.linalg import *
from Qobj import *


def ptrace(rho,sel):
    """
    Compute partial trace of composite quantum object formed by :func:`qutip.tensor`
    
    Arguments:
    
        rho (*Qobj*): Input composite quantum object
    
        sel (*int* or *list/array*): index or indices for components to keep.
    
    Returns the density matrix of components from sel as a *Qobj* 
    """
    if isinstance(sel,int):
        sel=array([sel])
    sel=asarray(sel)
    drho=rho.dims[0]
    N=prod(drho)
    M=prod(asarray(drho).take(sel))
    if prod(rho.dims[1]) == 1:
        rho = rho * rho.dag()
    perm = sp.lil_matrix((M*M,N*N))
    rest=setdiff1d(arange(len(drho)),asarray(sel)) #all elements in range(len(drho)) not in sel set
    ilistsel=selct(sel,drho)
    indsel=list2ind(ilistsel,drho)
    ilistrest=selct(rest,drho)
    indrest=list2ind(ilistrest,drho)
    irest=(indrest-1)*N+indrest
    m=0
    for k in indsel:
        temp=(k-1)*N
        for l in indsel:
            m=m+1
            col=irest+temp+l-1
            for i in xrange(len(col)):
                perm[m-1,int(col[i][0])-1]=1
    perm.tocsr()
    rws=prod(shape(rho.data))
    rho1=Qobj()

    rhdata=dot(perm,rho.data.tolil().reshape((rws,1)))
    rhdata=rhdata.tolil().reshape((M,M))
    rho1.data=rhdata.tocsr()

    dims_kept0=asarray(rho.dims[0]).take(sel)
    dims_kept1=asarray(rho.dims[0]).take(sel)
    rho1.dims=[dims_kept0.tolist(),dims_kept1.tolist()]
    rho1.shape=[prod(dims_kept0),prod(dims_kept1)]
    rho1.size=[1,1]
    return Qobj(rho1)



def list2ind(ilist,dims):
	"""!
	Private function returning indicies
	"""
	ilist=asarray(ilist)
	dims=asarray(dims)
	irev=fliplr(ilist)-1
	fact=append(array([1]),(cumprod(flipud(dims)[:-1])))
	fact=fact.reshape(len(fact),1)
	return sort(dot(irev,fact)+1,0)

def selct(sel,dims):
	"""
	Private function finding selected components
	"""
	sel=asarray(sel)#make sure sel is array
	dims=asarray(dims)#make sure dims is array
	rlst=dims.take(sel)
	rprod=prod(rlst)
	ilist=ones((rprod,len(dims)));
	counter=arange(rprod)
	for k in xrange(len(sel)):
		ilist[:,sel[k]]=remainder(fix(counter/prod(dims[sel[k+1:]])),dims[sel[k]])+1
	return ilist
