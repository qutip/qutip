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
import scipy.sparse as sp
from scipy.linalg import *
from qutip.Qobj import *

def ptrace(rho,sel):
    """
    Compute partial trace of composite quantum object formed by :func:`qutip.tensor`
    
    Args:
    
        rho (Qobj): Input composite quantum object.
    
        sel (int or list/array): index or indices for components to keep.
    
    Returns: 
    
        The density matrix of components from sel as a Qobj. 
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
    rest=setdiff1d(arange(len(drho)),sel) #all elements in range(len(drho)) not in sel set
    ilistsel=selct(sel,drho)
    indsel=list2ind(ilistsel,drho)
    ilistrest=selct(rest,drho)
    indrest=list2ind(ilistrest,drho)
    irest=(indrest-1)*N+indrest-2
    # Possibly use parfor here if M > some value ?
    perm.rows=array([(irest+(indsel[int(floor(m/M))]-1)*N+indsel[int(mod(m,M))]).T[0] for  m in xrange(M**2)])
    #perm.data=ones_like(perm.rows,dtype=int)
    perm.data=ones_like(perm.rows)
    perm.tocsr()
    rws=prod(shape(rho.data))
    rho1=Qobj()
    rhdata=perm*csr_to_col(rho.data)
    rhdata=rhdata.tolil().reshape((M,M))
    rho1.data=rhdata.tocsr()
    dims_kept0=asarray(rho.dims[0]).take(sel)
    dims_kept1=asarray(rho.dims[0]).take(sel)
    rho1.dims=[dims_kept0.tolist(),dims_kept1.tolist()]
    rho1.shape=[prod(dims_kept0),prod(dims_kept1)]
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
	return array(sort(dot(irev,fact)+1,0),dtype=int)

def selct(sel,dims):
	"""
	Private function finding selected components
	"""
	sel=asarray(sel)#make sure sel is array
	dims=asarray(dims)#make sure dims is array
	rlst=dims.take(sel)
	rprod=prod(rlst)
	ilist=ones((rprod,len(dims)),dtype=int);
	counter=arange(rprod)
	for k in xrange(len(sel)):
		ilist[:,sel[k]]=remainder(fix(counter/prod(dims[sel[k+1:]])),dims[sel[k]])+1
	return ilist



def csr_to_col(mat):
    """
    Private function for reshape density matrix csr_matrix to a column csr_matrix
        without using lil (reshape) or csc (transpose) matrices which fail for large
        matricies.
    """
    mat.sort_indices()
    rows=array([len(range(mat.indptr[i],mat.indptr[i+1])) for i in xrange(mat.shape[1])])
    rows=[[k for j in xrange(rows[k])] for k in xrange(len(rows))] 
    rows=array([item for sublist in rows for item in sublist])
    datlen=len(mat.data)
    ptrs=zeros((datlen+2),dtype=int)
    ptrs[1:-1]=(mat.shape[1]*rows+mat.indices)+1
    ptrs[-1]=prod(mat.shape)
    values=arange(datlen+1)#values to use in ptrs
    counts=diff(ptrs) #number of times values should be repeated
    ptrs=zeros(sum(counts)+1,dtype=int)
    ptrs[-1]=datlen
    ptrs[:-1]=repeat(values,counts) #append the number of data elems (per csr format)
    inds=zeros(datlen,dtype=int) #since this is col vec, all inds = 0 
    out=sp.csr_matrix((mat.data,inds,ptrs),shape=(prod(mat.shape),1),dtype=complex)
    return out
