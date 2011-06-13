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
from Qobj import *
import scipy.linalg as la
from varargout import varargout

def simdiag(ops):
    """
    Simultaneous diagonalization of Commuting, Hermitian matricies.
    
    @param ops single or list/array of quantum objects
    """
    outputs=varargout()
    tol=1e-14
    start_flag=0
    if not any(ops):
        raise ValueError('Need at least one input operator.')
    if not isinstance(ops,(list,ndarray)):
        ops=array([ops])
    num_ops=len(ops)
    for jj in xrange(num_ops):
        A=ops[jj]
        shape=A.shape
        if shape[0]!=shape[1]:
            raise TypeError('Matricies must be square.')
        if start_flag==0:
            s=shape[0]
        if s!=shape[0]:
            raise TypeError('All matricies must be the same shape')
        if not A.isherm:
            raise TypeError('Matricies must be Hermitian')
        for kk in range(jj):
            B=ops[kk]
            if (A*B-B*A).norm()/(A*B).norm()>tol:
                raise TypeError('Matricies must commute.')
    #-----------------------------------------------------------------
    A=ops[0]
    eigvals,eigvecs=la.eig(A.full())
    zipped=zip(-eigvals,xrange(len(eigvals)))
    zipped.sort()
    ds,perm=zip(*zipped)
    ds=-real(array(ds));perm=array(perm)
    eigvecs_array=array([zeros((A.shape[0],1),dtype=complex) for k in xrange(A.shape[0])])
    
    for kk in xrange(len(perm)):#matrix with sorted eigvecs in columns
        eigvecs_array[kk][:,0]=eigvecs[:,perm[kk]]
    k=0
    rng=arange(len(eigvals))
    while k<len(ds):#find degenerate eigenvalues
        inds=array(abs(ds-ds[k])<max(tol,tol*abs(ds[k])))#get indicies of degenerate eigvals 
        inds=rng[inds]
        if len(inds)>1:#if at least 2 eigvals are degenerate
            eigvecs_array[inds]=degen(tol,eigvecs_array[inds],array([ops[kk] for kk in xrange(1,num_ops)]))
        k=max(inds)+1
    eigvals_out=zeros((num_ops,len(ds)),dtype=float)
    kets_out=array([Qobj(eigvecs_array[j]/la.norm(eigvecs_array[j]),dims=[ops[0].dims[0],[1]],shape=[ops[0].shape[0],1]) for j in xrange(len(ds))])
    if outputs==1:
        return kets_out
    else:
        for kk in xrange(num_ops):
            for j in xrange(len(ds)):
			    eigvals_out[kk,j]=real(dot(eigvecs_array[j].conj().T,ops[kk].data*eigvecs_array[j]))
        return kets_out,eigvals_out
    



def degen(tol,in_vecs,ops):
    n=len(ops)
    if n==0:
        return in_vecs
    A=ops[0]
    vecs=column_stack(in_vecs)
    eigvals,eigvecs=la.eig(dot(vecs.conj().T,dot(A.full(),vecs)))
    zipped=zip(-eigvals,xrange(len(eigvals)))
    zipped.sort()
    ds,perm=zip(*zipped)
    ds=-real(array(ds));perm=array(perm)
    vecsperm=zeros(eigvecs.shape,dtype=complex)
    for kk in xrange(len(perm)):#matrix with sorted eigvecs in columns
        vecsperm[:,kk]=eigvecs[:,perm[kk]]
    vecs_new=dot(vecs,vecsperm)
    vecs_out=array([zeros((A.shape[0],1),dtype=complex) for k in xrange(len(ds))])
    for kk in xrange(len(perm)):#matrix with sorted eigvecs in columns
		vecs_out[kk][:,0]=vecs_new[:,kk]
    k=0
    rng=arange(len(ds))
    while k<len(ds):
        inds=array(abs(ds-ds[k])<max(tol,tol*abs(ds[k])))#get indicies of degenerate eigvals 
        inds=rng[inds]
        if len(inds)>1:#if at least 2 eigvals are degenerate
            vecs_out[inds]=degen(tol,vecs_out[inds],array([ops[jj] for jj in xrange(1,n)]))
        k=max(inds)+1
    return vecs_out






if __name__ == "__main__":
	from Qobj import *
	from tensor import *
	from operators import *
	sx1=tensor(sigmax(),qeye(2),qeye(2))
	sy1=tensor(sigmay(),qeye(2),qeye(2))

	sx2=tensor(qeye(2),sigmax(),qeye(2))
	sy2=tensor(qeye(2),sigmay(),qeye(2))

	sx3=tensor(qeye(2),qeye(2),sigmax())
	sy3=tensor(qeye(2),qeye(2),sigmay())

	op1=sx1*sy2*sy3
	op2=sy1*sx2*sy3
	op3=sy1*sy2*sx3
	op4=sx1*sx2*sx3
	
	x,y=simdiag([op1,op2,op3,op4])
	print y
	print ''
	print x
	
	    
	
	











