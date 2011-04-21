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

def simdiag(ops):
    """
    Simultaneous diagonalization of Commuting, Hermitian matricies
    """
    tol=1e-12
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
        if not isherm(A):
            raise TypeError('Matricies must be Hermitian')
        for kk in range(jj):
            B=ops[kk]
            if (A*B-B*A).norm()/(A*B).norm()>tol:
                raise TypeError('Matricies must commute.')
    A=ops[0]
    eigvals,eigvecs=la.eig(A.full())
    zipped=zip(-eigvals,xrange(len(eigvals)))
    zipped.sort()
    ds,perm=zip(*zipped)
    ds=-real(array(ds));perm=array(perm)
    eigvecs_perm=zeros(A.shape,dtype=complex)
    for kk in range(len(perm)):#matrix with sorted eigvecs in columns
		eigvecs=eigvecs[:,perm]
    k=0
    while k<len(ds):#find degenerate eigenvalues
        inds=abs(ds-ds[k])<max(tol,tol*abs(ds[k]))
        if sum(inds)>1:
            degen_vecs=eigvecs[:,inds]
            degen_vecs=degen(tol,degen_vecs,array([ops[kk] for kk in xrange(1,num_ops)]))
        k=max(array(range(len(inds)))[inds])+1
    eigvecs[:,inds]=degen_vecs
    eigvals_out=zeros((num_ops,len(ds)))
    for kk in xrange(num_ops):
        eigvals_out[kk,:]=eigvecs.conj(),ops[kk]
    return eigvals_out,eigvecs
    



def degen(tol,vlist,ops):
	if len(ops)==0:
		return vlist
	else:
		num_ops=len(ops)
		A=ops[0]
		evals,evecs=la.eigh(dot(vlist.conj().T,A.data*vlist))
		zipped=zip(-evals,xrange(len(evals)))
		zipped.sort()
		ds,perm=zip(*zipped)
		ds=-real(array(ds));perm=array(perm)
		for kk in xrange(len(perm)):#matrix with sorted eigvecs in columns
			evecs=evecs[:,perm]
		vnew=dot(vlist,evecs)
		k=0
		while k<len(ds):
			inds=abs(ds-ds[k])<max(tol,tol*abs(ds[k]))
			if sum(inds)>1:
				vnew=vnew[:,inds]
				vnew[:,inds]=degen(tol,vnew,array([ops[kk] for kk in xrange(2,num_ops)]))
			k=max(array(range(len(inds)))[inds])+1
		return vnew



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
	
	x,y=simdiag([op1])
	print x
	print ''
	print y[:,1]
	
	











