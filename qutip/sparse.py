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

"""
This module contains a collection of sparse routines to get around 
having to use dense matrices.
"""

import scipy.sparse as sp
from scipy import ceil,floor,mod,union1d,real,array,sort,flipud,arange,fliplr,hstack,delete
import scipy.linalg as la

def sp_L2_norm(op):
    """
    L2 norm for ket or bra Qobjs
    """
    if op.type=='super' or op.type=='oper':
        raise TypeError("Use L2-norm for ket or bra states only.")
    return la.norm(op.data.data,2)


def sp_inf_norm(op):
    """
    Infinity norm for Qobj
    """
    return max([sum(abs((op.data[k,:]).data)) for k in xrange(op.shape[0])])



def sp_one_norm(op):
    """
    One norm for Qobj
    """
    return max(array([sum(abs((op.data[:,k]).data)) for k in xrange(op.shape[1])]))


def sp_eigs(op,vecs=True,sparse=None,sort='low',eigvals=0,tol=0,maxiter=100000):
    """
    Returns Eigenvalues and Eigenvectors for Qobj.  Uses sparse eigen-solver if dims >=10 and sparse!=False.
    
    Args:
    
        op (Qobj): Input Qobj
        
        vecs (bool): Return eigenvectors?
        
        sparse (bool): Use sparse solver?. default = None (automatic)
        
        sort: (str): Return lowest or highest eigenvals/vecs
        
        eigvals (int): Number of eigenvals/vecs to return.  Default = 0 (return all)
        
        tol (float): Tolerance for sparse eigensolver.  Default = 0 (Machine precision)
        
        maxiter (int): Max. number of iterations used by sparse sigensolver.
    
    Returns:
    
        Array of eigenvalues and (by default) array of corresponding Eigenvectors.
    """
    use_sparse=1000 # of dims at which to use sparse solver
    if op.type=='ket' or op.type=='bra':
        raise TypeError("Can only diagonalize operators and superoperators")
    N=op.shape[0]
    if eigvals==N:eigvals=0
    if eigvals>N:raise ValueError("Number of requested eigen vals/vecs must be <= N.")
    if eigvals>0 and not op.isherm:sparse=True #only saprse routine can get slected eigs for nonherm matricies
    #set sparse=True for dims >= 10 if not already set
    if N>=use_sparse and sparse==None:
        sparse=True
    if N<use_sparse and sparse==None:
        sparse==False
    remove_one=False
    if eigvals==(N-1) and sparse:#calculate all eigenvalues and remove one at output if using sparse
        eigvals=0
        remove_one=True
    #set number of large and small eigenvals/vecs
    if eigvals==0:#user wants all eigs (default)
        D=int(ceil(N/2.0))
        num_large=N-D
        if not mod(N,2):M=D
        else:M=D-1
        num_small=N-M
    else:#if user wants only a few eigen vals/vecs
        if sort=='low':
            num_small=eigvals
            num_large=0
        elif sort=='high':
            num_large=eigvals
            num_small=0
        else:
            raise ValueError("Invalid option for 'sort'.")
    
    #Sparse routine
    big_vals=array([])
    small_vals=array([])
    if sparse:       
        if vecs:
            #big values
            if num_large>0:
                if op.isherm:
                    big_vals,big_vecs=sp.linalg.eigsh(op.data,k=num_large,which='LM',tol=tol,maxiter=maxiter)
                else:
                    big_vals,big_vecs=sp.linalg.eigs(op.data,k=num_large,which='LR',tol=tol,maxiter=maxiter)
                big_vecs=sp.csr_matrix(big_vecs,dtype=complex)
                big_vals=big_vals
            #small values
            if num_small>0:
                if op.isherm:
                    small_vals,small_vecs=sp.linalg.eigsh(op.data,k=num_small,which='SM',tol=tol,maxiter=maxiter)
                else:
                    small_vals,small_vecs=sp.linalg.eigs(op.data,k=num_small,which='SR',tol=tol,maxiter=maxiter)
                small_vecs=sp.csr_matrix(small_vecs,dtype=complex)
            if num_large!=0 and num_small!=0:
                evecs=sp.hstack([small_vecs,big_vecs],format='csr') #combine eigenvector sets
            elif num_large!=0 and num_small==0:
                evecs=big_vecs
            elif num_large==0 and num_small!=0:
                evecs=small_vecs
        else:
            if op.isherm:
                if num_large>0:
                    big_vals=sp.linalg.eigsh(op.data,k=num_large,which='LR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                if num_small>0:
                    small_vals=sp.linalg.eigsh(op.data,k=num_small,which='SR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                    small_vals=small_vals
            else:
                if num_large>0:
                    big_vals=sp.linalg.eigs(op.data,k=num_large,which='LR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                if num_small>0:
                    small_vals=sp.linalg.eigs(op.data,k=num_small,which='SR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
        evals=hstack((small_vals,big_vals))
        _zipped = zip(evals,range(len(evals)))
        _zipped.sort()
        evals,perm = zip(*_zipped)
        if op.isherm:evals=real(evals)
        perm=array(perm)
    
    #Dense routine for dims <10 use faster dense routine (or use if user set sparse==False)
    else:
        if vecs:
            if op.isherm:
                if eigvals==0:
                    evals,evecs=la.eigh(op.full())
                else:
                    if num_small>0:
                        evals,evecs=la.eigh(op.full(),eigvals=[0,num_small-1])
                    if num_large>0:
                        evals,evecs=la.eigh(op.full(),eigvals=[N-num_large,N-1])
            else:
                evals,evecs=la.eig(op.full())
            evecs=sp.csr_matrix(evecs,dtype=complex)
        else:
            if op.isherm:
                if eigvals==0:
                    evals=la.eigvalsh(op.full())
                else:
                    if num_small>0:
                        evals=la.eigvalsh(op.full(),eigvals=[0,num_small-1])
                    if num_large>0:
                        evals=la.eigvalsh(op.full(),eigvals=[N-num_large,N-1])
            else:
                evals=la.eigvals(op.full())
        #sort return values
        _zipped = zip(evals,range(len(evals)))
        _zipped.sort()
        evals,perm = zip(*_zipped)
        if op.isherm:evals=real(evals)
        perm=array(perm)
        
    #return eigenvectors
    if vecs:
        evecs=array([evecs[:,k] for k in perm])
    if sort=='high':#flip arrays to largest values first
        if vecs:
            evecs=flipud(evecs)
        evals=flipud(evals)
    #remove last element if requesting N-1 eigs and using sparse
    if remove_one and sparse:
            evals=delete(evals,-1)
            if vecs:
                evecs=delete(evecs,-1)
    if vecs:    
        return evals,evecs
    else:
        return evals









