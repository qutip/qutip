# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
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
from scipy import sqrt,ceil,floor,mod,union1d,real,array,sort,flipud,arange,fliplr,hstack,delete
import scipy.linalg as la
from numpy import frexp, shape

def sp_fro_norm(op):
    """
    Frobius norm for Qobj
    """
    op=op*op.dag()
    return sqrt(op.tr())


def sp_inf_norm(op):
    """
    Infinity norm for Qobj
    """
    return max([sum(abs((op.data[k,:]).data)) for k in range(op.shape[0])])


def sp_L2_norm(op):
    """
    L2 norm for ket or bra Qobjs
    """
    if op.type=='super' or op.type=='oper':
        raise TypeError("Use L2-norm for ket or bra states only.")
    return la.norm(op.data.data,2)



def sp_max_norm(op):
    """
    Max norm for Qobj
    """
    if any(op.data.data):
        max_nrm=max(abs(op.data.data))
    else:
        max_nrm=0
    return max_nrm


def sp_one_norm(op):
    """
    One norm for Qobj
    """
    return max(array([sum(abs((op.data[:,k]).data)) for k in range(op.shape[1])]))


def sp_eigs(op,vecs=True,sparse=False,sort='low',eigvals=0,tol=0,maxiter=100000):
    """Returns Eigenvalues and Eigenvectors for Qobj.
    Uses dense eigen-solver unless user sets sparse=True.
    
    Parameters
    ----------
    op : qobj 
        Input quantum operator
    
    vecs : bool {True , False}
        Flag for requesting eigenvectors
    
    sparse : bool {False , True}
        Flag to use sparse solver
    
    sort : str {'low' , 'high}
        Return lowest or highest eigenvals/vecs
    
    eigvals : int 
        Number of eigenvals/vecs to return.  Default = 0 (return all)
    
    tol : float
        Tolerance for sparse eigensolver.  Default = 0 (Machine precision)
    
    maxiter : int 
        Max. number of iterations used by sparse sigensolver.
    
    Returns
    -------
    Array of eigenvalues and (by default) array of corresponding Eigenvectors.
    
    """
    if op.type=='ket' or op.type=='bra':
        raise TypeError("Can only diagonalize operators and superoperators")
    N=op.shape[0]
    if eigvals==N:eigvals=0
    if eigvals>N:raise ValueError("Number of requested eigen vals/vecs must be <= N.")
    if eigvals>0 and not op.isherm:sparse=True #only sparse routine can get selected eigs for nonherm matricies
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
                    big_vals,big_vecs=sp.linalg.eigs(op.data,k=num_large,which='LM',tol=tol,maxiter=maxiter)
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
                    big_vals=sp.linalg.eigsh(op.data,k=num_large,which='LM',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                if num_small>0:
                    small_vals=sp.linalg.eigsh(op.data,k=num_small,which='SM',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                    small_vals=small_vals
            else:
                if num_large>0:
                    big_vals=sp.linalg.eigs(op.data,k=num_large,which='LR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
                if num_small>0:
                    small_vals=sp.linalg.eigs(op.data,k=num_small,which='SR',return_eigenvectors=False,tol=tol,maxiter=maxiter)
        evals=hstack((small_vals,big_vals))
        _zipped = list(zip(evals,range(len(evals))))
        _zipped.sort()
        evals,perm = list(zip(*_zipped))
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
        _zipped = list(zip(evals,range(len(evals))))
        _zipped.sort()
        evals,perm = list(zip(*_zipped))
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


def sp_expm(qo):
    """
    Sparse matrix exponential of a quantum operator.
    Called by the Qobj expm method.
    """
    A=qo.data #extract Qobj data (sparse matrix)
    m_vals=array([3,5,7,9,13])
    theta=array([0.01495585217958292,0.2539398330063230,0.9504178996162932,2.097847961257068,5.371920351148152],dtype=float)
    normA=sp_one_norm(qo)
    if normA<=theta[-1]:
        for ii in range(len(m_vals)):
            if normA<=theta[ii]:
                F=_pade(A, m_vals[ii])
                break
    else:
        t,s=frexp(normA/theta[-1])
        s=s-(t==0.5)
        A=A/2.0**s
        F=_pade(A, m_vals[-1])
        for i in range(s):
            F=F*F

    return F

def _pade(A, m):
    n=shape(A)[0]
    c=_padecoeff(m)
    if m!=13:
        apows= [[] for jj in range(int(ceil((m+1)/2)))]
        apows[0]=sp.eye(n,n,format='csr')
        apows[1]=A*A
        for jj in range(2,int(ceil((m+1)/2))):
            apows[jj]=apows[jj-1]*apows[1]
        U=sp.lil_matrix((n,n)).tocsr(); V=sp.lil_matrix((n,n)).tocsr()
        for jj in range(m,0,-2):
            U=U+c[jj]*apows[jj//2]
        U=A*U
        for jj in range(m-1,-1,-2):
            V=V+c[jj]*apows[(jj+1)//2]
        F=la.solve((-U+V).todense(),(U+V).todense())
        return sp.lil_matrix(F).tocsr()
    elif m==13:
        A2=A*A
        A4=A2*A2
        A6=A2*A4
        U = A*(A6*(c[13]*A6+c[11]*A4+c[9]*A2)+c[7]*A6+c[5]*A4+c[3]*A2+c[1]*sp.eye(n,n).tocsr())
        V = A6*(c[12]*A6 + c[10]*A4 + c[8]*A2)+ c[6]*A6 + c[4]*A4 + c[2]*A2 + c[0]*sp.eye(n,n).tocsr()
        F=la.solve((-U+V).todense(),(U+V).todense()) 
        return sp.csr_matrix(F)

def _padecoeff(m):
    """
    Private function returning coefficients for Pade approximation.
    """
    if m==3:
        return array([120, 60, 12, 1])
    elif m==5:
        return array([30240, 15120, 3360, 420, 30, 1])
    elif m==7:
        return array([17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1])
    elif m==9:
        return array([17643225600, 8821612800, 2075673600, 302702400, 30270240,2162160, 110880, 3960, 90, 1])
    elif m==13:
        return array([64764752532480000, 32382376266240000, 7771770303897600,1187353796428800, 129060195264000, 10559470521600,670442572800, 33522128640, 1323241920,40840800, 960960, 16380, 182, 1])



