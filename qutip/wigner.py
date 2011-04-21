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
import scipy.linalg as la
import time
from tensor import tensor
from Qobj import *
from ptrace import ptrace
from basis import basis
from istests import *



def wigner(state,xvec,yvec,*args):
    N=len(xvec)
    dx=xvec[1]-xvec[0]
    if la.norm(xvec-arange(-N/2.0,N/2.0)*dx)/la.norm(xvec)>1e-6:
        raise TypeError('xvec should be of the form array(-N/2,N/2)')
    if not any(args):
        g=sqrt(2)
    else:
        g=args[0]

    if isket(state):
        wmat = wfunc1(state, xvec, yvec, g)
    elif isoper(state):
        v,d=eig(state.full())
        #v=diag(v)
        #d=diag(d)
        wmat=0
        kmax=state.shape[0]
        for k in arange(1,kmax+1):
            wmat1 = wfunc1(d[:,k], xvec, yvec, g)
            wmat  = wmat+real(v[k]*wmat1)
    else:
        raise TypeError('Invalid operand for wigner')
    return wmat


def wfunc1(psi, xvec, yvec, *args):
    if not any(args):
        g=sqrt(2)
    else:
        g=args[0]
    n=psi.shape[0]
    psi=trans(psi)
    S=oscfunc(n,xvec*g/sqrt(2))
    xpsi=dot(psi.full(),S)
    wigmat,p = wigner1(xpsi, xvec * g / sqrt(2), yvec * g / sqrt(2))
    #yval=yval*sqrt(2)/g
    wigmat=0.5*g**2*real(wigmat.T)
    return wigmat

    
def wigner1(psi,x,y):
    n=2*psi.shape[1]
    z1=hstack([array([[0]]),fliplr(psi.conj()),zeros([1,n/2-1])])
    z2=hstack([array([[0]]),psi,zeros([1,n/2-1])])
    w=la.toeplitz(zeros([n/2,1]),z1)*flipud(la.toeplitz(zeros([n/2,1]),z2))
    w=hstack([w[:,n/2:n],w[:,0:n/2]])
    w=fft(w)
    w=real(hstack([w[:,3*n/4:n],w[:,0:n/4]]))
    p=arange(-n/4,n/4)*pi/(n*(x[1]-x[0]))
    w=w/(p[1]-p[0])/n
    return w,p
    
    

def oscfunc(N,x):
    lx=len(x)
    S=zeros([N,lx])
    S[0,:]=exp(-x[:]**2/2.0)/pi**0.25
    if N==1:
        return S
    else:            
        S[1,:]=sqrt(2)*x[:]*S[0,:]
        for k in arange(1,N-1):
            S[k+1,:]=sqrt(2.0/(k+1))*x[:]*S[k,:]-sqrt(((k+1)-1.0)/(k+1))*S[k-1,:]
        return S

            
#-------------------------------------------------------------------------------
# Q FUNCTION
#
def qfunc(state, xvec, yvec, *args):
    if not any(args):
        g=sqrt(2)
    else:
        g=args[0]

    X,Y = meshgrid(xvec, xvec)
    amat = 0.5*g*(X + Y * 1j);

    if isoper(state):
        ketflag = 0
    elif isket(state):
        ketflag = 1
    else:
        TypeError('Invalid state operand to qfunc.') 

    N = prod(state.dims)
    qmat = zeros(size(amat))

    if isket(state):
        qmat = qfunc1(state, amat)
    elif isoper(state):
        v,d = eig(state.full())
        qmat = 0
        kmax = state.shape[0]
        for k in arange(0, kmax):
            qmat1 = qfunc1(d[:,k], amat)
            qmat  = qmat + real(d[k] * qmat1)

    qmat = 0.25 * qmat * g**2;
    return qmat

#
# Q-function for a pure state: Q = |<alpha|psi>|^2 / pi
#
# |psi>   = the state in fock basis
# |alpha> = the coherent state with amplitude alpha
#
def qfunc1(psi, alpha_mat):

    n = prod(psi.shape)
    psi = array(trans(psi).full())[0,:]

    #print "psi       = ", psi
    #print "factorial = ", factorial(arange(0, n))
    #print "coeff = ", fliplr([psi/sqrt(factorial(arange(0, n)))])[0]

    qmat1 = abs(polyval(fliplr([psi/sqrt(factorial(arange(0, n)))])[0], conjugate(alpha_mat))) ** 2;
    qmat1 = real(qmat1) * exp(-abs(alpha_mat)**2) / pi;

    return qmat1


