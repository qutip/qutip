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
from scipy import zeros,array,arange,exp,real,imag,conj,copy,sqrt,meshgrid
import scipy.sparse as sp
import scipy.linalg as la
from qutip.tensor import tensor
from qutip.Qobj import *
from qutip.states import *
from qutip.istests import *
try:#for scipy v <= 0.90
    from scipy import factorial
except:#for scipy v >= 0.10
    from scipy.misc import factorial
    
def wigner(psi,xvec,yvec,g=sqrt(2)):
    """Wigner function for a state vector or density matrix 
    at points xvec+i*yvec.
    
    Parameters
    ----------
    state : qobj 
        A state vector or density matrix.
    
    xvec : array_like
        x-coordinates at which to calculate the Wigner function.
    
    yvec : array_like
        y-coordinates at which to calculate the Wigner function.
        
    g : float
        Scaling factor for a = 0.5*g*(x+iy), default g=sqrt(2).
    
    Returns
    --------
    W : array
        Values representing the Wigner function calculated over the specified range [xvec,yvec].
    
    
    """
    if psi.type=='ket' or psi.type=='oper':
        M=prod(psi.shape[0])
    elif psi.type=='bra':
        M=prod(psi.shape[1])
    else:
        raise TypeError('Input state is not a valid operator.')
    X,Y = meshgrid(xvec, yvec)
    amat = 0.5*g*(X + 1.0j*Y)
    wmat=zeros(shape(amat))
    Wlist=array([zeros(shape(amat),dtype=complex) for k in range(M)])
    Wlist[0]=exp(-2.0*abs(amat)**2)/pi
    if psi.type=='ket' or psi.type=='bra':
        psi=ket2dm(psi)
    wmat=real(psi[0,0])*real(Wlist[0])
    for n in range(1,M):
        Wlist[n]=(2.0*amat*Wlist[n-1])/sqrt(n)
        wmat+= 2.0*real(psi[0,n]*Wlist[n])
    for m in range(M-1):
        temp=copy(Wlist[m+1])
        Wlist[m+1]=(2.0*conj(amat)*temp-sqrt(m+1)*Wlist[m])/sqrt(m+1)
        for n in range(m+1,M-1):
            temp2=(2.0*amat*Wlist[n]-sqrt(m+1)*temp)/sqrt(n+1)
            temp=copy(Wlist[n+1])
            Wlist[n+1]=temp2
        wmat+=real(psi[m+1,m+1]*Wlist[m+1])
        for k in range(m+2,M):
            wmat+=2.0*real(psi[m+1,k]*Wlist[k])
    return 0.5*wmat*g**2
            
#-------------------------------------------------------------------------------
# Q FUNCTION
#
def qfunc(state, xvec, yvec, g=sqrt(2)):
    """Q-function of a given state vector or density matrix 
    at points xvec+i*yvec.
    
    Parameters
    ----------
    state : qobj 
        A state vector or density matrix.
    
    xvec : array_like
        x-coordinates at which to calculate the Wigner function.
    
    yvec : array_like
        y-coordinates at which to calculate the Wigner function.
        
    g : float
        Scaling factor for a = 0.5*g*(x+iy), default g=sqrt(2).
    
    Returns
    --------
    Q : array
        Values representing the Q-function calculated over the specified range [xvec,yvec].
    
    """
    X,Y = meshgrid(xvec, yvec)
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
        d,v = la.eig(state.full())
        # d[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        qmat = zeros(shape(amat))
        for k in arange(0, len(d)):
            qmat1 = qfunc1(v[:,k], amat)
            qmat += real(d[k] * qmat1)

    qmat = 0.25 * qmat * g**2;
    return qmat

#
# Q-function for a pure state: Q = |<alpha|psi>|^2 / pi
#
# |psi>   = the state in fock basis
# |alpha> = the coherent state with amplitude alpha
#
def qfunc1(psi, alpha_mat):
    """
    private function used by qfunc
    """
    n = prod(psi.shape)
    if isinstance(psi, Qobj):
        psi = array(psi.trans().full())[0,:]
    else:
        psi = psi.T

    qmat1 = abs(polyval(fliplr([psi/sqrt(factorial(arange(0, n)))])[0], conjugate(alpha_mat))) ** 2;
    qmat1 = real(qmat1) * exp(-abs(alpha_mat)**2) / pi;

    return qmat1





