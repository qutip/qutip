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
import os
from numpy import array
from scipy import *
from scipy.linalg import *
import scipy.sparse as sp

from Qobj import Qobj
from basis import qutrit_basis

#
# Spin opeators
#

def jmat(j,*args):
    """
    Higher-order spin operators
    
    Parameter j *float* spin
    Parameter args *str* which operator to return 'x','y','z','+','-'
    
    Returns *Qobj* spin operator
    """
    if (fix(2*j)!=2*j) or (j<0):
        raise TypeError('j must be a non-negative integer or half-integer')
    if not args:
        a1=Qobj(0.5*(jplus(j)+jplus(j).conj().T))
        a2=Qobj(0.5*1j*(jplus(j)-jplus(j).conj().T))
        a3=Qobj(jz(j))
        return array([a1,a2,a3])
    if args[0]=='+':
        A=jplus(j)
    elif args[0]=='-':
        A=jplus(j).conj().T
    elif args[0]=='x':
        A=0.5*(jplus(j)+jplus(j).conj().T)
    elif args[0]=='y':
        A=-0.5*1j*(jplus(j)-jplus(j).conj().T)
    elif args[0]=='z':
        A=jz(j)
    else:
        raise TypeError('Invlaid type')
    return Qobj(A.tocsr())


def jplus(j):
    m=arange(j,-j-1,-1)
    N=len(m)
    return sp.spdiags(sqrt(j*(j+1.0)-(m+1.0)*m),1,N,N,format='csr')


def jz(j):
    m=arange(j,-j-1,-1)
    N=len(m)
    return sp.spdiags(m,0,N,N,format='csr')

#
# Pauli spin 1/2 operators:
#

def sigmap():
    """
    creation operator for Pauli spins.
    """
    return jmat(1/2.,'+')

def sigmam():
    """
    annihilation operator for Pauli spins.
    """
    return jmat(1/2.,'-')

def sigmax():
    """
    Pauli spin 1/2 sigma x operator
    """
    return 2.0*jmat(1.0/2,'x')

def sigmay():
    """
    Pauli spin 1/2 sigma y operator
    """
    return 2.0*jmat(1.0/2,'y')

def sigmaz():
    """
    Pauli spin 1/2 sigma z operator
    """
    return 2.0*jmat(1.0/2,'z')



#
#DESTROY returns annihilation operator for N dimensional Hilbert space
# out = destroy(N), N is integer value &  N>0
#
def destroy(N):
    '''
    Destruction (lowering) operator
    
    Parameter N *int* dimension of hilbert space
    
    Returns *Qobj*
    '''
    if not isinstance(N,int):#raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    return Qobj(sp.spdiags(sqrt(range(0,N)),1,N,N,format='csr'))

#
#CREATE returns creation operator for N dimensional Hilbert space
# out = create(N), N is integer value &  N>0
#
def create(N):
    '''
    Creation (raising) operator
    
    Parameter N *int* dimension of hilbert space
    
    Returns *Qobj*
    '''
    if not isinstance(N,int):#raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo=destroy(N) #create operator using destroy function
    qo.data=qo.data.T #transpsoe data in Qobj
    return Qobj(qo)


#
#QEYE returns identity operator for an N dimensional space
# a = qeye(N), N is integer & N>0
#
def qeye(N):
    """
    Identity operator
    
    Parameter N *int* dimension of hilbert space
    
    Returns *Qobj*
    """
    N=int(N)
    if (not isinstance(N,int)) or N<0:#check if N is int and N>0
        raise ValueError("N must be integer N>=0")
    return Qobj(sp.eye(N,N,dtype=complex,format='csr'))
    

def num(N):
    '''
    Number operator
    
    Parameter N *int* dimension of hilbert space
    
    Returns *Qobj*
    '''
    data=sp.spdiags(arange(N),0,N,N,format='csr')
    return Qobj(data)


def squeez(N,sp):
    '''
    single-mode Squeezing operator
    
    Parameter N *int* dimension of hilbert space
    Parameter sp *real* or *complex* squeezing parameter
    
    Returns *Qobj*
    '''
    a=destroy(N)
    op=(1/2.0)*conj(sp)*(a**2)-(1/2.0)*sp*(a.dag())**2
    return op.expm()


def displace(N,alpha):
    '''
    Single-mode displacement operator
    
    Parameter N *int* dimension of hilbert space
    Parameter alpha *real* or *complex* displacment amplitude
    
    Returns *Qobj*
    '''
    a=destroy(N)
    D=(alpha*a.dag()-conj(alpha)*a).expm()
    return D

#
# Three-level operators (qutrits)
#
def qutrit_ops():
    ''' 
    Return the operators for a three level system (qutrit)
    
    Parameters None
    
    Returns *array* of qutrit operators
    '''
    one, two, three = qutrit_basis()
    sig11 = one * one.dag()
    sig22 = two * two.dag()
    sig33 = three * three.dag()
    sig12 = one * two.dag()
    sig23 = two * three.dag()
    sig31 = three * one.dag()
    return array([sig11, sig22, sig33, sig12, sig23, sig31])
