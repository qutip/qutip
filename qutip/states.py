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
from Qobj import Qobj
from operators import destroy

def basis(N,*args):
    """
    @brief Generate the vector representation of a number state.
	
    a subtle incompability with the quantum optics toolbox: here
        basis(N, 0) = ground state
    but in QO toolbox:
        basis(N, 1) = ground state
	
    @param N the number of states
    @param args integer corresponding to desired number state
    
    @returns quantum object representing the requested number state |args>
    """
    if (not isinstance(N,int)) or N<0:#check if N is int and N>0
        raise ValueError("N must be integer N>=0")
    if not any(args):#if no args then assume vacuum state 
        args=0
    if not isinstance(args,int):#if input arg!=0
        if not isinstance(args[0],int):#check if args is not int
            raise ValueError("need integer for basis vector index")
        args=args[0]
    if args<0 or args>(N-1): #check if args is within bounds
        raise ValueError("basis vector index need to be in 0=<indx<=N-1")
    bas=zeros([N,1]) #column vector of zeros
    bas[args]=1 # 1 located at position args
    return Qobj(bas) #return Qobj


def coherent(N,alpha):
    """
    @brief Generates a coherent state with eigenvalue alpha in a 
        N-dimensional Hilbert space via displacement 
        operator on vacuum state
    
    @param N number of levels in truncated Hilbert space
    @param alpha eigenvalue for coherent state
    
    @returns Qobj quantum object for coherent state
    """
    x=basis(N,0)
    a=destroy(N)
    D=(alpha*a.dag()-conj(alpha)*a).expm()
    return D*x

def coherent_dm(N, alpha):
    """
    @brief Generate the density matrix representation of a coherent 
        state via outer product
    
    @param N number of levels in truncated Hilbert space
    @param alpha eigenvalue for coherent state
    
    @returns Qobj density matrix representation of coherent state
    """
    psi = coherent(N,alpha)
    return psi * psi.dag()

def coherent_fast(N,alpha):
    """
    Generate a coherent state	
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    data = zeros([N,1],dtype=complex)
    n = arange(N)
    data[:,0] = exp(-(abs(alpha)**2)/2.0)*(alpha**(n))/sqrt(factorial(n))
    return Qobj(data)

def coherent_dm_fast(N,alpha):
    """
    Generate a coherent state	
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    psi = coherent_fast(N, alpha)
    return psi * psi.dag()

def fock_dm(N, *args):
    """
    @brief Generate the density matrix representation of a Fock state via outer product.
    
    @param N number of levels in truncated Hilbert space
    @param m int corresponding to desired number state, defaults to 0 if omitted
    
    @returns Qobj density matrix representation of Fock state
    """
    if not args:
        psi=basis(N)
    else:
        psi=basis(N, args[0])
    return psi * psi.dag()

def fock(N, *args):
    """
    @brief Generates the vector representation of a bosonic Fock (number) state. 
        Same as `basis' function.
    
    @param N the number of states in the Hilbert space
    @param m int corresponding to desired number state, defaults to 0 if omitted
    
    @returns Qobj quantum object representing the requested number state |args>
    """
    if not args:
        return basis(N)
    else:
        return basis(N, args[0])

def thermal_dm(N, n):
    """
    Generates the density matrix for a thermal state of n particles

    @param N: the number of states
    @param n: expectational value for number of particles in thermal state
    """

    i=arange(N)  
    rm = diag((1.0+n)**(-1.0)*(n/(1.0+n))**(i)) #populates diagonal terms (the only nonzero terms in matrix)
    return Qobj(rm)


def ket2dm(Q):
    """
    @brief Takes input ket or bra vector and returns density matrix 
        formed by outer product.
    
    @param Q Ket or bra vector
    
    @returns Qobj Density matrix formed by outer product
    """
    if Q.type=='ket':
        out=Q*Q.dag()
    elif Q.type=='bra':
        out=Q.dag()*Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)















if __name__ == "__main__":
    print fock(5,1)
    print (coherent(5,.1)*coherent(5,.1).dag()).tr()
    print ket2dm(coherent(3,.1).dag())

