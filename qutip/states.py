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
from basis import basis
from operators import destroy

def coherent(N,alpha):
    """
    Generate a coherent state
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    x=basis(N,0)
    a=destroy(N)
    D=(alpha*a.dag()-conj(alpha)*a).expm()
    return D*x

def coherent_dm(N, alpha):
    """
    Generate the density matrix representation of a coherent state
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    psi = coherent(N,alpha)
    return psi * psi.dag()

def coherent_fast(N,alpha):
    """
    Generate a coherent state	
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    data = zeros([N,1])
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

def fock_dm(N, m):
    """
    Generate the density matrix for a fock state

    @param N the number of states
    @param m the fock state number
    """
    psi = basis(N, m)
    return psi * psi.dag()

def fock(N, m):
    """
    Generate a state vector for a fock state

    @param N the number of states
    @param m the fock state number
    """
    return basis(N, m)

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
    Takes input ket or bra vector and returns density matrix formed by outer product.
    @param Q: Ket or bra vector
    @return dm: Density matrix formed by outer product
    """
    if Q.type=='ket':
        out=Q*Q.dag()
    elif Q.type=='bra':
        out=Q.dag()*Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)















if __name__ == "__main__":
    print (coherent(5,.1)*coherent(5,.1).dag()).tr()
    print ket2dm(coherent(3,.1).dag())

