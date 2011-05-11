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
    D=(alpha*a-conj(alpha)*a.dag()).expm()
    return D*x


def coherent_dm(N, alpha):
    """
    Generate the density matrix representation of a coherent state
    @param N the number of states
    @param alpha the coherent state amplitude (complex)
    """
    psi = coherent(N,alpha)
    return psi * psi.dag()


def fock_dm(N, m):
    """
    Generate the density matrix for a fock state

    @param N the number of states
    @param m the fock state number
    """
    psi = basis(N, m)
    return psi * psi.dag()


def thermal_dm(N, n):
    """
    Generates the density matrix for a thermal state of n particles

    @param N: the number of states
    @param n: expectational value for number of particles in thermal state
    """

    i=arange(N)  
    rm = diag((1.0+n)**(-1.0)*(n/(1.0+n))**(i)) #populates diagonal terms (the only nonzero terms in matrix)
    return Qobj(rm)


if __name__ == "__main__":
    print (coherent(5,.1)*coherent(5,.1).dag()).tr()
    

