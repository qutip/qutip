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
    Generate the vector representation of a number state.
	
    a subtle incompability with the quantum optics toolbox: In QuTiP::
 
        basis(N, 0) = ground state

    but in QO toolbox::

        basis(N, 1) = ground state
	
    N the number of states
    args integer corresponding to desired number state
    
    Returns quantum object representing the requested number state ``|args>``
    
    Example::
        
        >>> basis(5,2)
        Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
        Qobj data = 
        [[ 0.+0.j]
         [ 0.+0.j]
         [ 1.+0.j]
         [ 0.+0.j]
         [ 0.+0.j]]
         
    """
    if (not isinstance(N,int)) or N<0:
        raise ValueError("N must be integer N>=0")
    if not any(args):#if no args then assume vacuum state 
        args=0
    if not isinstance(args,int):#if input arg!=0
        if not isinstance(args[0],int):
            raise ValueError("need integer for basis vector index")
        args=args[0]
    if args<0 or args>(N-1): #check if args is within bounds
        raise ValueError("basis vector index need to be in 0=<indx<=N-1")
    bas=zeros([N,1]) #column vector of zeros
    bas[args]=1 # 1 located at position args
    return Qobj(bas)


def qutrit_basis():
    """
    Return the basis states for a three level system (qutrit)
    
    Parameters None
    
    Returns *array* of qutrit basis vectors
    """
    return array([basis(3,0), basis(3,1), basis(3,2)])


def coherent(N,alpha):
    """
    Generates a coherent state with eigenvalue alpha in a 
    N-dimensional Hilbert space via displacement operator on vacuum state
    
    N number of levels in truncated Hilbert space
    alpha eigenvalue for coherent state
    
    Returns Qobj quantum object for coherent state
    
    Example::
        
        >>> coherent(5,0.25j)
        Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
        Qobj data = 
        [[  9.69233235e-01+0.j        ]
         [  0.00000000e+00+0.24230831j]
         [ -4.28344935e-02+0.j        ]
         [  0.00000000e+00-0.00618204j]
         [  7.80904967e-04+0.j        ]]
         
    """
    x=basis(N,0)
    a=destroy(N)
    D=(alpha*a.dag()-conj(alpha)*a).expm()
    return D*x

def coherent_dm(N, alpha):
    """
    Generate the density matrix representation of a coherent 
    state via outer product
    
    N number of levels in truncated Hilbert space
    alpha eigenvalue for coherent state
    
    Returns Qobj density matrix representation of coherent state
    
    Example::
    
        >>> coherent_dm(3,0.25j)
        Quantum object: dims = [[3], [3]], shape = [3, 3], type = oper, isHerm = True
        Qobj data = 
        [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
         [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
         [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j        ]]
         
    """
    psi = coherent(N,alpha)
    return psi * psi.dag()

def coherent_fast(N,alpha):
    """
    Generate a coherent state	

    N the number of states
    alpha the coherent state amplitude (complex)
    """
    data = zeros([N,1],dtype=complex)
    n = arange(N)
    data[:,0] = exp(-(abs(alpha)**2)/2.0)*(alpha**(n))/sqrt(factorial(n))
    return Qobj(data)

def coherent_dm_fast(N,alpha):
    """
    Generate a coherent state	

    N the number of states
    alpha the coherent state amplitude (complex)
    """
    psi = coherent_fast(N, alpha)
    return psi * psi.dag()

def fock_dm(N, *args):
    """
    Generate the density matrix representation of a Fock state via outer product.
    
    N number of levels in truncated Hilbert space
    m int corresponding to desired number state, defaults to 0 if omitted
    
    Returns Qobj density matrix representation of Fock state
    
    Example::
    
         >>> fock_dm(3,1)
         Quantum object: dims = [[3], [3]], shape = [3, 3], type = oper, isHerm = True
         Qobj data = 
         [[ 0.+0.j  0.+0.j  0.+0.j]
          [ 0.+0.j  1.+0.j  0.+0.j]
          [ 0.+0.j  0.+0.j  0.+0.j]]
    """
    if not args:
        psi=basis(N)
    else:
        psi=basis(N, args[0])
    return psi * psi.dag()

def fock(N, *args):
    """
    Generates the vector representation of a bosonic Fock (number) state. 
    Same as :func:`qutip.states.basis` function.
    
    Arguments:

        `N` (*int*): the number of states in the Hilbert space.
        
        `m` (*int*): corresponding to desired number state, defaults to 0 if omitted.
    
    Returns Qobj quantum object representing the requested number state :math:`\left|\mathrm{args}\\right>`.
    
    Example::
    
        >>> fock(4,3)
        Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
        Qobj data = 
        [[ 0.+0.j]
         [ 0.+0.j]
         [ 0.+0.j]
         [ 1.+0.j]]
    """
    if not args:
        return basis(N)
    else:
        return basis(N, args[0])

def thermal_dm(N, n):
    """
    Generates the density matrix for a thermal state of n particles

    N: the number of states
    n: expectational value for number of particles in thermal state
    
    Returns *Qobj* for thermal state
    
    Example::
        >>> thermal_dm(5,1)
        Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
        Qobj data = 
        [[ 0.50000+0.j  0.00000+0.j  0.00000+0.j  0.00000+0.j  0.00000+0.j]
         [ 0.00000+0.j  0.25000+0.j  0.00000+0.j  0.00000+0.j  0.00000+0.j]
         [ 0.00000+0.j  0.00000+0.j  0.12500+0.j  0.00000+0.j  0.00000+0.j]
         [ 0.00000+0.j  0.00000+0.j  0.00000+0.j  0.06250+0.j  0.00000+0.j]
         [ 0.00000+0.j  0.00000+0.j  0.00000+0.j  0.00000+0.j  0.03125+0.j]]
    
    """

    i=arange(N)  
    rm = diag((1.0+n)**(-1.0)*(n/(1.0+n))**(i)) #populates diagonal terms (the only nonzero terms in matrix)
    return Qobj(rm)


def ket2dm(Q):
    """
    Takes input ket or bra vector and returns density matrix 
    formed by outer product.
    
    Q Ket or bra vector
    
    Returns Qobj Density matrix formed by outer product
    
    Example::
    
        >>> x=basis(3,2)
        >>> ket2dm(x)
        Quantum object: dims = [[3], [3]], shape = [3, 3], type = oper, isHerm = True
        Qobj data = 
        [[ 0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j]]
         
    """
    if Q.type=='ket':
        out=Q*Q.dag()
    elif Q.type=='bra':
        out=Q.dag()*Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)




