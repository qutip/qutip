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
from scipy import *
from qutip.Qobj import Qobj
from qutip.operators import destroy
import scipy.sparse as sp

def basis(N,*args):
    """Generates the vector representation of a Fock state.
	
    Parameters
    ----------
    N : int 
        Number of Fock states in Hilbert space.
    
    args : int 
        ``int`` corresponding to desired number state, defaults
        to 0 if omitted.
    
    Returns
    -------
    state : qobj
      Qobj representing the requested number state ``|args>``.
    
    Examples
    --------        
    >>> basis(5,2)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data = 
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]]
         
    Notes
    -----
    
    A subtle incompability with the quantum optics toolbox: In QuTiP::
 
        basis(N, 0) = ground state

    but in qotoolbox::

        basis(N, 1) = ground state
         
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
    bas=sp.lil_matrix((N,1)) #column vector of zeros
    bas[args,0]=1 # 1 located at position args
    bas=bas.tocsr()
    return Qobj(bas)


def qutrit_basis():
    """Basis states for a three level system (qutrit)
    
    Returns
    -------
    qstates : array
        Array of qutrit basis vectors
    
    """
    return array([basis(3,0), basis(3,1), basis(3,2)])


def coherent(N,alpha):
    """Generates a coherent state with eigenvalue alpha. 
    
    Constructed using displacement operator on vacuum state.
    
    Parameters
    ----------
    N : int 
        Number of Fock states in Hilbert space.
        
    alpha : float/complex 
        Eigenvalue of coherent state.
    
    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state
    
    Examples
    --------        
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
    """Density matrix representation of a coherent state.
    
    Constructed via outer product of :func:`qutip.states.coherent`
    
    Parameters
    ----------
    N : int 
        Number of Fock states in Hilbert space.
    
    alpha : float/complex
        Eigenvalue for coherent state.
    
    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.
    
    Examples
    --------    
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], shape = [3, 3], type = oper, isHerm = True
    Qobj data = 
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j        ]]
         
    """
    psi = coherent(N,alpha)
    return psi * psi.dag()


def fock_dm(N, *args):
    """Density matrix representation of a Fock state 
    
    Constructed via outer product of :func:`qutip.states.fock`.
    
    Parameters
    ----------
    N : int 
        Number of Fock states in Hilbert space.
    
    m : int 
        ``int`` for desired number state, defaults to 0 if omitted.
    
    Returns
    -------
    dm : qobj
        Density matrix representation of Fock state.
    
    Examples
    --------    
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
    """Bosonic Fock (number) state. 
    
    Same as :func:`qutip.states.basis`.
    
    Parameters
    ----------
    N : int 
        Number of states in the Hilbert space.
        
    m : int
        ``int`` for desired number state, defaults to 0 if omitted.
    
    Returns
    -------
        Requested number state :math:`\left|\mathrm{args}\\right>`.
    
    Examples
    --------    
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
    """Density matrix for a thermal state of n particles

    Parameters
    ----------
    N : int 
        Number of basis states in Hilbert space.
    
    n : float 
        Expectation value for number of particles in thermal state.
    
    Returns
    -------
    dm : qobj
        Thermal state density matrix.
    
    Examples
    --------
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
    rm = sp.spdiags((1.0+n)**(-1.0)*(n/(1.0+n))**(i),0,N,N,format='csr') #populates diagonal terms (the only nonzero terms in matrix)
    return Qobj(rm)


def ket2dm(Q):
    """Takes input ket or bra vector and returns density matrix 
    formed by outer product.
    
    Parameters
    ----------
    Q : qobj    
        Ket or bra type quantum object.
    
    Returns
    -------
    dm : qobj    
        Density matrix formed by outer product of `Q`.
    
    Examples
    --------    
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


#
# projection operator
#
def projection(N, n, m):
    ''' The projection operator that projects state |m> on state |n>.
    
    i.e., |n><m|.
    
    Returns
    -------
    oper : qobj
         Requested projection operator.
    '''
    ket1 = basis(N, n)
    ket2 = basis(N, m)
    
    return ket1 * ket2.dag()

#
# quantum state number helper functions
#
def state_number_enumerate(dims, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print state
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    """
        
    if state is None:
        state = zeros(len(dims))
        
    if idx == len(dims):
        yield array(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, state, idx+1):
                yield s
                
def state_number_index(dims, state):
    """
    Return the index of a quantum state corresponding to state,
    given a system with dimensions given by dims. 

    Example:
        
        >>> state_index([2,2,2], [1,1,0])
        6.0

    """
    return sum([state[i] * prod(dims[i+1:]) for i, d in enumerate(dims)])

def state_number_qobj(dims, state):
    """
    Return a Qobj representation of a quantum state specified by the state
    array `state`.

    Example:
        
        >>> state_number_qobj([2,2,2], [1,0,1])
        Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
        Qobj data =
        [[ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 1.]
         [ 0.]
         [ 0.]]
    """
    return tensor([fock(dims[i], s) for i, s in enumerate(state)])

