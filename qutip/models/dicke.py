"""
Dynamics for dicke states exploiting permutational invariance
"""
from math import factorial
from decimal import Decimal
import numpy as np

from scipy import constants
from scipy.integrate import odeint, ode
from scipy.sparse import dok_matrix, csr_matrix, block_diag
from qutip import Qobj, spre, spost
from qutip import sigmax, sigmay, sigmaz, sigmap, sigmam
from qutip.solver import Result, Options
from qutip import *

from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.mesolve import _generic_ode_solve
from piqs.cy.dicke import Dicke as _Dicke
from piqs.cy.dicke import Pim as _Pim
from piqs.cy.dicke import (_j_min, _j_vals, _get_blocks, 
                      jmm1_dictionary, _num_dicke_states,
                      _num_dicke_ladders)
# ============================================================================
# Functions necessary to generate the Lindbladian/Liouvillian for j, m, m1
# ============================================================================
def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems.

    Parameters
    -------
    N: int
        The number of two level systems
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    if (not float(N).is_integer()):
        raise ValueError("Number of TLS should be an integer")

    if (N < 1):
        raise ValueError("Number of TLS should be non-negative")

    nds = (N / 2 + 1)**2 - (N % 2) / 4
    return int(nds)


def num_dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space for a
    collection of N two-level systems. It counts how many different "j" exist.
    Or the number of blocks in the block diagonal matrix.

    Parameters
    -------
    N: int
        The number of two level systems.
    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5
    return int(Nj)


def num_tls(nds):
    """
    The number of two level systems (TLS), given the number of Dicke states.
    Inverse function of _num_dicke_states(N)

    Parameters
    ----------
    nds: int
         The number of Dicke states
    Returns
    -------
    N: int
        The number of two level systems
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2 * (np.sqrt(nds) - 1)
    else:
        # N is odd
        N = 2 * (np.sqrt(nds + 1 / 4) - 1)

    return int(N)


def isdiagonal(matrix):
    """
    Check if a matrix is diagonal either if it is a Qobj or a ndarray
    """
    if isinstance(matrix, Qobj):
        matrix = matrix.full()

    isdiag = np.all(matrix == np.diag(np.diagonal(matrix)))

    return isdiag


def dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems. Same function, named "num_dicke_states",
    is present in the file with cythonized code, dicke.pyx. 

    Parameters
    -------
    N: int
        The number of two level systems
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    if (not float(N).is_integer()):
        raise ValueError("Number of TLS should be an integer")

    if (N < 1):
        raise ValueError("Number of TLS should be non-negative")

    nds = (N / 2 + 1)**2 - (N % 2) / 4

    return int(nds)

def dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space for a
    collection of N two-level systems. It counts how many different "j" exist.
    Or the number of blocks in the block diagonal matrix. Same function, 
    named "num_dicke_ladders", is present in the file with cythonized code, dicke.pyx.

    Parameters
    -------
    N: int
        The number of two level systems.
    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5

    return int(Nj)

class Piqs(object):
    """
    The Dicke States class.

    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2

    hamiltonian : Qobj matrix
        An Hamiltonian H in the reduced basis set by `reduced_algebra()`.
        Matrix dimensions are (nds, nds), with nds = _num_dicke_states.
        The hamiltonian is assumed to be with hbar = 1.
        default: H = jz_op(N)

    emission : float
        Incoherent emission coefficient (also nonradiative emission)
        default: 0.0

    dephasing : float
        Local dephasing coefficient
        default: 0.0

    pumping : float
        Incoherent pumping coefficient
        default: 0.0

    collective_emission : float
        Collective (superradiant) emmission coefficient
        default: 1.0

    collective_pumping : float
        Collective pumping coefficient
        default: 0.0

    collective_dephasing : float
        Collective dephasing coefficient
        default: 0.0
    nds : int
        The number of Dicke states
        default: nds(2) = 4

    dshape : tuple
        The tuple (nds, nds)
        default: (4,4)

    blocks : array
        A list which gets the number of cumulative elements at each block
        boundary
        default:  array([3, 4])
    """

    def __init__(self, N=1, hamiltonian=None,
                 emission=0., dephasing=0., pumping=0.,
                 collective_emission=0., collective_dephasing=0., collective_pumping=0.):
        self.N = N
        self.hamiltonian = hamiltonian

        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_emission = collective_emission
        self.collective_dephasing = collective_dephasing
        self.collective_pumping = collective_pumping
        
        self.nds = _num_dicke_states(self.N)
        self.dshape = (_num_dicke_states(self.N), _num_dicke_states(self.N))

    def __repr__(self):
        """
        Print the current parameters of the system.
        """
        string = []
        string.append("N = {}".format(self.N))
        string.append("Hilbert space dim = {}".format(self.dshape))
        string.append("collective_emission = {}".format(self.collective_emission))
        string.append("emission = {}".format(self.emission))
        string.append("dephasing = {}".format(self.dephasing))
        string.append("pumping = {}".format(self.pumping))
        string.append(
            "collective_dephasing = {}".format(
                self.collective_dephasing))
        string.append(
            "collective_pumping = {}".format(
                self.collective_pumping))

        return "\n".join(string)

    def lindbladian(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics as a
        sparse matrix using a Cythonized function

        Returns
        ----------
        lindbladian: Qobj superoperator (sparse)
                The matrix size is (nds**2, nds**2) where nds is the number of
                Dicke states.
        """
        cythonized_dicke = _Dicke(int(self.N),
                                float(self.emission),
                                float(self.dephasing),
                                float(self.pumping),
                                float(self.collective_emission),
                                float(self.collective_dephasing),
                                float(self.collective_pumping))

        return cythonized_dicke.lindbladian()

    def diagonal_lindbladian(self):
        """
        A faster implementation for diagonal Hamiltonians which only computes
        the Lindblad terms for the diagonal elements and hence is much faster.
        """
        system = _Pim(int(self.N), float(self.emission), float(self.dephasing),
              float(self.pumping), float(self.collective_emission),
              float(self.collective_dephasing), float(self.collective_pumping))

        M = system.generate_matrix()
        return M

    def liouvillian(self):
        """
        Gives the total liouvillian in the jmm1 basis |j, m > < j, m1|
        """

        lindblad = self.lindbladian()

        if self.hamiltonian is None:
            liouv = lindblad
        else:
            hamiltonian = self.hamiltonian
            hamiltonian_superoperator = - 1j * \
                spre(hamiltonian) + 1j * spost(hamiltonian)
            liouv = lindblad + hamiltonian_superoperator

        return liouv

    def solve(self, initial_state, tlist, options=None, progress_bar=None):
        """
        Solver for the Dicke model optimized with QuTiP's spmat functions
        """
        if isdiagonal(initial_state) and isdiagonal(self.hamiltonian):
            pass


    def eigenstates(self, liouvillian):
        """
        Calculates the eigenvalues and eigenvectors of the Liouvillian,
        removing the spurious ones of eigenvectors with corresponding
        non-hermitian density matrices.

        Parameters
        ----------
        liouvillian: Qobj superoperator type
            The Liouvillian of which to calculate the spectrum.

        Returns
        -------
        eigen_states: list of Qobj
            The list of eigenvalues and correspondent eigenstates.
        """
        unpruned_eigenstates = liouvillian.eigenstates()
        eigen_states = self.prune_eigenstates(unpruned_eigenstates)

        return eigen_states

    def prune_eigenstates(self, liouvillian_eigenstates):
        """
        Removes the spurious eigenvalues and eigenvectors of the Liouvillian.
        Spurious means that the given eigenvector has elements outside of the
        block diagonal matrix.

        Parameters
        ----------
        liouvillian_eigenstates: list of Qobj
            A list with the eigenvalues and eigenvectors of the Liouvillian
            including spurious ones.

        Returns
        -------
        correct_eigenstates: list of Qobj
            The list with the correct eigenvalues and eigenvectors of the
            Liouvillian.
        """

        N = self.N
        block_mat = block_matrix(N)
        nnz_tuple_bm = [(i, j) for i, j in zip(*block_mat.nonzero())]

        # 0. Create  a copy of the eigenvalues to approximate values
        eig_val, eig_vec = liouvillian_eigenstates
        tol = 10
        eig_val_round = np.round(eig_val, tol)

        # 2. Use 'block_matrix(N)' to remove eigenvectors with matrix
        # elements
        # outside of the block matrix.
        forbidden_eig_index = []
        for k in range(0, len(eig_vec)):
            dm = vector_to_operator(eig_vec[k])
            nnz_tuple = [(i, j) for i, j in zip(*dm.data.nonzero())]
            for i in nnz_tuple:
                if i not in nnz_tuple_bm:
                    if np.round(dm[i], tol) != 0:
                        forbidden_eig_index.append(k)

        forbidden_eig_index = np.array(list(set(forbidden_eig_index)))
        # 3. Remove the forbidden eigenvalues and eigenvectors.
        correct_eig_val = np.delete(eig_val, forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec, forbidden_eig_index)
        correct_eigenstates = correct_eig_val, correct_eig_vec

        return correct_eigenstates


# ============================================================================
# Utility functions for operators in the Dicke basis
# ============================================================================
def energy_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m, alpha> have the same energy
    (hbar * omega_0 * m) given N two-level systems.
    The use of the Decimals class allows to explore N > 1000,
    unlike the built-in function 'scipy.special.binom(N, N/2 + m)'

    Parameters
    ----------
    N: int
        The number of two level systems.
    m: float
        Total spin z-axis projection eigenvalue.
        This is proportional to the total energy)

    Returns
    -------
    degeneracy: int
        The energy degeneracy
    """
    numerator = Decimal(factorial(N))
    d1 = Decimal(factorial(N / 2 + m))
    d2 = Decimal(factorial(N / 2 - m))

    degeneracy = numerator / (d1 * d2)

    return int(degeneracy)

def state_degeneracy(N, j):
    """
    Calculates the degeneracy of the Dicke state |j, m>.
    Each state |j, m> includes D(N,j) irreducible representations |j, m,alpha>.
    Uses Decimals to calculate higher numerator and denominators numbers.

    Parameters
    ----------
    N: int
        The number of two level systems.

    j: float
        Total spin eigenvalue (cooperativity)

    Returns
    -------
    degeneracy: int
        The state degeneracy
    """
    numerator = Decimal(factorial(N)) * Decimal(2 * j + 1)
    denominator_1 = Decimal(factorial(N / 2 + j + 1))
    denominator_2 = Decimal(factorial(N / 2 - j))

    degeneracy = numerator / (denominator_1 * denominator_2)
    degeneracy = int(np.round(float(degeneracy)))

    if degeneracy < 0:
        raise ValueError("m-degeneracy must be >=0")

    return degeneracy

def m_degeneracy(N, m):
    """
    The number of Dicke states |j, m> with same energy (hbar * omega_0 * m)
    for N two-level systems.

    Parameters
    ----------
    N : int
        The number of two level systems
    m: float
        Total spin z-axis projection eigenvalue (proportional to the total
        energy)

    Returns
    -------
    degeneracy: int
        The m-degeneracy
    """
    degeneracy = N / 2 + 1 - abs(m)
    if degeneracy % 1 != 0 or degeneracy <= 0:
        e = "m-degeneracy must be integer >=0, "
        e.append("but degeneracy = {}".format(degeneracy))
        raise ValueError(e)

    return int(degeneracy)

def jx_op(N):
    """
    Builds the Jx operator in the same basis of the reduced density matrix
    rho(j,m,m').    

    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jx_operator: Qobj matrix
        The Jx operator as a QuTiP object. The dimensions are (nds,nds) where
        nds is the number of Dicke states.         
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    block_diagonal = block_matrix(N).todense()

    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
                s = s + 1
    jx_operator = 1/2*(jp_operator + jm_operator)

    return Qobj(jx_operator)

def jy_op(N):
    """
    Builds the Jy operator in the same basis of the reduced density matrix
    rho(j,m,m').    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jy_operator: Qobj matrix
        The Jy operator as a QuTiP object. The dimensions are (nds,nds) where
        nds is the number of Dicke states.    
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    block_diagonal = block_matrix(N).todense()
    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
                s = s + 1
    jy_operator = 1j/2*(jm_operator - jp_operator)

    return Qobj(jy_operator)

def jz_op(N):
    """
    Builds the Jz operator in the same basis of the reduced density matrix
    rho(j,m,m'). Jz is diagonal in this basis.   
    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    
    Returns
    -------
    jz_operator: Qobj matrix
        The Jz operator as a QuTiP object. The dimensions are (nds,nds)
        where nds is the number of Dicke states.      
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    jz_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                jz_operator[s,s] = m
                s = s + 1

    return Qobj(jz_operator)

def j2_op(N):
    """
    Builds the J^2 operator in the same basis of the reduced density matrix
    rho(j,m,m'). J^2 is diagonal in this basis.   
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    j2_operator: Qobj matrix
        The J^2 operator as a QuTiP object. The dimensions are (nds,nds) where
        nds is the number of Dicke states.  
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    j2_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                j2_operator[s,s] = j * (j + 1)
                s = s + 1

    return Qobj(j2_operator) 

def jp_op(N):
    """
    Builds the Jp operator in the same basis of the reduced density matrix
    rho(j,m,m').   
    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    
    Returns
    -------
    jp_operator: Qobj matrix
        The Jp operator as a QuTiP object. The dimensions are (nds,nds) where
        nds is the number of Dicke states.    
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    jp_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = ap(j,m-1)
                s = s + 1

    return Qobj(jp_operator)

def jm_op(N):
    """
    Builds the Jm operator in the same basis of the reduced density matrix rho(j,m,m').    

    Parameters
    ----------
    N: int 
        Number of two-level systems

    Returns
    -------
    jm_operator: Qobj matrix
        The Jm operator as a QuTiP object. The dimensions are (nds,nds) where nds is
        the number of Dicke states.    
    """
    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] = am(j,m+1)
                s = s + 1

    return Qobj(jm_operator)

def ap( j, m):
    """
    Calculate A_{+} for value of j, m.
    """
    a_plus = np.sqrt((j - m) * (j + m + 1))
    
    return(a_plus)

def am(j, m):
    """
    Calculate A_{m} for value of j, m.
    """
    a_minus = np.sqrt((j + m) * (j - m + 1))
    
    return(a_minus)

# ============================================================================
# Operators definitions in the uncoupled SU(2) basis used for comparison
# ============================================================================
def su2_algebra(N):
    """
    Creates the vector (sx, sy, sz, sm, sp) with the spin operators of a
    collection of N two-level systems (TLSs). Each element of the vector,
    i.e., sx, is a vector of Qobs objects (spin matrices), as it cointains the
    list of the SU(2) Pauli matrices for the N TLSs. Each TLS operator
    sx[i], with i = 0, ..., (N-1), is placed in a 2^N-dimensional
    Hilbert space.

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    su2_operators: list
        A list of Qobs matrices (Qutip objects) - [sx, sy, sz, sm, sp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    sx = [0 for i in range(N)]
    sy = [0 for i in range(N)]
    sz = [0 for i in range(N)]
    sm = [0 for i in range(N)]
    sp = [0 for i in range(N)]
    sx[0] = 0.5 * sigmax()
    sy[0] = 0.5 * sigmay()
    sz[0] = 0.5 * sigmaz()
    sm[0] = sigmam()
    sp[0] = sigmap()

    # 2. Place operators in total Hilbert space
    for k in range(N - 1):
        sx[0] = tensor(sx[0], identity(2))
        sy[0] = tensor(sy[0], identity(2))
        sz[0] = tensor(sz[0], identity(2))
        sm[0] = tensor(sm[0], identity(2))
        sp[0] = tensor(sp[0], identity(2))

    # 3. Cyclic sequence to create all N operators
    a = [i for i in range(N)]
    b = [[a[i - i2] for i in range(N)] for i2 in range(N)]

    # 4. Create N operators
    for i in range(1, N):
        sx[i] = sx[0].permute(b[i])
        sy[i] = sy[0].permute(b[i])
        sz[i] = sz[0].permute(b[i])
        sm[i] = sm[0].permute(b[i])
        sp[i] = sp[0].permute(b[i])

    su2_operators = [sx, sy, sz, sm, sp]

    return su2_operators

def collective_algebra(N):
    """
    Uses the module su2_algebra to create the collective spin algebra
    Jx, Jy, Jz, Jm, Jp in the uncoupled basis of the two-level system
    (TLS) SU(2) Pauli matrices. Each collective operator is placed in a
    Hilbert space of dimension 2^N.

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    collective_operators: vector of Qobs matrices (Qutip objects)
        collective_operators = [Jx, Jy, Jz, Jm, Jp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)

    si_TLS = su2_algebra(N)

    sx = si_TLS[0]
    sy = si_TLS[1]
    sz = si_TLS[2]
    sm = si_TLS[3]
    sp = si_TLS[4]

    jx = sum(sx)
    jy = sum(sy)
    jz = sum(sz)
    jm = sum(sm)
    jp = sum(sp)

    collective_operators = [jx, jy, jz, jm, jp]

    return collective_operators

def j_algebra(N):
    """
    Gives the list with the collective operators of the total algebra, using the reduced basis
    |j,m><j,m'| in which the density matrix is expressed.    
    The list returned is [J^2, J_x, J_y, J_z, J_+, J_-]. 
    Parameters
    ----------
    N: int 
        Number of two-level systems    
    Returns
    -------
    red_alg: list
        Each element of the list is a Qobj matrix (QuTiP class) of dimensions (nds,nds). nds = number of Dicke states.    
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)

    jz_operator = dok_matrix((nds, nds))
    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                jz_operator[s,s] = m
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] = am(j,m+1)
                s = s + 1
    jx_operator = 1/2*(jp_operator + jm_operator)
    jy_operator = 1j/2*(jm_operator - jp_operator)

    j_alg = [Qobj(jx_operator), Qobj(jy_operator), Qobj(jz_operator), Qobj(jp_operator), Qobj(jm_operator)]

    return j_alg


def c_ops_tls(N=2, emission=0., dephasing=0., pumping=0., collective_emission=0.,
              collective_dephasing=0., collective_pumping=0.):
    """
    Create the collapse operators (c_ops) of the Lindblad master equation in
    the in the uncoupled basis of the two-level system (TLS) SU(2) Pauli matrices. 
    The collapse operator list can be given to the Qutip algorithm 'mesolve'. 
    Notice that the operators are placed in a Hilbert space of dimension 2^N.
    Thus the method is suitable only for small N, N max of order 10.

    Parameters
    ----------
    N: int
        The number of two level systems
        default = 2
    emission: float
        default = 0
        incoherent emission coefficient (also nonradiative emission)
    dephasing: float
        default = 0
        Dephasing coefficient
    pumping: float
        default = 0
        Incoherent pumping coefficient
    collective_emission: float
        default = 0
        Collective (superradiant) emission coefficient
    collective_dephasing: float
        default = 0
        Collective dephasing coefficient
    collective_pumping: float
        default = 0
        Collective pumping coefficient

    Returns
    -------
    c_ops: c_ops vector of matrices
        c_ops contains the collapse operators for the Lindbla

    """
    N = int(N)

    if N > 10:
        print("""Warning! N > 10. dim(H) = 2^N. Use the permutational
              invariant methods for large N. """)

    [sx, sy, sz, sm, sp] = su2_algebra(N)
    [jx, jy, jz, jm, jp] = collective_algebra(N)

    c_ops = []

    if emission != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(emission) * sm[i])

    if dephasing != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(dephasing) * sz[i])

    if pumping != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(pumping) * sp[i])

    if collective_emission != 0:
        c_ops.append(np.sqrt(collective_emission) * jm)

    if collective_dephasing != 0:
        c_ops.append(np.sqrt(collective_dephasing) * jz)

    if collective_pumping != 0:
        c_ops.append(np.sqrt(collective_pumping) * jp)

    return c_ops

# ============================================================================
# State definitions in the Dicke basis with an option for basis transformation
# ============================================================================
def dicke_basis(N, jmm1 = None, basis = "dicke"):
    """
    Initialize the density matrix of a Dicke state. The default basis is "dicke", 
    which creates coefficients for each jmm1 value in the dictionary jmm1. 
    For instance, if we start from the most excited state for N = 2, 
    we have the following state represented as a
    density matrix of size (nds, nds) or (4, 4).

    1 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0 
    
    Similarly, if the state was rho0 = |1, 0> <1, 0| + |0, 0><0, 0|, the density
    matrix in the Dicke basis would be:

    0 0 0 0
    0 1 0 0
    0 0 0 0
    0 0 0 1

    The mapping for the (i, k) index of the density matrix to the |j, m>
    values is given by the cythonized function `jmm1_dictionary`.

    This function can thus be used to build arbitrary states in the Dicke
    basis.

    Parameters
    ==========
    N: int
        The number of two-level systems

    jmm1: dict
        A dictionary of {(j, m, m1): p} which gives the coefficient of the (j, m, m1)
        state in the density matrix.
    """
    if basis == "uncoupled":
        raise NotImplemented

    if jmm1 == None:
        msg = "Please specify the jmm1 values as a dictionary"
        msg += "or use the `excited(N)` function to create an"
        msg += "excited state where jmm1 = {(N/2, N/2, N/2): 1}"
        raise AttributeError(msg)

    nds = _num_dicke_states(N)
    rho = np.zeros((nds, nds))

    jmm1_dict = jmm1_dictionary(N)[1]

    for key in jmm1:
        i, k = jmm1_dict[key]
        rho[i, k] = jmm1[key]

    return Qobj(rho)

def dicke_state(N, j, m, basis = "dicke"):
    """
    Initialize the density matrix in the Dicke basis state. 
    For instance, if the superradiant state is given |j, m> = |1, 0> for N = 2, 
    the state is represented as a density matrix of size (nds, nds) or (4, 4),

    0 0 0 0
    0 1 0 0
    0 0 0 0
    0 0 0 0 

    Parameters
    ==========
    N: int
        The number of two-level systems

    j: float
        The eigenvalue j of the Dicke state |j, m>.

    m: float
        The eigenvalue m of the Dicke state |j, m>.

    """
    if basis == "uncoupled":
        raise NotImplemented

    nds = num_dicke_states(N)
    rho = np.zeros((nds, nds))

    jmm1_dict = jmm1_dictionary(N)[1]

    i, k = jmm1_dict[(j, m, m)]
    rho[i, k] = 1.

    return Qobj(rho) 


def excited(N, basis = "dicke"):
    """
    Generates the density matrix for the Dicke state |N/2, N/2>, default in the
    Dicke basis. If the argument `basis` is "uncoupled" then it generates
    the state in a 2**N dim Hilbert space.
    """
    if basis == "uncoupled":
        return _uncoupled_excited(N)

    jmm1 = {(N/2, N/2, N/2): 1}
    return dicke_basis(N, jmm1)

def superradiant(N, basis = "dicke"):
    """
    Generates the superradiant dicke state as |N/2, 0> or |N/2, 0.5>
    in the Dicke basis
    """
    if basis == "uncoupled":
        return _uncoupled_superradiant(N)

    if N%2 == 0:
        jmm1 = {(N/2, 0, 0):1.}
        return dicke(N, jmm1)
    else:
        jmm1 = {(N/2, 0.5, 0.5):1.}
    return dicke_basis(N, jmm1)

def css(N, a=1/np.sqrt(2), b=1/np.sqrt(2), basis = "dicke"):
    """
    Loads the separable spin state |CSS>= Prod_i^N(a|1>_i + b|0>_i)
    into the reduced density matrix rho(j,m,m'). 
    The default state is the symmetric CSS, |CSS> = |+>.
    """
    if basis == "uncoupled":
        return _uncoupled_css(N, a, b)

    nds = _num_dicke_states(N)
    num_ladders = _num_dicke_ladders(N)
    rho = dok_matrix((nds, nds))

    # loop in the allowed matrix elements
    jmm1_dict = jmm1_dictionary(N)[1]

    j = 0.5 * N 
    mmax = int(2 * j + 1)
    for i in range(0, mmax):
        m = j - i
        psi_m = np.sqrt(float(energy_degeneracy(N, m))) * a**( N * 0.5 + m) * b**( N * 0.5 - m)
        for i1 in range(0, mmax):
            m1 = j - i1
            row_column = jmm1_dict[(j, m, m1)]
            psi_m1 = np.sqrt(float(energy_degeneracy(N, m1))) * a**( N * 0.5 + m1) * b**( N * 0.5 - m1)
            rho[row_column] = psi_m * psi_m1
    
    return Qobj(rho)

def ghz(N, basis = "dicke"):
    """
    Generates the the density matric of the GHZ state
    """
    if basis == "uncoupled":
        return _uncoupled_ghz(N)

    nds = _num_dicke_states(N)
    rho = dok_matrix((nds, nds))

    rho[0,0] = 1/2
    rho[N,N] = 1/2
    rho[N,0] = 1/2
    rho[0,N] = 1/2

    return Qobj(rho)

def ground(N, basis = "dicke"):
    """
    Generates the density matric of the ground state for N spins
    """
    if basis == "uncoupled":
        return _uncoupled_ground(N)

    nds = _num_dicke_states(N)
    rho = dok_matrix((nds, nds))

    rho[-1, -1] = 1

    return Qobj(rho)

def uncoupled_identity(N):
    """
    Generates the identity in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    identity: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)
    rho = np.zeros((2**N, 2**N))
    for i in range(0, 2**N):
        rho[i, i] = 1

    spin_dim = [2 for i in range(0, N)]
    spins_dims = list((spin_dim, spin_dim))
    identity = Qobj(rho, dims = spins_dims)

    return identity

# Uncoupled states in the full Hilbert space.
# These functions will be accessed when the input to a state
# generation function is supplied with the parameter `basis`
# as "uncoupled". Otherwise, we work in the default Dicke
# basis
def _uncoupled_excited(N):
    """
    Generates a initial dicke state |N/2, N/2> as a Qobj in a 2**N
    dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)
    jz = collective_algebra(N)[2]
    en, vn = jz.eigenstates()
    psi0 = vn[2**N - 1]

    return psi0

def _uncoupled_superradiant(N):
    """
    Generates a initial dicke state |N/2, 0> (N even) or |N/2, 0.5> (N odd)
    as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)
    jz = collective_algebra(N)[2]
    en, vn = jz.eigenstates()
    psi0 = vn[2**N - N]

    return psi0

def _uncoupled_ground(N):
    """
    Generates a initial dicke state |N/2, - N/2> as a Qobj in a 2**N
    dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)
    jz = collective_algebra(N)[2]
    en, vn = jz.eigenstates()
    psi0 = vn[0]

    return psi0

def _uncoupled_ghz(N):
    """
    Generates the GHZ density matrix in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    ghz: Qobj matrix (QuTiP class) with the correct dimensions (dims)
    """
    N = int(N)

    rho = np.zeros((2**N, 2**N))
    rho[0, 0] = 1 / 2
    rho[2**N - 1, 0] = 1 / 2
    rho[0, 2**N - 1] = 1 / 2
    rho[2**N - 1, 2**N - 1] = 1 / 2

    spin_dim = [2 for i in range(0, N)]
    spins_dims = list((spin_dim, spin_dim))

    rho = Qobj(rho, dims=spins_dims)

    return rho

# This needs to be consistent to allow for coefficients a and b in the state
# as a|0> + b|1>
def _uncoupled_css(N, a, b):
    """
    Generates the CSS density matrix in a 2**N dimensional Hilbert space.
    The CSS states are non-entangled states of the form
    |a, b> =  \Prod_i (a|1>_i + b|0>_i).

    Parameters
    ----------
    N: int
        The number of two level systems
    a: complex
        The coefficient of the |1_i> state
    b: complex
        The coefficient of the |0_i> state
    Returns
    -------
    css: Qobj matrix (QuTiP class) with the correct dimensions (dims)
    """
    N = int(N)

    # 1. Define i_th factorized density matrix in the uncoupled basis
    rho_i = np.zeros((2,2), dtype=complex)
    rho_i[0,0] = a * np.conj(a)
    rho_i[1,1] = b * np.conj(b)
    rho_i[0,1] = a * np.conj(a)
    rho_i[1,0] = b * np.conj(b)
    rho_i = Qobj(rho_i)

    rho = [0 for i in range(N)]
    rho[0] = rho_i

    # 2. Place single-two-level-system density matrices in total Hilbert space
    for k in range(N - 1):
        rho[0] = tensor(rho[0], identity(2))

    # 3. Cyclic sequence to create all N factorized density matrices |CSS>_i<CSS|_i
    a = [i for i in range(N)]
    b = [[a[i - i2] for i in range(N)] for i2 in range(N)]

    # 4. Create all other N-1 factorized density matrices
    # |+><+| = Prod_(i=1)^N |CSS>_i<CSS|_i
    for i in range(1, N):
        rho[i] = rho[0].permute(b[i])

    identity_i = Qobj(np.eye(2**N), dims=rho[0].dims, shape=rho[0].shape)
    rho_tot = identity_i

    for i in range(0, N):
        rho_tot = rho_tot * rho[i]

    return rho_tot

def uncoupled_thermal(N, omega_0, temperature):
    """
    Gives the thermal state for a collection of N two-level systems with
    H = omega_0 * j_z. It is calculated in the full 2**N Hilbert state on the
    eigenstates of H in the uncoupled basis, not the Dicke basis.

    Parameters
    ----------
    N: int
        The number of two level systems

    omega_0: float
        The resonance frequency of each two-level system (homogeneous)

    temperature: float
        The absolute temperature in Kelvin

    Returns
    -------
    rho_thermal: Qobj operator
        The thermal state calculated in the full Hilbert space 2**N
    """
    N = int(N)
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)

    jz = collective_algebra(N)[2]
    m_list = jz.eigenstates()[0]
    m_list = np.flip(m_list, 0)

    rho_thermal = np.zeros(jz.shape)

    for i in range(jz.shape[0]):
        rho_thermal[i, i] = np.exp(- x * m_list[i])
    rho_thermal = Qobj(rho_thermal, dims=jz.dims, shape=jz.shape)

    zeta = uncoupled_partition_function(N, omega_0, temperature)
    rho_thermal = rho_thermal / zeta
    return rho_thermal


def uncoupled_partition_function(N, omega_0, temperature):
    """
    Gives the partition function for a collection of N two-level systems
    with H = omega_0 * j_z.
    It is calculated in the full 2**N Hilbert state, using the eigenstates of
    H in the uncoupled basis, not the Dicke basis.

    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin
    Returns
    -------
    zeta: float
        The partition function for the thermal state of H calculated summing
        over all 2**N states
    """
    N = int(N)
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)

    jz = collective_algebra(N)[2]
    m_list = jz.eigenstates()[0]

    zeta = 0
    for m in m_list:
        zeta = zeta + np.exp(- x * m)
    return zeta

def block_matrix(N):
    """
    Gives the block diagonal matrix filled with 1 if the matrix element
    is allowed in the reduced basis |j,m><j,m'|.   
    Parameters
    ----------
    N: int 
        Number of two-level systems
    
    Returns
    -------
    block_matr: ndarray
        A block diagonal matrix of ones with dimension (nds,nds), where
        nds is the number of Dicke states for N two-level systems.   
    """
    nds = _num_dicke_states(N)
    
    #create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N/2 + 1 - 0.5 *(N%2))
    blocks_list = [ (2*(i+1*(N%2))+1*((N+1)%2)) for i in range(blocks_dimensions)]
    blocks_list = np.flip(blocks_list,0)

    #create a list with each block matrix as element  

    square_blocks = []
    k = 0
    for i in blocks_list:
        square_blocks.append(np.ones((i,i)))
        k = k + 1

    return block_diag(square_blocks)
