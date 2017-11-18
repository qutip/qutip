"""
Generates a Lindblad superoperator for a Dicke system
"""
from math import factorial
from decimal import Decimal

import numpy as np

from scipy.integrate import odeint, ode

from scipy import constants
from scipy.sparse import *
from qutip import Qobj, spre, spost
from qutip.solver import Result


def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems.

    Parameters
    -------
    N: int
        The number of two level systems.    
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    nds = (N/2 + 1)**2 - (N % 2)/4
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

def num_two_level(nds):
    """
    The number of two level systems (TLS), given the number of Dicke states. 
    Inverse function of num_dicke_states(N)
    
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
        N = 2 * (np.sqrt(nds + 1/4) - 1)
    
    return int(N)

def get_blocks(N):
        """
        A list which gets the number of cumulative elements at each block boundary.

        For N = 4

        1 1 1 1 1 
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1 
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Thus, the blocks are [5, 8, 9] denoting that after the first block 5 elements
        have been accounted for and so on. This function will later be helpful in the
        calculation of j, m, m' value for a given (row, col) index in this matrix.

        Returns
        -------
        blocks: arr
            An array with the number of cumulative elements at the boundary of each block
        """
        num_blocks = num_dicke_ladders(N)
        blocks = np.array([i * (N + 2 - i) for i in range(1, num_blocks + 1)], dtype = int)
        return blocks
    
class Dicke(object):
    """
    The Dicke States class.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
        
    emission : float
        Collective spontaneous emmission coefficient
        default: 1.0

    hamiltonian : Qobj matrix
        An Hamiltonian H in the reduced basis set by `reduced_algebra()`. 
        Matrix dimensions are (nds, nds), with nds = num_dicke_states.
        The hamiltonian is assumed to be with hbar = 1. 
        default: H = jz_op(N)
                
    loss : float
        Incoherent loss coefficient
        default: 0.0
        
    dephasing : float
        Local dephasing coefficient
        default: 0.0
        
    pumping : float
        Incoherent pumping coefficient
        default: 0.0
    
    collective_pumping : float
        Collective pumping coefficient
        default: 0.0

    collective_dephasing : float
        Collective dephasing coefficient
        default: 0.0
    """
    def __init__(self, N = 2, hamiltonian = None,
                 loss = 0., dephasing = 0., pumping = 0., emission = 1.,
                 collective_pumping = 0., collective_dephasing = 0.):
        self.N = N
        if hamiltonian == None:
            self.hamiltonian = jz_op(N)
        else :
            self.hamiltonian = hamiltonian
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        self.blocks = get_blocks(N)
        self.nds = num_dicke_states(N)
        self.dshape = (num_dicke_states(N), num_dicke_states(N))
        
        self.blocks = get_blocks(self.N)
        
    def j_vals(self):
        """
        Get the valid values of j for given N.
        """
        j = np.arange(j_min(self.N), self.N/2 + 1)
        return j
    
    def m_vals(self, j):
        """
        Get all the possible values of m or $m^\prime$ for given j.
        """
        return np.arange(-j, j+1)
    
    def get_index(self, jmm1):
        """
        Get the index in the density matrix for this j, m, m1 value.
        """
        j, m, m1 = jmm1
        _k = j - m1
        _k_prime = j - m

        blocks = self.blocks
        block_number = int(self.N/2 - j)

        offset = 0
        if block_number > 0:
            offset = blocks[block_number - 1]

        i = _k_prime + offset
        k = _k + offset

        return (int(i), int(k))
        
    def jmm1_flat(self):
        """
        A dictionary with keys: (l) and values: (j, m, m1) for a block-diagonal flattened matrix
        in the |j, m> <j, m1| basis. l is the position of the particular jmm1 in the flattened vector.
        """
        N = self.N
        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)
        
        jmm1_flat = {}
        
        # loop in the allowed matrix elements
        for j in self.j_vals():
            for m in self.m_vals(j):
                for m1 in self.m_vals(j):
                    jmm1 = (j, m, m1)
                    i, k = self.get_index(jmm1)
                    l = nds * i  + k
                    jmm1_flat[l] = jmm1

        return jmm1_flat
    
    def _get_element_flat(self, jmm):
        """
        Get the (l) index for given tuple (j, m, m1) from the flattened block diagonal matrix.
        """
        i, k = self.get_index(jmm)
        nds = num_dicke_states(self.N)
        
        l = nds * i + k
        
        return l
        
    def lindblad_sup(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics as a sparse matrix using COO.

        Returns
        ----------
        lindblad_qobj: Qobj superoperator (sparse)
            The matrix size is (nds**2, nds**2) where nds is the number of Dicke states.

        """
        N = self.N
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        jmm1_row = self.jmm1_flat()

        S = dok_matrix((nds**2, nds**2))

        # perform loop in each row of matrix 
        for r in jmm1_row:
            j, m, m1 = jmm1_row[r]
            jmm1_1 = (j, m, m1)
            jmm1_2 = (j, m+1, m1+1)
            jmm1_3 = (j+1, m+1, m1+1)
            jmm1_4 = (j-1, m+1, m1+1)
            jmm1_5 = (j+1, m, m1)
            jmm1_6 = (j-1, m, m1)
            jmm1_7 = (j+1, m-1, m1-1)
            jmm1_8 = (j, m-1, m1-1)
            jmm1_9 = (j-1, m-1, m1-1)
            
            t1 = self.tau1(jmm1_1)
            c1 = self._get_element_flat(jmm1_1)
            
            S[r, c1] = t1

            # generate taus in the given row
            # checking if the taus exist
            # and load taus in the lindbladian in the correct position

            if jmm1_2 in jmm1_row.values():
                t2 = self.tau2(jmm1_2)
                c2 = self._get_element_flat(jmm1_2)
                S[r, c2] = t2

            if jmm1_3 in jmm1_row.values():
                t3 = self.tau3(jmm1_3)
                c3 = self._get_element_flat(jmm1_3)
                S[r, c3] = t3

            if jmm1_4 in jmm1_row.values():
                t4 = self.tau4(jmm1_4)
                c4 = self._get_element_flat(jmm1_4)
                S[r, c4]

            if jmm1_5 in jmm1_row.values():
                t5 = self.tau5(jmm1_5)
                c5 = self._get_element_flat(jmm1_5)
                S[r, c5] = t5

            if jmm1_6 in jmm1_row.values():
                t6 = self.tau6(jmm1_6)
                c6 = self._get_element_flat(jmm1_6)
                S[r, c6] = t6

            if jmm1_7 in jmm1_row.values():                
                t7 = self.tau7(jmm1_7)
                c7 = self._get_element_flat(jmm1_7)
                S[r, c7] = t7 

            if jmm1_8 in jmm1_row.values():
                t8 = self.tau8(jmm1_8)
                c8 = self._get_element_flat(jmm1_8)
                S[r, c8] = t8

            if jmm1_9 in jmm1_row.values():
                t9 = self.tau9(jmm1_9)
                c9 = self._get_element_flat(jmm1_9)
                S[r, c9] = t9

        #convert matrix into CSR sparse
        lindblad_matrix = S.tocsr()

        #make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]],[[nds], [nds]]]
        lindblad_qobj = Qobj(lindblad_matrix, dims = llind_dims)

        return lindblad_matrix
        
    def tau1(self, jmm1):
        """
        Calculate tau1 for value of j, m, m'
        """
        j, m, m1 = jmm1
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping
        yCD = self.collective_dephasing
        
        N = self.N  
        N = float(N)

        spontaneous = yS / 2 * (2 * j * (j + 1) - m * (m - 1) - m1 * (m1 - 1))
        losses = yL/2 * (N + m + m1)
        pump = yP/2 * (N - m - m1)
        collective_pump = yCP / 2 * (2 * j * (j + 1) - m * (m + 1) - m1 * (m1 + 1))
        collective_dephase = yCD / 2 * (m - m1)**2
        
        if j <= 0:
            dephase = yD * N/4
        else :
            dephase = yD/2 * (N/2 - m * m1 * (N/2 + 1)/ j /(j + 1))

        t1 = spontaneous + losses + pump + dephase + collective_pump + collective_dephase
        
        return(-t1)
    
    def tau2(self, jmm1):
        """
        Calculate tau2 for given j, m, m'
        """
        j, m, m1 = jmm1
        yS = self.emission
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if yS == 0:
            spontaneous = 0.0
        else:            
            spontaneous = yS * np.sqrt((j + m) * (j - m + 1)* (j + m1) * (j - m1 + 1))

        if (yL == 0) or (j <= 0):
            losses = 0.0
        else:            
            losses = yL / 2 * np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1)) * (N/2 + 1) / (j * (j + 1))

        t2 = spontaneous + losses

        return (t2)
    
    def tau3(self, jmm1):
        """
        Calculate tau3 for given j, m, m'
        """
        j, m, m1 = jmm1
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0) or (j <= 0) :
            t3 = 0.0
        else:
            t3 = yL / 2 * np.sqrt((j + m) * (j + m - 1) * (j + m1) * (j + m1 - 1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t3)
    
    def tau4(self, jmm1):
        """
        Calculate tau4 for given j, m, m'
        """
        j, m, m1 = jmm1
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0)  or ( (j + 1) <= 0):
            t4 = 0.0
        else:
            t4 = yL / 2 * np.sqrt((j - m + 1) * (j - m + 2) * (j - m1 + 1) * (j - m1 + 2)) * (N/2 - j )/((j + 1)* (2 * j + 1))

        return (t4)
    
    def tau5(self, jmm1):
        """
        Calculate tau5 for given j, m, m'
        """
        j, m, m1 = jmm1
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if (yD == 0)  or (j <= 0):
            t5 = 0.0
        else:                    
            t5 = yD / 2 * np.sqrt((j**2 - m**2) * (j**2 - m1**2))* (N/2 + j + 1) / (j * (2 * j + 1))

        return (t5)
    
    def tau6(self, jmm1):
        """
        Calculate tau6 for given j, m, m'
        """
        j, m, m1 = jmm1
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if yD == 0:
            t6 = 0.0
        else:            
            t6 = yD / 2 * np.sqrt(((j + 1)**2 - m**2) * ((j + 1)**2 - m1**2)) * (N/2 - j )/((j + 1) * (2 * j + 1))

        return (t6)
    
    def tau7(self, jmm1):
        """
        Calculate tau7 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            t7 = 0.0
        else:    
            t7 = yP / 2 * np.sqrt((j - m - 1) * (j - m)* (j - m1 - 1) * (j - m1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t7)
    
    def tau8(self, jmm1):
        """
        Calculate tau8 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            pump = 0.0
        else:    
            pump = yP / 2 * np.sqrt((j + m + 1) * (j - m) * (j + m1 + 1) * (j - m1)) * (N/2 + 1) / (j * (j + 1))

        if yCP == 0:
            collective_pump = 0.0
        else:    
            collective_pump = yCP * np.sqrt((j - m) * (j + m + 1) * (j + m1 + 1) * (j - m1))
        
        t8 = pump + collective_pump
        
        return (t8)
    
    def tau9(self, jmm1):
        """
        Calculate tau9 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0):
            t9 = 0.0
        else:    
            t9 = yP / 2 * np.sqrt((j + m + 1) * (j + m + 2) *(j + m1 + 1) * (j + m1 + 2)) * (N/2 - j )/((j + 1)*(2 * j + 1))

        return (t9)
